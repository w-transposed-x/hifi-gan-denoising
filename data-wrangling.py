#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:14:15 2021

@author: Wilson Ye, Matt Linder
"""
# Data wrangling for DAPS, RIRs, and Noise datsets for HiFi-GAN

import argparse
import os
import shutil
import csv
from datetime import datetime
import utils.s3_utils as s3_utils
import utils.metrics as metrics

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.cuda.amp import autocast

import utils
from models.hifi_gan import Generator
from models.wavenet import WaveNet


def prepare_local_inference_data(source, destination):
    folders = os.listdir(source)
    folders = [f for f in folders if f.startswith('i')]
    print(len(folders))
    
    for folder in folders:
        path = os.path.join(source, folder)
        for file in os.listdir(path):
            if ("script5" in file):
                origin = os.path.join(path, file)
                shutil.copyfile(origin, destination + '/' + file)
                 

def prepare_s3_inference_data(source, destination): 
    count = 0
    folders = s3_utils.load_folder_paths_from_path(source)
    folders = [f for f in folders if 'iphone' in f or 'ipad' in f]

    for folder in folders:
        for file in s3_utils.load_file_paths(folder):
            if ("script5" in file):
                destination_path = destination + file.split('/')[-1]
                print(destination_path + '\n')
                s3_utils.copy_objects(file, destination_path)
                count += 1
    print(f"total files copied into destination folder: {count}")
    

def inference(args):

    # Import hparams
    if args.hparams is None:
        from hparams import hparams as hp
    else:
        hp = utils.core.import_module(args.hparams)

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Initializing model, optimizer, criterion and scaler
    model = Generator(wavenet=WaveNet())
    model.to(args.device)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.eval()

    if os.path.isdir(args.input):
        inference_files = utils.core.dir_walk(args.input, ('.wav', '.mp3', '.ogg'))
    elif os.path.isfile(args.input):
        inference_files = [args.input]
    else:
        raise Exception('input must be .wav file or dir containing audio files.')

    with torch.no_grad():
        for file in inference_files:
            filename = os.path.splitext(os.path.split(file)[1])[0]
            x, _ = librosa.load(file, sr=hp.dsp.sample_rate, mono=True)
            target_length = len(x)
            x = torch.tensor(x).to(args.device)
            x = utils.data.preprocess_inference_data(x,
                                                     hp.inference.batched,
                                                     hp.inference.batch_size,
                                                     hp.inference.sequence_length,
                                                     hp.dsp.sample_rate)

            with autocast(enabled=hp.training.mixed_precision):
                y = [model.inference(x_batch) for x_batch in x]
            y = utils.data.postprocess_inference_data(y, hp.inference.batched, hp.dsp.sample_rate)
            y = y[:target_length].detach().cpu().numpy()
            sf.write(os.path.join(args.output_dir, f'{filename}_denoised.wav'), y.astype(np.float32),
                     samplerate=hp.dsp.sample_rate)

def compute_metrics(original, inference):
    
    fields = ['file', 'pesq', 'stoi']
    results = []
    orig_files = os.listdir(original)
    
    for f in orig_files:
        orig_path = os.path.join(original, f)
        infer_path = os.path.join(inference, f.replace('.wav', '_denoised.wav'))
        snd_orig, sr_0 = librosa.load(orig_path, sr=16000)
        snd_denoise, sr_1 = librosa.load(infer_path, sr=16000)    
        pesq = metrics.pesq_score(snd_orig, snd_denoise, samplerate=16000)
        stoi = metrics.stoi_score(snd_orig, snd_denoise, samplerate=16000)
        result = (f, pesq, stoi)
        results.append(result)

    now = datetime.now()
    output_filename = 'metrics-' + now.strftime("%m/%d/%Y-%H:%M:%S")
    with open(output_filename, 'w') as f: 
      
        # using csv.writer method from CSV package 
        write = csv.writer(f) 
          
        write.writerow(fields) 
        write.writerows(results) 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make Audio Inference')
    
    #prepare inference data
    parser.add_argument('--prepare_inference_data', required=False, type=str, default=False, help='prepare inference data')
    parser.add_argument('--prepare_inference_data_source', required=False, type=str, help='prepare inference data source folder')
    parser.add_argument('--prepare_inference_data_destination', required=False, type=str, help='prepare inference data destination folder')
    
    
    #execute inference
    parser.add_argument('--run_inference', required=False, type=str, default=False, help='run inference')
    parser.add_argument('--input', required=False, type=str, help='file or folder to run inference on')
    parser.add_argument('--output_dir', required=False, type=str, help='dir to save results in')
    parser.add_argument('--checkpoint', required=False, type=str, help='path to checkpoint file')
    parser.add_argument('--device', default='cpu',
                        choices=['cpu'] + [f'cuda:{d}' for d in range(torch.cuda.device_count())], type=str,
                        help='device to perform inference with')
    parser.add_argument('--hparams', type=str, help='path to hparams.py file')
    
    #compute metrics
    parser.add_argument('--compute_metrics', required=False, type=str, default=False, help='compute metrics')
    parser.add_argument('--compute_metrics_data_original', required=False, type=str, help='computer metrics original original folder')
    parser.add_argument('--compute_metrics_data_inference', required=False, type=str, help='compute metrics inference data folder')
    
    
    args = parser.parse_args()
    

    
    #prepare inference data argument check
    if (args.prepare_inference_data == True and (args.prepare_inference_data_source == None or args.prepare_inference_data_destination == None)):
        parser.error("--prepare_inference_data requires --prepare_inference_data_source and --prepare_inference_data_destination.")
    
    if args.prepare_inference_data:    
        prepare_s3_inference_data(args.prepare_inference_data_source, args.prepare_inference_data_destination)
    
    
    #execute inference argument check
    if (args.run_inference == True and (args.input == None or args.input == None or args.output_dir == None)):
        parser.error("--run_inference requires other input")
        
    if args.run_inference:
        inference(args)
        
    
    #compute metrics
    if (args.compute_metrics == True and (args.compute_metrics_data_original == None or args.compute_metrics_data_inference == None)):
        parser.error("--compute_metrics requires --compute_metrics_data_original and --compute_metrics_data_inference.")
    
    if args.compute_metrics:    
        compute_metrics(args.compute_metrics_data_original, args.compute_metrics_data_inference)


        