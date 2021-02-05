import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.cuda.amp import autocast

import utils
from models.hifi_gan import Generator
from models.wavenet import WaveNet

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Use HiFiGAN')
    parser.add_argument('--input', required=True, type=str, help='file or folder to run inference on')
    parser.add_argument('--output_dir', required=True, type=str, help='dir to save results in')
    parser.add_argument('--checkpoint', required=True, type=str, help='path to checkpoint file')
    parser.add_argument('--device', default='cpu',
                        choices=['cpu'] + [f'cuda:{d}' for d in range(torch.cuda.device_count())], type=str,
                        help='device to perform inference with')
    parser.add_argument('--hparams', type=str, help='path to hparams.py file')
    args = parser.parse_args()

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
            
            #we noticed some tenors in y are not 2D and it caused torch.cat issues at postprocess_inference_data
            for i in range(len(y)):
                if len(y[i].shape) < 2:
                    y[i] = y[i].unsqueeze(0)
                
            y = utils.data.postprocess_inference_data(y, hp.inference.batched, hp.dsp.sample_rate)
            y = y[:target_length].detach().cpu().numpy()
            sf.write(os.path.join(args.output_dir, f'{filename}_denoised.wav'), y.astype(np.float32),
                     samplerate=hp.dsp.sample_rate)
