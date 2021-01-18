import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from hparams import hparams as hp
from logs import Logger
from models.hifi_gan import HiFiGAN
from models.wavenet import WaveNet
from utils.data import ConvDataset, collate
from utils.loss import CombinedLoss


def validation(model, criterion, valid_loader, process_group, current_phase):
    validation_loss = dict()
    audio_data = dict()

    model.eval()
    with torch.no_grad():
        with tqdm(desc='Valid', total=hp.training.n_validation_steps) as pbar:
            for inputs, ground_truth in valid_loader:

                if pbar.n >= hp.training.n_validation_steps:
                    break

                inputs = inputs.to(args.local_rank, non_blocking=True)
                ground_truth = ground_truth.to(args.local_rank, non_blocking=True)

                with autocast(enabled=hp.training.mixed_precision):
                    if current_phase == 0:

                        prediction = utils.core.ddp(model).generator.wavenet(inputs)
                        wavenet_loss = criterion.sample_loss(ground_truth, prediction) + criterion.spectrogram_loss(
                            ground_truth, prediction)

                        wavenet_loss = utils.core.all_reduce(wavenet_loss, group=process_group)

                        utils.core.list_into_dict(validation_loss, 'wavenet', wavenet_loss.item())

                        if pbar.n == hp.training.n_validation_steps - 1:
                            audio_data['input'] = inputs.detach().cpu().numpy()
                            audio_data['ground_truth'] = ground_truth.detach().cpu().numpy()
                            audio_data['wavenet'] = prediction.detach().cpu().numpy()

                    elif current_phase == 1:

                        prediction, prediction_postnet = utils.core.ddp(model).generator(inputs)
                        wavenet_loss = criterion.sample_loss(ground_truth, prediction) + criterion.spectrogram_loss(
                            ground_truth, prediction)
                        wavenet_postnet_loss = criterion.sample_loss(ground_truth, prediction_postnet) \
                                               + criterion.spectrogram_loss(ground_truth, prediction_postnet)

                        wavenet_loss = utils.core.all_reduce(wavenet_loss, group=process_group)
                        wavenet_postnet_loss = utils.core.all_reduce(wavenet_postnet_loss, group=process_group)

                        utils.core.list_into_dict(validation_loss, 'wavenet', wavenet_loss.item())
                        utils.core.list_into_dict(validation_loss, 'wavenet-postnet', wavenet_postnet_loss.item())

                        if pbar.n == hp.training.n_validation_steps - 1:
                            audio_data['input'] = inputs.detach().cpu().numpy()
                            audio_data['ground_truth'] = ground_truth.detach().cpu().numpy()
                            audio_data['wavenet'] = prediction.detach().cpu().numpy()
                            audio_data['wavenet-postnet'] = prediction_postnet.detach().cpu().numpy()

                    else:

                        prediction, prediction_postnet, prediction_scores, \
                        discriminator_scores, L_FM_G = model(inputs, ground_truth)

                        _, wavenet_loss, wavenet_postnet_loss, \
                        G_loss, D_losses = criterion(pbar.n, ground_truth, prediction, prediction_postnet,
                                                     prediction_scores,
                                                     discriminator_scores, L_FM_G)

                        wavenet_loss = utils.core.all_reduce(wavenet_loss, group=process_group)
                        wavenet_postnet_loss = utils.core.all_reduce(wavenet_postnet_loss, group=process_group)
                        G_loss = utils.core.all_reduce(G_loss, group=process_group)
                        D_losses = [utils.core.all_reduce(D_loss, group=process_group) for D_loss in D_losses]

                        if G_loss is not None:
                            utils.core.list_into_dict(validation_loss, 'wavenet', wavenet_loss.item())
                            utils.core.list_into_dict(validation_loss, 'wavenet-postnet', wavenet_postnet_loss.item())
                            utils.core.list_into_dict(validation_loss, 'G', G_loss.item())
                        utils.core.list_into_dict(validation_loss, 'D_16kHz', D_losses[0].item())
                        utils.core.list_into_dict(validation_loss, 'D_8kHz', D_losses[1].item())
                        utils.core.list_into_dict(validation_loss, 'D_4kHz', D_losses[2].item())
                        utils.core.list_into_dict(validation_loss, 'D_mel', D_losses[3].item())

                        if pbar.n == hp.training.n_validation_steps - 1:
                            audio_data['input'] = inputs.detach().cpu().numpy()
                            audio_data['ground_truth'] = ground_truth.detach().cpu().numpy()
                            audio_data['wavenet'] = prediction.detach().cpu().numpy()
                            audio_data['wavenet-postnet'] = prediction_postnet.detach().cpu().numpy()

                    pbar.set_postfix(losses={key: value[-1] for key, value in validation_loss.items()})
                    pbar.update()

        for key, value in validation_loss.items():
            validation_loss[key] = np.mean(value)

        return validation_loss, audio_data


def training(model, optimizer, criterion, scaler, logger, process_group, run_dir):
    global phase, step


    for current_phase, phase_params in hp.training.scheme.items():
        if current_phase < phase:
            continue
        step_offset = sum([params['steps'] for i, params in hp.training.scheme.items() if i < current_phase])

        # Update learning rate
        optimizer.param_groups[0]['lr'] = phase_params['lr_wavenet']
        if phase_params['lr_wavenet-postnet'] is not None:
            optimizer.param_groups[1]['lr'] = phase_params['lr_wavenet-postnet']
        if phase_params['lr_discriminator'] is not None:
            optimizer.param_groups[2]['lr'] = phase_params['lr_discriminator']

        # Initialize data loaders
        train_data = ConvDataset(sp_files=utils.core.parse_data_structure(hp.files.train_speaker),
                                 ir_files=utils.core.parse_data_structure(hp.files.train_ir),
                                 noise_files=utils.core.parse_data_structure(hp.files.train_noise),
                                 augmentation=phase_params['augmentation'],
                                 validation=False)
        valid_data = ConvDataset(
            sp_files=utils.core.parse_data_structure(hp.files.valid_speaker),
            ir_files=utils.core.parse_data_structure(hp.files.valid_ir),
            noise_files=utils.core.parse_data_structure(hp.files.valid_noise),
            augmentation=phase_params['augmentation'],
            validation=True)
        train_loader = DataLoader(dataset=train_data,
                                  collate_fn=collate,
                                  batch_size=phase_params['batch_size'],
                                  num_workers=hp.training.num_workers if not args.dev else 0,
                                  pin_memory=True)
        valid_loader = DataLoader(dataset=valid_data,
                                  collate_fn=collate,
                                  batch_size=phase_params['batch_size'],
                                  num_workers=hp.training.num_workers if not args.dev else 0,
                                  pin_memory=False)

        with tqdm(desc=f'Train {phase_params["modules"]}', total=phase_params['steps']) as pbar:
            pbar.update(step)
            for inputs, ground_truth in train_loader:

                model.train()

                if pbar.n >= phase_params['steps']:
                    break

                inputs = inputs.to(args.local_rank, non_blocking=True)
                ground_truth = ground_truth.to(args.local_rank, non_blocking=True)

                training_loss = dict()

                with autocast(enabled=hp.training.mixed_precision):
                    if current_phase == 0:

                        prediction = utils.core.ddp(model).generator.wavenet(inputs)
                        loss = criterion.sample_loss(ground_truth, prediction) + criterion.spectrogram_loss(
                            ground_truth, prediction)

                        training_loss['wavenet'] = loss.item()

                    elif current_phase == 1:

                        prediction, prediction_postnet = utils.core.ddp(model).generator(inputs)
                        wavenet_loss = criterion.sample_loss(ground_truth, prediction) + criterion.spectrogram_loss(
                            ground_truth, prediction)
                        wavenet_postnet_loss = criterion.sample_loss(ground_truth, prediction_postnet) \
                                               + criterion.spectrogram_loss(ground_truth, prediction_postnet)
                        loss = wavenet_loss + wavenet_postnet_loss

                        training_loss['wavenet'] = wavenet_loss.item()
                        training_loss['wavenet-postnet'] = wavenet_postnet_loss.item()

                    else:

                        prediction, prediction_postnet, prediction_scores, \
                        discriminator_scores, L_FM_G = model(inputs, ground_truth)

                        loss, wavenet_loss, wavenet_postnet_loss, \
                        G_loss, D_losses = criterion(pbar.n, ground_truth, prediction, prediction_postnet,
                                                     prediction_scores,
                                                     discriminator_scores, L_FM_G)

                        if G_loss is not None:
                            training_loss['wavenet'] = wavenet_loss.item()
                            training_loss['wavenet-postnet'] = wavenet_postnet_loss.item()
                            training_loss['G'] = G_loss.item()
                        training_loss['D_16kHz'] = D_losses[0].item()
                        training_loss['D_8kHz'] = D_losses[1].item()
                        training_loss['D_4kHz'] = D_losses[2].item()
                        training_loss['D_mel'] = D_losses[3].item()

                if not args.dev:
                    loss = utils.core.all_reduce(loss, group=process_group)

                optimizer.zero_grad()

                if hp.training.mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if hp.training.mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                pbar.set_postfix(loss=training_loss)
                pbar.update()
                step = pbar.n

                if args.local_rank == 0:
                    logger.log_training(pbar.n + step_offset, {
                        'training.loss': training_loss,
                    })

                if pbar.n % hp.training.validation_every_n_steps == 0:
                    validation_loss, audio_data = validation(model, criterion, valid_loader,
                                                             process_group, current_phase)

                    if args.local_rank == 0:
                        logger.log_validation(model=utils.core.ddp(model),
                                              step=pbar.n + step_offset,
                                              scalars={'validation.loss': validation_loss},
                                              audio_data=audio_data)
                        utils.core.save_checkpoint(run_dir, utils.core.ddp(model), optimizer, scaler, current_phase,
                                                   pbar.n)

        if current_phase < 2:
            phase = current_phase + 1
            step = 0


if __name__ == '__main__':
    # Run 'python -m torch.distributed.launch --nproc_per_node=<DEVICE_COUNT> train.py [--checkpoint]' in command line.

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train HiFiGAN')
    parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint file to continue training from.')
    parser.add_argument('--local_rank', default=0, type=int, help='Is set automatically by torch.distributed.launch')
    parser.add_argument('--dev', action='store_true', help='Run script without multi-GPU for debugging.')
    args = parser.parse_args()
    if args.dev:
        args.local_rank = 0

    # DDP setup
    torch.cuda.set_device(args.local_rank)
    if not args.dev:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        process_group = torch.distributed.new_group([i for i in range(torch.cuda.device_count())], backend='nccl')
    else:
        process_group = None

    # Initializing model, optimizer, criterion and scaler
    model = HiFiGAN(generator=WaveNet())
    model.cuda(args.local_rank)
    optimizer = torch.optim.Adam([
        {'params': model.generator.wavenet.parameters()},
        {'params': model.generator.postnet.parameters()},
        {'params': model.discriminators.parameters()}
    ])
    criterion = CombinedLoss(args.local_rank)
    scaler = torch.cuda.amp.GradScaler() if hp.training.mixed_precision else None

    # Wrap model in DDP
    if not args.dev:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    # Load parameters from checkpoint to resume training
    if args.checkpoint is not None:
        run_dir = os.path.dirname(os.path.dirname(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=f'cuda:{args.local_rank}')
        utils.core.ddp(model).load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        phase = checkpoint['phase']
        step = checkpoint['step']
    else:
        phase = 0
        step = 0
        run_dir = utils.core.get_run_dir(process_group, args.local_rank, args.dev)

    # Initializing logger
    logger = Logger(os.path.join(run_dir, 'logs')) if args.local_rank == 0 else None

    # Auto select best algorithm to maximize GPU utilization
    cudnn.benchmark = True

    # Start main loop
    try:
        training(model, optimizer, criterion, scaler, logger, process_group, run_dir)
    except KeyboardInterrupt:
        pass
    finally:
        if args.local_rank == 0:
            utils.core.save_checkpoint(run_dir, utils.core.ddp(model), optimizer, scaler, phase, step)
            print('Saved checkpoint.')
