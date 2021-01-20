import os
import time
from importlib import util

import pandas as pd
import torch
import torch.distributed

import utils.s3_utils as s3_utils
 

def import_module(path):
    spec = util.spec_from_file_location('hparams', path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_run_dir(process_group, rank):
    # Makes sure that only one process creates folders in multi-gpu setup
    if rank == 0:
        now = time.strftime("%Y-%m-%d__%H_%M_%S", time.localtime())
        run_dir = os.path.join('runs', now)
        os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)
        with open('tmp', 'w') as fp:
            fp.write(run_dir)

    torch.distributed.barrier(process_group)

    # Syncs run directory between processes
    with open('tmp', 'r') as fp:
        run_dir = fp.readline()
    if rank == 0:
        os.remove('tmp')

    return run_dir


def save_checkpoint(run_dir, model, optimizer, scaler, phase, step):
    pt = {
        'model_state_dict': model.state_dict(),
        'generator_state_dict': model.generator.state_dict(),
        'optimizer_generator_state_dict': optimizer['generator'].state_dict(),
        'optimizer_discriminator_state_dict': optimizer['discriminator'].state_dict(),
        'scaler_generator_state_dict': scaler['generator'].state_dict() if scaler['generator'] is not None else None,
        'scaler_discriminator_state_dict': scaler['discriminator'].state_dict() if scaler[
                                                                                       'discriminator'] is not None else None,
        'n_conditioning_dims': model.generator.wavenet.n_conditioning_dims,
        'phase': phase,
        'step': step,
    }
    torch.save(pt, os.path.join(run_dir, 'checkpoints/latest_checkpoint.pt'))
    torch.save(pt, os.path.join(run_dir, f'checkpoints/checkpoint_{phase}_{step}.pt'))


def ddp(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def dir_walk(path, ext):
    file_list = [os.path.join(root, name)
                 for root, dirs, files in os.walk(path)
                 for name in sorted(files)
                 if name.endswith(ext)
                 and not name.startswith('.')]
    return file_list


def parse_data_structure(path):
    # Checks if supplied path points to dir or dataframe
    assert path != '', 'Please set your data paths in hparams.py.'
    if s3_utils.is_path_existed(path):
        files = s3_utils.load_file_paths(path)
        df = pd.DataFrame(files, columns=['path'])
        return df
    elif path.endswith('.pkl'):
        df = pd.read_pickle(path)
        if not isinstance(df, pd.DataFrame):
            raise IOError(f'"{path}" does not point to a valid pandas DataFrame.')
        if 'path' not in df.columns:
            raise Exception(f'DataFrame at "{path}" does not contain required column "path".')
        return df
    else:
        raise IOError(f'Argument  path  must point to directory or pickled DataFrame, but points to "{path}".')


def list_into_dict(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = [value]
    else:
        dictionary[key].append(value)


def all_reduce(tensor, group):
    torch.distributed.all_reduce(tensor, group=group)
    return tensor / torch.cuda.device_count()
