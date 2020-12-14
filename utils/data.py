from functools import partial

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

import utils as utils
from hparams import hparams as hp


class ConvDataset(IterableDataset):
    def __init__(self, sp_files, ir_files, noise_files, augmentation, validation=False):
        self.sp_files = sp_files
        self.ir_files = ir_files
        self.noise_files = noise_files
        self.augmentation = augmentation
        self.validation = validation
        self.n_conditioning_dims = hp.model.wavenet.n_conditioning_dims

    def __iter__(self):
        rng = np.random.default_rng()

        while True:
            # Files
            sp_idx = rng.integers(len(self.sp_files))
            sp_file = self.sp_files.iloc[sp_idx]['path']
            ir_idx = rng.integers(len(self.ir_files))
            ir_file = self.ir_files.iloc[ir_idx]['path']
            noise_idx = rng.integers(len(self.noise_files))
            noise_file = self.noise_files.iloc[noise_idx]['path']

            # Load and augment audio
            x, h, noise = self.load_and_augment(sp_file, ir_file, noise_file, self.augmentation)

            # Perform convolution
            y = self.convolve(x, h)

            # Add noise
            y = self.add_noise(y, noise)

            # Pad target to be of same length as input
            x = np.pad(x, (0, len(y) - len(x)), 'constant')

            yield x, y, sp_file, ir_file

    def load_and_augment(self, sp_file, ir_file, noise_file, augmentation):
        rng = np.random.default_rng()

        sp_audio = utils.dsp.load_audio(sp_file, max_seconds=3)
        sp_audio = utils.dsp.trim_speaker(sp_audio)
        if augmentation['speaker']:
            sp_audio = self.augment_sp(sp_audio)

        rnd = rng.random()
        if rnd > hp.training.chance_for_no_reverb or self.validation:
            ir_audio = utils.dsp.load_audio(ir_file)
            ir_audio = utils.dsp.trim_ir(ir_audio, hp.dsp.sample_rate)
            if augmentation['ir']:
                # IR augmentation is not to specification, too many unknowns regarding T60 augmentation.
                # See comment in utils.dsp.apply_t60_augmentation() and project README.
                ir_audio = self.augment_ir(ir_audio)
        else:
            ir_audio = np.array([1.0])
        rnd = rng.random()
        if rnd > hp.training.chance_for_no_noise or self.validation:
            noise_audio = utils.dsp.load_audio(noise_file)
            # Augments noise in all phases of training, paper was a little fuzzy regarding the exact augment scheme
            if augmentation['noise']:
                noise_audio = self.augment_noise(noise_audio)
        else:
            noise_audio = None

        return sp_audio, ir_audio, noise_audio

    def convolve(self, x, h):
        y = np.convolve(x, h)
        return librosa.util.normalize(y) * librosa.db_to_amplitude(hp.dsp.max_vol)

    def speaker_conditioning(self, label):
        # Convert speaker label (str) to binary vector to serve as speaker conditioning
        binary = np.asarray([int(i) for i in bin(label)[2:]])
        return np.pad(binary, (self.n_conditioning_dims - len(binary), 0), 'constant')

    def augment_sp(self, x):
        # Resamples speaker audio to vary speed if speaker conditioning is all zero.
        # Introduces pitch shift, paper wasn't clear on whether they keep the pitch fixed.
        # Also applies random gain factor.
        rng = np.random.default_rng()
        resample_factor = rng.uniform(low=hp.augmentation.sp_resample_factor_bounds[0],
                                      high=hp.augmentation.sp_resample_factor_bounds[1])
        x = librosa.core.resample(x, hp.dsp.sample_rate, hp.dsp.sample_rate * resample_factor)
        gain_factor = rng.uniform(low=hp.augmentation.sp_gain_factor_bounds[0],
                                  high=hp.augmentation.sp_gain_factor_bounds[0])
        return x * gain_factor

    def augment_ir(self, h):
        rng = np.random.default_rng()

        # DRR augmentation according to Bryan (2019)
        random_drr = rng.uniform(low=hp.augmentation.ir_drr_bounds_db[0],
                                 high=hp.augmentation.ir_drr_bounds_db[1])
        h = utils.dsp.apply_drr_augmentation(h, random_drr)

        # T60 augmentation according to Bryan (2019) does not work reliably
        # h = utils.dsp.apply_t60_augmentation(h, random_t60)

        # Random band gain factors sampled from normal distribution for gentle EQing,
        # sampling from uniform distribution seemed excessive. No details were provided in the paper.
        distrib = partial(rng.normal,
                          loc=hp.augmentation.ir_rand_eq_mean_std_db[0],
                          scale=hp.augmentation.ir_rand_eq_mean_std_db[1])
        h = utils.filter.random_eq(h,
                                   fs=hp.dsp.sample_rate,
                                   fraction=hp.augmentation.rand_eq_filter_fraction,
                                   order=hp.augmentation.rand_eq_filter_order,
                                   limits_freq=hp.augmentation.rand_eq_limits_freq,
                                   distribtion=distrib)
        return h

    def augment_noise(self, noise):
        # Random multi-band filtering the noise, no details were provided in the paper.
        rng = np.random.default_rng()
        distrib = partial(rng.uniform,
                          low=hp.augmentation.noise_rand_eq_limits_db[0],
                          high=hp.augmentation.noise_rand_eq_limits_db[1])
        return utils.filter.random_eq(noise,
                                      fs=hp.dsp.sample_rate,
                                      fraction=hp.augmentation.rand_eq_filter_fraction,
                                      order=hp.augmentation.rand_eq_filter_order,
                                      limits_freq=hp.augmentation.rand_eq_limits_freq,
                                      distribtion=distrib)

    def add_noise(self, y, noise):
        rng = np.random.default_rng()

        if noise is None:
            return y

        if len(noise) < len(y):
            n_reps = int(np.ceil(len(y) / len(noise)))
            noise = np.tile(noise, n_reps)

        max_idx = np.max((len(noise) - len(y), 1))
        idx = rng.integers(max_idx)
        noise = noise[idx:idx + len(y)]

        y_rms = np.std(y)
        noise_rms = np.std(noise)
        SNR = y_rms / (noise_rms + 1e-8)

        random_SNR = librosa.db_to_amplitude(rng.uniform(
            low=hp.augmentation.noise_rand_snr_bounds_db[0],
            high=hp.augmentation.noise_rand_snr_bounds_db[1]))
        factor = random_SNR / SNR
        noise /= factor
        return librosa.util.normalize(y + noise)


def collate(batch):
    # Cuts sequences of equal length from convolved signals and returns tensors
    rng = np.random.default_rng()
    max_offsets = [
        x if x > 0 else 0
        for x in [(len(x[0]) - hp.training.sequence_length) for x in batch]
    ]
    offsets = [rng.integers(0, offset + 1) for offset in max_offsets]
    ground_truth = [
        pad(x[0][offsets[i]:offsets[i] + hp.training.sequence_length])
        for i, x in enumerate(batch)
    ]
    inputs = [
        pad(x[1][offsets[i]:offsets[i] + hp.training.sequence_length])
        for i, x in enumerate(batch)
    ]

    ground_truth = np.stack(ground_truth).astype(np.float32)
    inputs = np.stack(inputs).astype(np.float32)

    ground_truth = torch.tensor(ground_truth)
    inputs = torch.tensor(inputs)

    return inputs, ground_truth


def pad(x):
    diff = hp.training.sequence_length - len(x)
    if diff > 0:
        x = np.pad(x, (0, diff), mode='constant')
    return x


def preprocess_inference_data(x, batched, batch_size, sequence_length, sample_rate):
    # Reshapes inference signal into batch for batched inference with 5 ms overlap between folds
    if not batched or len(x) <= sequence_length:
        return [x]
    else:
        overlap = int(0.005 * sample_rate)  # 5ms overlap between folds
        hop_size = int(sequence_length - overlap)
        num_folds = int(1 + np.ceil((len(x) - sequence_length) / hop_size))
        pad_len = int((sequence_length + (num_folds - 1) * hop_size) - len(x))
        x = F.pad(x, (0, pad_len))
        folds = [x[i * hop_size:i * hop_size + sequence_length] for i in range(num_folds)]
        return [torch.stack(folds[i:i + batch_size]) for i in range(0, len(folds), batch_size)]


def postprocess_inference_data(y, batched, sample_rate):
    # Reshapes batch into output signal
    if not batched:
        return y[0]
    else:
        y = torch.cat(y, dim=0)
        sequence_length = y.shape[1]
        overlap = int(0.005 * sample_rate)  # 5ms overlap between folds
        hop_size = int(sequence_length - overlap)
        t = np.linspace(-1, 1, overlap, dtype=np.float64)
        fade_in = torch.tensor(np.sqrt(0.5 * (1 + t))).to(y.device)
        fade_out = torch.tensor(np.sqrt(0.5 * (1 - t))).to(y.device)
        y[1:, :overlap] *= fade_in
        y[:-1, -overlap:] *= fade_out
        unfolded = torch.zeros(sequence_length + (y.shape[0] - 1) * hop_size).to(y.device)
        for i in range(y.shape[0]):
            start = i * hop_size
            unfolded[start:start + sequence_length] += y[i]
        return unfolded
