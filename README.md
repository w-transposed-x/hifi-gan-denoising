# HiFi-GAN: High-Fidelity Denoising and Dereverberation Based on Speech Deep Features in Adversarial Networks

This is an **unofficial PyTorch implementation**
of [the above mentioned paper by Su et al. (2020)](https://arxiv.org/abs/2006.05694).

## Requirements

librosa 0.8.0  
numpy 1.18.1  
pandas 1.0.1  
scipy 1.4.1  
soundfile 0.10.3  
torch 1.6.0  
torchaudio 0.6.0  
tqdm 4.54.1

## Usage

### Data

Data for training can be supplied in several ways. In hparams.py (hparams.files), you can specify paths to your data. In
all cases, paths must point to either a directory containing audio files (.wav) or a .pkl file of a Pandas Dataframe.
All audio data should have a sample rate of 16kHz or above.  
**In the case of specifying directories**, files can directly be contained in the specified directory.  
**In the case specifying .pkl files**, the Dataframe for speakers, IRs and noises must contain a column labeled `path`,
with paths to audio files as its rows.

### Training

Training can be performed on multiple GPUs. Run

`python -m torch.distributed.launch --nproc_per_node=<DEVICE_COUNT> train.py [--checkpoint]`

in the command line, replacing `<DEVICE_COUNT>` with the number of CUDA devices in your system, optionally providing the
path to a checkpoint file when resuming training from an earlier checkpoint.  
You can monitor training using [tensorboard](https://pytorch.org/docs/stable/tensorboard.html). Pass the path
to `runs/<RUN_DIR>/logs` as the `--logdir` parameter.

### Inference

Run

`python inference.py --checkpoint <CHECKPOINT> --input <INPUT> --output_dir <OUTPUT_DIR> [--device <DEVICE>] [--hparams <HPARAMS>]`

in the command line and replace `<CHECKPOINT>` with the path to a checkpoint file, `<INPUT>` with the path to either a
single audio file or a directory of audio files you wish to perform inference on and `<OUTPUT_DIR>` with the path to the
desired directory to store outputs in (will be created automatically). Optionally, specify a `<DEVICE>` to run inference
on (e.g. `cpu` or `cuda:0`) and/or the path to a `<HPARAMS>` file if you want to use hparams other than the ones
specified in hparams.py.

## Experiences

In our experiments, we've not yet been able to reproduce the results reported in the original paper in terms of the
prediction's subjectively perceived audio quality. For training, we used the following datasets:

### Speaker data

[Nautilus Speaker Characterization (NSC) Corpus](https://www.qu.tu-berlin.de/menue/forschung/abgeschlossene_projekte/nsc_corpus/)
. We chose NSC over [DAPS](https://ccrma.stanford.edu/~gautham/Site/daps.html) (the dataset used in the original paper)
since NSC features 300 individual speakers compared to DAPS's 20. Also, for our application, German speakers are
preferable for training.

### IR data

We were unable to relaibly perform the RT60 augmentation described by [Bryan (2019)](https://arxiv.org/abs/1909.03642).
To ensure enough variety in IR data, we instead used a selection from a series of IR datasets, resulting in a custom
collection of ~100,000 individual IRs.

### Noise data

As described in the original paper, we used noise data from the REVERB Challenge database, contained in
the [Room Impulse Response and Noise Database ](https://www.openslr.org/28/).






