import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from hparams import hparams as hp


class PostNet(nn.Module):
    def __init__(self):
        super(PostNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(hp.model.postnet.n_layers):
            in_channels = 1 if i == 0 else hp.model.postnet.n_channels
            out_channels = hp.model.postnet.n_channels if i != (hp.model.postnet.n_layers - 1) else 1
            self.layers.append(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=hp.model.postnet.kernel_size,
                          padding=hp.model.postnet.kernel_size // 2,
                          bias=True)
            )

    def forward(self, prediction):
        for layer in self.layers:
            prediction = torch.tanh(layer(prediction))
        return prediction


class WaveGAN(nn.Module):
    def __init__(self, resample_freq):
        super(WaveGAN, self).__init__()
        self.resample = torchaudio.transforms.Resample(orig_freq=hp.dsp.sample_rate,
                                                       new_freq=resample_freq)
        self.convs = nn.ModuleList()
        self.convs.append(
            nn.Conv1d(in_channels=1,
                      out_channels=hp.model.wavegan.in_conv_n_channels,
                      kernel_size=hp.model.wavegan.in_conv_kernel_size,
                      bias=True)
        )
        for i in range(hp.model.wavegan.strided_conv_n_layers):
            in_channels = hp.model.wavegan.in_conv_n_channels if i == 0 else hp.model.wavegan.strided_n_conv_channels[
                i - 1]
            self.convs.append(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=hp.model.wavegan.strided_n_conv_channels[i],
                          kernel_size=hp.model.wavegan.strided_conv_kernel_size[i],
                          stride=hp.model.wavegan.strided_conv_stride[i],
                          padding=0,
                          groups=hp.model.wavegan.strided_conv_groups[i],
                          bias=True)
            )
        self.convs.append(
            nn.Conv1d(in_channels=hp.model.wavegan.strided_n_conv_channels[-1],
                      out_channels=hp.model.wavegan.conv_1x1_1_n_channels,
                      kernel_size=hp.model.wavegan.conv_1x1_1_kernel_size,
                      padding=0)
        )
        self.convs.append(
            nn.Conv1d(in_channels=hp.model.wavegan.conv_1x1_1_n_channels,
                      out_channels=1,
                      kernel_size=hp.model.wavegan.conv_1x1_2_kernel_size,
                      padding=0)
        )

    def forward(self, ground_truth, prediction):
        ground_truth = self.resample(ground_truth)
        prediction = self.resample(prediction)
        L_FM_G = torch.zeros(ground_truth.shape[0]).to(ground_truth.device)
        for i, conv in enumerate(self.convs):
            ground_truth = conv(ground_truth)
            prediction = conv(prediction)
            if i < len(self.convs) - 1:
                L_FM_G += torch.norm(prediction - ground_truth, p=1, dim=[d for d in range(1, ground_truth.dim())]) / \
                          ground_truth.shape[1]
                ground_truth = F.leaky_relu(ground_truth)
                prediction = F.leaky_relu(prediction)
        return torch.mean(ground_truth.squeeze(), dim=-1), torch.mean(prediction.squeeze(), dim=-1), torch.mean(L_FM_G)


class SpecGANBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(SpecGANBlock, self).__init__()
        pH = int(((stride[0] - 1) * hp.model.specgan.n_mels - stride[0] + kernel_size[0]) / 2)
        pW = int(((stride[1] - 1) * np.ceil(hp.training.sequence_length / hp.model.specgan.n_mels)
                  - stride[1] + kernel_size[1]) / 2)
        self.in_conv = nn.Conv2d(in_channels=in_channels,
                                 out_channels=2 * hp.model.specgan.n_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=(pH, pW),
                                 bias=False)  # bias = False since it's immediately followed by BatchNorm
        self.norm = nn.BatchNorm2d(2 * hp.model.specgan.n_channels)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.norm(x)
        a, b = torch.split(x, hp.model.specgan.n_channels, dim=1)
        return a * torch.sigmoid(b)


class SpecGAN(nn.Module):
    def __init__(self):
        super(SpecGAN, self).__init__()
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=hp.model.specgan.n_fft,
                                                             hop_length=hp.model.specgan.hop_length,
                                                             normalized=True)
        self.mel_scale = torchaudio.transforms.MelScale(n_mels=hp.model.specgan.n_mels,
                                                        sample_rate=hp.dsp.sample_rate,
                                                        f_min=hp.model.specgan.f_min,
                                                        f_max=hp.model.specgan.f_max)
        self.blocks = nn.ModuleList()
        for i in range(hp.model.specgan.n_stacks):
            in_channels = 1 if i == 0 else hp.model.specgan.n_channels
            self.blocks.append(SpecGANBlock(in_channels=in_channels,
                                            kernel_size=hp.model.specgan.kernel_sizes[i],
                                            stride=hp.model.specgan.strides[i]))
        self.out_conv = nn.Conv2d(in_channels=hp.model.specgan.n_channels,
                                  out_channels=1,
                                  kernel_size=hp.model.specgan.out_conv_kernel_size,
                                  stride=hp.model.specgan.out_conv_stride)

    def forward(self, ground_truth, prediction):
        ground_truth = self.mel_scale(self.spectrogram(ground_truth) / self.spectrogram.n_fft)
        prediction = self.mel_scale(self.spectrogram(prediction) / self.spectrogram.n_fft)
        L_FM_G = torch.zeros(ground_truth.shape[0]).to(ground_truth.device)
        for block in self.blocks:
            ground_truth = block(ground_truth)
            prediction = block(prediction)
            L_FM_G += torch.norm(prediction - ground_truth, p=1, dim=[d for d in range(1, ground_truth.dim())]) / \
                      ground_truth.shape[1]
        ground_truth = self.out_conv(ground_truth)
        prediction = self.out_conv(prediction)
        return torch.mean(ground_truth.squeeze(), dim=(1, 2)), torch.mean(prediction.squeeze(), dim=(1, 2)), torch.mean(
            L_FM_G)


class Generator(nn.Module):
    def __init__(self, wavenet):
        super(Generator, self).__init__()
        self.wavenet = wavenet
        self.postnet = PostNet()

    def forward(self, x):
        prediction = self.wavenet(x)
        return prediction, self.postnet(prediction)

    def inference(self, x):
        if len(x.shape) == 1:
            x = x[None, None, :]
        else:
            x = x[:, None, :]
        _, y = self.forward(x)
        return y.squeeze()


class HiFiGAN(nn.Module):
    def __init__(self, generator):
        super(HiFiGAN, self).__init__()
        self.generator = Generator(generator)
        self.discriminators = nn.ModuleList()
        for resample_freq in hp.training.resample_freqs:
            self.discriminators.append(
                WaveGAN(resample_freq)
            )
        self.discriminators.append(
            SpecGAN()
        )

    def forward(self, x, ground_truth):
        x = x[:, None, :]
        ground_truth = ground_truth[:, None, :]
        prediction, prediction_postnet = self.generator(x)
        ground_truth_scores = []
        prediction_scores = []
        L_FM_G = []
        for discriminator in self.discriminators:
            ground_truth_score, prediction_score, L = discriminator(ground_truth, prediction)
            ground_truth_scores.append(ground_truth_score)
            prediction_scores.append(prediction_score)
            L_FM_G.append(L)
        discriminator_scores = list(zip(ground_truth_scores, prediction_scores))
        return prediction.squeeze(), prediction_postnet.squeeze(), prediction_scores, discriminator_scores, L_FM_G
