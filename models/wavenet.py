import torch
import torch.nn as nn

from hparams import hparams as hp


class ResLayer(nn.Module):
    def __init__(self, padding, dilation):
        super(ResLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels=hp.model.wavenet.n_channels_dilated,
                                      out_channels=hp.model.wavenet.n_channels_dilated * 2,
                                      kernel_size=hp.model.wavenet.kernel_size_dilated,
                                      stride=1,
                                      padding=padding,
                                      dilation=dilation,
                                      bias=True)

        self.conv_1x1 = nn.Conv1d(in_channels=hp.model.wavenet.n_channels_dilated,
                                  out_channels=hp.model.wavenet.n_channels_dilated * 2,
                                  kernel_size=1,
                                  bias=True)

    def forward(self, x):
        res = x
        x = self.conv_dilated(x)
        a, b = x.split(x.shape[1] // 2, dim=1)
        x = torch.tanh(a) * torch.sigmoid(b)
        x = self.conv_1x1(x)
        x, s = x.split(x.shape[1] // 2, dim=1)
        return x + res, s


class ContextStacks(nn.Module):
    def __init__(self):
        super(ContextStacks, self).__init__()

        self.dilation_layers = nn.ModuleList()
        for _ in range(hp.model.wavenet.n_stacks):
            for i in range(hp.model.wavenet.n_layers_per_stack):
                dilation = 2 ** i
                padding = (hp.model.wavenet.kernel_size_dilated - 1) // 2 * dilation
                self.dilation_layers.append(
                    ResLayer(padding, dilation)
                )

    def forward(self, x):
        skip_outs = []
        for dilation_layer in self.dilation_layers:
            x, s = dilation_layer(x)
            skip_outs.append(s)
        return torch.relu(torch.sum(torch.stack(skip_outs), dim=0))


class WaveNet(nn.Module):
    """
    As opposed to "Rethage et al. (2018) - A Wavenet for Speech Denoising" we do not use target field prediction
    but instead predict all output samples in a single pass.
    We do not think target field prediction is necessary due to using batched processing instead
    (cf. https://github.com/r9y9/wavenet_vocoder).
    """

    def __init__(self):
        super(WaveNet, self).__init__()
        self.n_conditioning_dims = hp.model.wavenet.n_conditioning_dims
        self.in_conv = nn.Conv1d(in_channels=1,
                                 out_channels=hp.model.wavenet.n_channels_dilated,
                                 kernel_size=1,
                                 bias=True)
        self.stacks = ContextStacks()
        self.out_conv_1 = nn.Conv1d(in_channels=hp.model.wavenet.n_channels_dilated,
                                    out_channels=hp.model.wavenet.n_channels_out_1,
                                    kernel_size=hp.model.wavenet.kernel_size_out_1,
                                    padding=hp.model.wavenet.kernel_size_out_1 // 2,
                                    bias=True)
        self.out_conv_2 = nn.Conv1d(in_channels=hp.model.wavenet.n_channels_out_1,
                                    out_channels=hp.model.wavenet.n_channels_out_2,
                                    kernel_size=hp.model.wavenet.kernel_size_out_2,
                                    padding=hp.model.wavenet.kernel_size_out_2 // 2,
                                    bias=True)
        self.out_1x1 = nn.Conv1d(in_channels=hp.model.wavenet.n_channels_out_2,
                                 out_channels=1,
                                 kernel_size=1,
                                 padding=0,
                                 bias=True)

    def forward(self, x):
        if len(x.shape) == 2 and x.shape[1] == hp.training.sequence_length:
            x = x.unsqueeze(1)

        x = self.in_conv(x)
        x = self.stacks(x)
        x = torch.relu(self.out_conv_1(x))
        x = self.out_conv_2(x)
        return self.out_1x1(x)
