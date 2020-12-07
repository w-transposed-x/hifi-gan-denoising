import librosa
from torch.utils.tensorboard import SummaryWriter

from hparams import hparams as hp


class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)
        self.scalars = dict()

    def log_training(self, step, scalars):
        for plot, series in scalars.items():
            self.add_scalars(plot, series, step)

    def log_validation(self, model, step, scalars, audio_data=None):
        for plot, series in scalars.items():
            self.add_scalars(plot, series, step)

        # Log and plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), step)

        # Log audio data
        if audio_data is not None:
            for i in range(hp.training.batch_size):
                for name, sound_data in audio_data.items():
                    self.add_audio(f'{i}_{name}',
                                   librosa.util.normalize(sound_data[i].squeeze()),
                                   global_step=step,
                                   sample_rate=hp.dsp.sample_rate)
