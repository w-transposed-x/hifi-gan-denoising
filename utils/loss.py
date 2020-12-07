import torch
import torch.nn.functional as F
import torchaudio

from hparams import hparams as hp


class L_Adv_G():
    def __init__(self):
        pass

    def __call__(self, prediction_score):
        return torch.clamp(1 - prediction_score, min=0)


class L_Dk():
    def __init__(self):
        pass

    def __call__(self, ground_truth_score, prediction_score):
        return torch.clamp(1 + prediction_score, min=0) + torch.clamp(1 - ground_truth_score, min=0)


class SampleLoss():
    def __init__(self):
        pass

    def __call__(self, ground_truth, prediction):
        ground_truth = ground_truth.squeeze()
        prediction = prediction.squeeze()
        return F.l1_loss(prediction, ground_truth)


class SpectrogramLoss():
    def __init__(self, rank):
        self.stft_1 = torchaudio.transforms.Spectrogram(n_fft=hp.loss.spectrogramloss.n_fft[0],
                                                        hop_length=hp.loss.spectrogramloss.hop_length[0],
                                                        normalized=True).to(rank)
        self.stft_2 = torchaudio.transforms.Spectrogram(n_fft=hp.loss.spectrogramloss.n_fft[1],
                                                        hop_length=hp.loss.spectrogramloss.hop_length[1],
                                                        normalized=True).to(rank)
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=-hp.dsp.min_vol)

    def __call__(self, ground_truth, prediction):
        ground_truth = ground_truth.squeeze()
        prediction = prediction.squeeze()
        ground_truth_stft1 = self.amp_to_db(
            self.stft_1(ground_truth) / self.stft_1.n_fft
        )
        ground_truth_stft2 = self.amp_to_db(
            self.stft_2(ground_truth) / self.stft_1.n_fft
        )
        prediction_stft1 = self.amp_to_db(
            self.stft_1(prediction) / self.stft_2.n_fft
        )
        prediction_stft2 = self.amp_to_db(
            self.stft_2(prediction) / self.stft_2.n_fft
        )
        return F.mse_loss(prediction_stft1, ground_truth_stft1) + F.mse_loss(prediction_stft2, ground_truth_stft2)


class CombinedLoss():
    def __init__(self, rank):
        self.l_adv_g = L_Adv_G()
        self.l_dk = L_Dk()
        self.sample_loss = SampleLoss()
        self.spectrogram_loss = SpectrogramLoss(rank)

    def __call__(self, step, ground_truth, prediction, prediction_postnet, prediction_scores, discriminator_scores,
                 L_FM_G):

        # Discriminator losses
        D_losses = []
        for i in range(4):
            D_loss = torch.mean(self.l_dk(discriminator_scores[i][0], discriminator_scores[i][1])) + L_FM_G[i]
            D_losses.append(D_loss)

        combined_loss = torch.sum(torch.stack(D_losses))

        # Generator loss
        if step % 1 == 0:  # Modulo set to 1 to prevent unused parameters error when using DDP, deviating from the paper
            wavenet_loss = self.sample_loss(ground_truth, prediction) + self.spectrogram_loss(ground_truth, prediction)
            wavenet_postnet_loss = self.sample_loss(ground_truth, prediction_postnet) + self.spectrogram_loss(
                ground_truth, prediction_postnet)

            adversarial_loss = torch.sum(torch.stack(
                [torch.mean(self.l_adv_g(prediction_score)) for prediction_score in prediction_scores]
            ))

            G_loss = wavenet_loss + wavenet_postnet_loss + adversarial_loss

            combined_loss += G_loss
        else:
            wavenet_loss = None
            wavenet_postnet_loss = None
            G_loss = None

        return combined_loss, \
               wavenet_loss, wavenet_postnet_loss, \
               G_loss, D_losses
