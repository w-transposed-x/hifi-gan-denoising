import librosa
import numpy as np
import soundfile as sf
from scipy import linalg, optimize, stats
from scipy.signal import hilbert

from hparams import hparams as hp
from utils.filter import filterbank


def load_audio(file, max_seconds=None):
    # Instead of loading audio as mono, a random channel is chosen since
    # some IR files might have a large number of channels.
    info = sf.info(file)
    if max_seconds is not None:
        rng = np.random.default_rng()
        start = rng.integers(low=0, high=np.max((int(info.frames - max_seconds * info.samplerate), 1)))
        audio = next(sf.blocks(file=file,
                               blocksize=max_seconds * info.samplerate,
                               start=start,
                               always_2d=True)).T
    else:
        audio, _ = sf.read(file, always_2d=True)
        audio = audio.T

    # Randomly chooses channel from signal
    audio = audio[np.random.choice(audio.shape[0]), :]
    if info.samplerate > hp.dsp.sample_rate:
        audio = librosa.core.resample(audio, info.samplerate, hp.dsp.sample_rate)

    # Removes DC offset if signal appears to be recorded audio (as opposed to synthetic IRs e.g.)
    if not all(audio >= 0.):
        audio -= np.mean(audio)

    # Normalize audio and scale to max_vol defined in hparams
    audio = librosa.util.normalize(audio) * librosa.db_to_amplitude(hp.dsp.max_vol)
    return audio


def generate_fades(win_length):
    win = np.ones(win_length)
    fade_len = hp.dsp.sample_rate // 1000  # 1ms fade
    win[:fade_len] = np.linspace(0, 1, fade_len)
    win[-fade_len:] = np.linspace(1, 0, fade_len)
    return win


def trim_speaker(x):
    y = librosa.effects.trim(x, top_db=15)
    fades = generate_fades(len(y[0]))
    y = y[0] * fades
    return y


def trim_ir(x, sr):
    fade_len = sr // 200  # 5ms fade
    x = x[:-int(len(x) / 16)]  # Cuts last 16th of IR to remove clicks that are prevalant in many datasets
    x[-fade_len:] *= np.linspace(1, 0, fade_len)  # Fades out last 5ms of IR
    x[:np.abs(x).argmax()] *= np.linspace(0., 1., np.abs(x).argmax()) ** 3  # Fades in to t0 (direct path)
    y = np.trim_zeros(x)  # Remove leading and trailing zeros
    return y


def get_t60(audio, fs):
    # Modified from python-acoustics (BSD-3-Clause License)
    # https://github.com/python-acoustics/python-acoustics

    # T30 params
    init = -5.0
    end = -35.0
    factor = 2.0

    abs_signal = np.abs(audio) / np.max(np.abs(audio))

    # Schroeder integration
    sch = np.cumsum(abs_signal[::-1] ** 2)[::-1]
    sch_db = 10.0 * np.log10(sch / np.max(sch))

    # Linear regression
    sch_init = sch_db[np.abs(sch_db - init).argmin()]
    sch_end = sch_db[np.abs(sch_db - end).argmin()]
    init_sample = np.where(sch_db == sch_init)[0][0]
    end_sample = np.where(sch_db == sch_end)[0][0]
    x = np.arange(init_sample, end_sample + 1) / fs
    y = sch_db[init_sample:end_sample + 1]
    slope, intercept = stats.linregress(x, y)[0:2]

    # Reverberation time T30
    db_regress_init = (init - intercept) / slope
    db_regress_end = (end - intercept) / slope
    t60 = factor * (db_regress_end - db_regress_init)
    return t60


def apply_t60_augmentation(h, desired_t60):
    # Mostly according to Bryan (2019), many details were unclear or missing however.

    a_t = lambda t, A, tau, An: np.sqrt(A ** 2 * np.exp(-2 * tau * t) + An ** 2)  # According to Karjalainen (2001)

    # t0 not clearly defined in Bryan (2019), supposed to be "late-field onset time" without providing definition.
    # Diving down the rabbit hole of references, we arrived at Abel & Huang (2006) who provided different methods
    # for calculating late-field onset time.
    t0 = detect_late_field_onset(h, threshold=1.0)
    h_l = h[t0:]
    t = np.arange(len(h_l)) / hp.dsp.sample_rate

    # Exponent of 0.4 according to old MATLAB code by Karjalainen
    # http://legacy.spa.aalto.fi/software/decay/
    y = np.abs(hilbert(np.concatenate((h_l[::-1], h_l)))) ** 0.4
    y = y[len(t):]

    # Method of arriving at initial params for curve_fit according to old Karjalainen MATLAB code mentioned above.
    half_len = int(np.ceil(len(h_l) / 2))
    tmat = np.stack((np.ones(half_len), t[:half_len]), axis=1)
    p = linalg.lstsq(tmat, librosa.amplitude_to_db(h_l[:half_len], ref=1.0))
    params, _ = optimize.curve_fit(a_t,
                                   t,
                                   y,
                                   p0=[y[:int(0.1 * len(y))].mean(), -p[0][1] / 8.7, y[-int(0.1 * len(y)):].mean()],
                                   maxfev=200000)
    tau_estimate_fullband = params[1]

    # Calculates desired fullband decay rate and gamma according to
    tau_desired = np.log(1000) / desired_t60
    gamma = tau_desired / tau_estimate_fullband

    # Initializes white noise
    z = np.random.normal(0, 1, len(h_l))

    # Initializes vector to sum subbands
    h_l_augmented = np.zeros_like(h_l)

    # Multi-band filter IR
    h_bands, freq_d, freq_u = filterbank(h, hp.dsp.sample_rate, 3, 3, [50, 7000])

    # Multi-band filter noise
    z_bands, _, _ = filterbank(z, hp.dsp.sample_rate, 3, 3, [50, 7000])

    # Accounts for edge freqs
    freq_d = np.insert(freq_d, 0, 0)
    freq_d = np.append(freq_d, freq_u[-1])

    for i, h_m in enumerate(h_bands):
        # Takes only late-field from filtered IR
        h_m = h_m[t0:]

        # Calculates DC offset
        DC_offset = h_m.mean()

        # Filtered noise
        z_m = z_bands[i]

        # Same param estimation as for fullband
        y_m = np.abs(hilbert(np.concatenate((h_m[::-1], h_m)) - DC_offset)) ** 0.4
        y_m = y_m[len(t):]
        p = linalg.lstsq(tmat, librosa.amplitude_to_db(h_m[:half_len], ref=1.0))
        params, _ = optimize.curve_fit(a_t,
                                       t,
                                       y_m,
                                       p0=[y_m[:int(0.1 * len(y_m))].mean(), -p[0][1] / 8.7,
                                           y_m[-int(0.1 * len(y_m)):].mean()],
                                       maxfev=200000)
        A_estimate_band, tau_estimate_band, An_estimate_band = tuple(params)

        # Desired subband decay rate according to Bryan (2019)
        tau_m_d = gamma * tau_estimate_band

        # Generates envelope for subband
        envelope_estimate_band = a_t(t, A_estimate_band, tau_estimate_band, An_estimate_band)

        # Detects noise floor onset
        noise_floor_onset = detect_noise_floor_onset(envelope_estimate_band)

        # Extends envelope and applies envelope to noise subband
        envelope_extended = extend_envelope(envelope_estimate_band, noise_floor_onset)
        z_m_with_env = z_m * (envelope_extended ** (1 / 0.4) + DC_offset) / np.std(z_m)

        # Crossfades original IR subband and synthetic noise subband
        h_m_extended = xfade_original_and_synth(h_m, z_m_with_env, noise_floor_onset, freq_d[i])

        # Augments subband reverberation time by applying (7) from Bryan (2019). We strongly suspect that this is where
        # the procedure fails. The attenuation or boost applied to the IR subband appears to be incorrect. Note that we
        # omitted -t0 in the exponential in our implementation since h_m only contains the late-field.
        # h_m_extended *= np.exp(-t  * ((tau_estimate_band - tau_m_d) / (tau_estimate_band * tau_m_d)))

        # We found an alternative, simpler update rule by Cabrera et al. (2011) that generates more reasonable
        # attenuation or boost values, yet still fail to generate augmentations that conform to the desired output.
        # https://pdfs.semanticscholar.org/e821/2a0451a5287c62ed744f505b82c37b69e074.pdf
        h_m_extended *= np.exp(-t * (tau_estimate_band - tau_m_d))

        h_l_augmented += h_m_extended

    # Concats beginning of original IR with augmented late-field
    h_augmented = np.concatenate([h[:t0], h_l_augmented])
    return h_augmented


def detect_late_field_onset(h, threshold=1.0):
    # According to Abel & Huang (2006)

    win_len = int(hp.dsp.sample_rate * 20 // 1000) + 1
    win = np.hanning(win_len)
    win /= win.sum()
    delta = int(win_len // 2)

    eta = np.zeros_like(h)
    h = np.pad(h, delta)

    for t in range(delta, len(h) - delta - 1):
        sigma = np.sqrt(np.sum(win * h[t - delta:t + delta + 1] ** 2))
        denom = (1 / 3 * np.sum(win * h[t - delta:t + delta + 1] ** 4)) ** (1 / 4)
        eta[t - delta] = sigma / denom

    return np.abs(eta - threshold).argmin()


def xfade_original_and_synth(h_m, z_m_with_env, noise_floor_onset, freq_d):
    if freq_d > 0:
        fade_len = int(hp.dsp.sample_rate / freq_d)
    else:
        fade_len = np.min((2 * int(hp.dsp.sample_rate * 100 // 1000), int((len(h_m) - 2) / 2)))
    if fade_len > len(h_m) - noise_floor_onset - 1:
        return h_m.copy()
    fade_out_orig = np.sqrt(0.5 * (1 - np.linspace(-1, 1, 2 * fade_len)))
    fade_in_synth = np.sqrt(np.linspace(0, 1, 2 * fade_len))
    win_orig = np.concatenate([np.ones(np.max([noise_floor_onset - fade_len, 1])), fade_out_orig])
    win_orig = np.pad(win_orig, (0, len(h_m) - len(win_orig)), mode='constant')
    win_synth = np.concatenate([np.zeros(np.max([noise_floor_onset - fade_len, 1])), fade_in_synth])
    win_synth = np.pad(win_synth, (0, len(z_m_with_env) - len(win_synth)), constant_values=1., mode='constant')
    return h_m * win_orig + z_m_with_env * win_synth


def extend_envelope(envelope_estimated, noise_floor_onset):
    # Fit straight line in dB scaling, return exponential, extended envelope.

    env_db = librosa.amplitude_to_db(envelope_estimated, ref=1.0)
    a = (env_db[noise_floor_onset] - env_db[0]) / np.max((noise_floor_onset, 1.))
    b = env_db[0]
    return librosa.db_to_amplitude(a * np.arange(len(envelope_estimated)) + b)


def detect_noise_floor_onset(envelope):
    # Finds point on envelope 3 dB above noise floor. No detailed specifications were provided, 3 dB were chosen by us
    # in reference to conventions for critical frequencies in filter design.

    fit_db = 20 * np.log10(envelope)
    reference = fit_db[-1]
    onset = np.abs(fit_db - reference - 3).argmin()
    return onset


def get_drr(h_early, h_late):
    # According to Bryan (2019)

    return 10 * np.log10(np.dot(h_early, h_early) / np.dot(h_late, h_late))


def apply_drr_augmentation(h, random_drr):
    # According to Bryan (2019)

    t0 = int(hp.dsp.sample_rate * 2.5 // 1000)
    win = np.hanning(2 * t0)
    td = np.abs(h).argmax()
    h_e = h_early(h, td, t0)
    h_l = h_late(h, td, t0)
    a, h_e, h_l, win = alpha(td, random_drr, h_e, h_l, win)
    h_e = a * win * h_e + (1 - win) * h_e
    return (h_e + h_l) / np.max(np.abs(h_e))


def alpha(td, drr_db, h_early, h_late, win):
    if len(win) // 2 > td:
        h_early = np.pad(h_early, (len(win) // 2 - td, 0), 'constant')
        h_late = np.pad(h_late, (len(win) // 2 - td, 0), 'constant')
    td = np.argmax(np.abs(h_early))
    win = np.pad(win, (td - len(win) // 2, 0), 'constant')
    win = np.pad(win, (0, len(h_early) - len(win)), 'constant')
    a = np.dot(win ** 2, h_early ** 2)
    b = np.sum((1 - win) * win * h_early ** 2)
    c = np.dot((1 - win) ** 2, h_early ** 2)
    d = 10 ** (drr_db / 10) * np.dot(h_late, h_late)
    inner = a * (d - c) + b ** 2
    if inner <= 0.:
        alpha = 1.  # Set alpha to 1 if inner part of sqrt would be negative (i.e. forgo DRR augmentation)
    else:
        alpha = (np.sqrt(inner) - b) / a
    alpha = np.clip(alpha, np.max(np.abs(h_late) / np.max(np.abs(h_early))), None)
    return alpha, h_early, h_late, win


def h_early(h, td, t0):
    h_early = np.zeros_like(h)
    h_early[np.clip(td - t0, 0, None):td + t0] = h[np.clip(td - t0, 0, None):td + t0]
    return h_early


def h_late(h, td, t0):
    h_l = np.zeros_like(h)
    h_l[:np.clip(td - t0, 0, None)] = h[:np.clip(td - t0, 0, None)]
    h_l[td + t0:] = h[td + t0:]
    return h_l
