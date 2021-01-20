class hparams:
    class dsp:
        sample_rate = 16000
        min_vol = -96  # in dBFS
        max_vol = 0  # in dBFS

    class training:
        num_workers = 8
        chance_for_no_reverb = 0.1  # Own choice, to ensure model learns to map dry speech to dry speech
        chance_for_no_noise = 0.1  # Own choice, to ensure model learns to map dry speech to dry speech
        sequence_length = 32000  # According to Su et al. (2020)
        resample_freqs = [16000, 8000, 4000]  # According to Su et al. (2020)
        mixed_precision = True
        validation_every_n_steps = 1000 #10000
        n_validation_steps = 200 #1000
        scheme = {  # According to Su et al. (2020), batch sizes chosen to max out 8GB VRAM
            0: {
                'modules': 'wavenet',
                'steps': 5000, #500000,
                'batch_size': 3,
                'lr_generator': 1e-3,
                'lr_discriminator': None,
                'augmentation': {
                    'speaker': False,
                    'ir': False,
                    'noise': True
                },
            },
            1: {
                'modules': 'wavenet-postnet',
                'steps': 5000, #500000,
                'batch_size': 3,
                'lr_generator': 1e-4,
                'lr_discriminator': None,
                'augmentation': {
                    'speaker': False,
                    'ir': False,
                    'noise': True
                },
            },
            2: {
                'modules': 'all',
                'steps': 500, #50000,
                'batch_size': 2,
                'lr_generator': 1e-5,
                'lr_discriminator': 1e-3,
                'augmentation': {
                    'speaker': False,
                    'ir': False,
                    'noise': True
                },
            }
        }

    class inference:
        batched = True
        batch_size = 3
        sequence_length = 32000

    class augmentation:
        rand_eq_filter_fraction = 3  # Own choice
        rand_eq_filter_order = 3  # Own choice
        rand_eq_limits_freq = [200, 20000]  # Own choice
        sp_resample_factor_bounds = [0.9, 1.1]  # Own choice, paper was inconclusive
        sp_gain_factor_bounds = [0.5, 1.]  # Own choice, paper was inconclusive
        ir_drr_bounds_db = [-6, 18]  # According to Bryan (2019)
        ir_rand_eq_mean_std_db = [0, 1.5]  # Own choice, paper was inconclusive
        noise_rand_eq_limits_db = [-12, 6]  # Own choice, paper was inconclusive
        noise_rand_snr_bounds_db = [20, 40]  # According to Bryan (2019)

    class model:
        class postnet:  # According to Su et al. (2020)
            n_layers = 12
            n_channels = 128
            kernel_size = 31

        class wavegan:  # According to Su et al. (2020)
            in_conv_n_channels = 16
            in_conv_kernel_size = 15
            strided_n_conv_channels = [64, 256, 1024, 1024]
            strided_conv_kernel_size = [41, 41, 41, 41]
            strided_conv_stride = [4, 4, 4, 4]
            strided_conv_groups = [4, 16, 64, 256]
            strided_conv_n_layers = len(strided_n_conv_channels)
            conv_1x1_1_n_channels = 1024
            conv_1x1_1_kernel_size = 5
            conv_1x1_2_kernel_size = 3

        class specgan:  # According to Su et al. (2020)
            n_channels = 32
            kernel_sizes = [(3, 9), (3, 8), (3, 8), (3, 6)]
            strides = [(1, 2), (1, 2), (1, 2), (1, 2)]
            n_stacks = len(kernel_sizes)
            out_conv_kernel_size = (36, 5)
            out_conv_stride = (36, 1)
            n_fft = 2048
            hop_length = 512
            n_mels = 80
            f_min = 20
            f_max = 8000

        class wavenet:  # According to Su et al. (2020)
            n_stacks = 2
            n_layers_per_stack = 10
            n_channels_dilated = 128
            kernel_size_dilated = 3
            n_channels_out_1 = 2048
            kernel_size_out_1 = 3
            n_channels_out_2 = 256
            kernel_size_out_2 = 3
            n_conditioning_dims = None

    class loss:
        class spectrogramloss:  # According to Su et al. (2020)
            n_fft = [2048, 512]
            hop_length = [512, 128]

    class files:
        # train_speaker = './data/train/DAPS'
        # valid_speaker = './data/test/DAPS'
        # train_ir = './data/train/IRs'
        # valid_ir = './data/test/IRs'
        # train_noise = './data/train/noise'
        # valid_noise = './data/test/noise'
        
        # For subsampled Test Run
        train_speaker = 'data/sub/train/DAPS'
        valid_speaker = 'data/sub/test/DAPS'
        train_ir = 'data/sub/train/IRs'
        valid_ir = 'data/sub/test/IRs'
        train_noise = 'data/sub/train/noise'
        valid_noise = 'data/sub/test/noise'