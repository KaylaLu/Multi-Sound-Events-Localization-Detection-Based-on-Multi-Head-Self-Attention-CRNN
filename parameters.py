#
# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#



'''Strategy Setting Parameters'''
# Define the running environment is on Eddie server or Localhost, choose between 'Eddie' and 'Local'
envMode = 'Local'

# Define the running laptop type, choose between 'win' and 'mac'
compMode = 'win'

# Batch Sampling Strategy, choose between 'OBS'(Original Batch Sampling) and 'BBS'(Balanced Batch Sampling)
batch_sampling = 'OBS'

# Define Loss Type, choose between 'original' and 'focal'(reweighting by class)
LossType = 'focal'

# Class Distribution in Time Frame Samples
Class_Distribution = [44385, 46659, 849, 1530, 5375, 28537, 3319, 1046, 39870, 4438, 861, 1474, 90]

# Params setting and working folder address setting
def get_params(argv='1'):
    print("SET: {}".format(argv))
    print(f"Running in {envMode} mode")

    '''Determine environment'''
    if envMode == 'Eddie':
        # Dataset dictionary path
        dataset_dir = '/exports/eddie/scratch/s2659473/FinalProj/Dataset/DCASE2024_SELD_dataset/'
        # Feature extraction dictionary path
        feat_label_dir = '/exports/eddie/scratch/s2659473/FinalProj/Output/seld_feat_label/'

    elif envMode == 'Local':
        if compMode == 'mac':
            dataset_dir = '/Volumes/WDBlack1T/DCASE2024_SELD_dataset/'
            feat_label_dir = '/Volumes/WDBlack1T/SELDnetResult/seld_feat_label/'
        elif compMode == 'win':
            dataset_dir = 'D:/FProj/DCASE2024_SELD_dataset'
            feat_label_dir = 'D:/FProj/SELDnetResult/seld_feat_label/'
        else:
            raise ValueError("Running local computer mode must be 'mac' or 'win'")

    else:
        raise ValueError("Running mode must be 'Eddie' or 'Local'")

    # ########### default parameters ##############
    params = dict(
        quick_test=True,  # To do quick test. Trains/test on small subset of dataset, and # of epochs

        finetune_mode=True,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='3_1_dev_split0_multiaccdoa_foa_model.h5',
        # pretrained_model_weights='6_1_dev_split0_multiaccdoa_mic_gcc_model.h5',

        model_dir='models',  # Dumps the trained models and training curves in this folder
        dcase_output_dir='results',  # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',  # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',  # 'foa' - ambisonic or 'mic' - microphone signals

        # FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        use_salsalite=False,  # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite=50,
        fmax_doa_salsalite=2000,
        fmax_spectra_salsalite=9000,

        # MODEL TYPE
        modality='audio',  # 'audio' or 'audio_visual
        multi_accdoa=True,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # DNN MODEL PARAMETERS
        label_sequence_length=50,    # Feature sequence length
        # Default batch_size to be 128
        batch_size=128,              # Batch size
        dropout_rate=0.05,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        nb_heads=8,
        nb_self_attn_layers=2,
        nb_transformer_layers=2,

        nb_rnn_layers=2,
        rnn_size=128,

        nb_fnn_layers=1,
        fnn_size=128,  # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=250,  # Train for maximum epochs
        lr=1e-3,

        # METRIC
        average='macro',                 # Supports 'micro': sample-wise average and 'macro': class-wise average,
        segment_based_metrics=False,     # If True, uses segment-based metrics, else uses frame-based metrics
        evaluate_distance=True,          # If True, computes distance errors and apply distance threshold to the detections
        lad_doa_thresh=20,               # DOA error threshold for computing the detection metrics
        lad_dist_thresh=float('inf'),    # Absolute distance error threshold for computing the detection metrics
        lad_reldist_thresh=float('1'),  # Relative distance error threshold for computing the detection metrics
    )

    params['dataset_dir'] = dataset_dir
    params['feat_label_dir'] = feat_label_dir

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        print("FOA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False
        params['use_salsalite'] = False
        params['finetune_mode'] = False
        params['batch_size'] = 114
        params['nb_epochs'] = 100

    elif argv == '3':
        print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['batch_size'] = 114
        params['nb_epochs'] = 120

    elif argv == '4':
        print("MIC + GCC + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['finetune_mode'] = False
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['batch_size'] = 114
        params['nb_epochs'] = 120

    elif argv == '5':
        print("MIC + SALSA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['finetune_mode'] = False
        params['use_salsalite'] = True
        params['multi_accdoa'] = False
        params['batch_size'] = 64
        params['nb_epochs'] = 120

    elif argv == '6':
        print("MIC + GCC + multi ACCDOA\n")
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['batch_size'] = 96
        params['nb_epochs'] = 120

    elif argv == '7':
        print("MIC + SALSA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['batch_size'] = 16
        params['nb_epochs'] = 120

    # Eval set
    elif argv == '8':
        print("Eval: FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['mode'] = 'eval'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '3_Mode3R1Ori_dev_split0_multiaccdoa_foa_model.h5'
        params['batch_size'] = 89
        params['nb_epochs'] = 120

    # Quick test mode
    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True
        params['finetune_mode'] = False
        params['pretrained_model_weights'] = ''
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['batch_size'] = 114

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]  # CNN time pooling
    params['patience'] = int(params['nb_epochs'])  # Stop training if patience is reached
    params['model_dir'] = params['model_dir'] + '_' + params['modality']
    params['dcase_output_dir'] = params['dcase_output_dir'] + '_' + params['modality']
    params['unique_classes'] = 13

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
