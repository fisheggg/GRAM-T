# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.

import os


def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=True,     # To do quick test. Trains/test on small subset of dataset, and # of epochs
    
        finetune_mode = False,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights = 'models/1_1_foa_dev_split6_model.h5', 

        # GRAM pretraining checkpoint for the encoder (overridden by $GRAM_CKPT)
        pretrained_gram_ckpt = '/projects/0/prjs1338/saved_models_rebuttal_ambisonics/Data=AudioSet/InChannels=7/WithNoise=None/Model=GRAM-T/ModelSize=base/MWMAE=True/NrGpus=1/LR=0.0002/BatchSize=96/NrSamples=16/PosLearn=False/Patching=frame/MaskPatch=160/InputL=200/Cluster=False/step=60000.ckpt',

        # INPUT PATH
        # dataset_dir='DCASE2020_SELD_dataset/',  # Base folder containing the foa/mic and metadata folders
        dataset_dir = '/projects/0/prjs1338/STARSS2022/32000',

        # OUTPUT PATHS
        # feat_label_dir='DCASE2020_SELD_dataset/feat_label_hnet/',  # Directory to dump extracted features and labels
        feat_label_dir='/projects/0/prjs1338/STARSS2022/32000/seld_feat_label_new',
 
        model_dir='models/',            # Dumps the trained models and training curves in this folder
        dcase_output_dir='results/',    # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',       # 'foa' - ambisonic or 'mic' - microphone signals

        #FEATURE PARAMS
        fs=32000,
        hop_len_s=0.01,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=128,

        # FOA CONVENTION: match STARSS (SN3D, Z-up) to GRAM pretraining (N3D, ch1-up).
        # Applied to dataset='foa' only; set both False to reproduce historical runs.
        sn3d_to_n3d=True,            # scale STARSS dipoles by sqrt(3) before RMS normalization; features go to <combination>_n3d
        swap_vertical_to_ch1=True,   # feature order [0,2,1,3,5,4,6] before the encoder

        # MODEL TYPE
        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # DNN MODEL PARAMETERS
        label_sequence_length=20,    # Feature sequence length
        batch_size=128,              # Batch size
        dropout_rate=0.05,             # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        nb_rnn_layers=2,
        rnn_size=128,        # RNN contents, length of list = number of layers, list value = number of nodes

        self_attn=False,
        nb_heads=4,

        nb_fnn_layers=1,
        fnn_size=128,             # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=100,              # Train for maximum epochs
        lr=1e-4,

        # METRIC
        average = 'macro',        # Supports 'micro': sample-wise average and 'macro': class-wise average
        lad_doa_thresh=20,

        # SELD seed: random/numpy/torch seeded once before data and model (as fox-replication); None = unseeded
        seed=42,
        deterministic=True,  # torch.use_deterministic_algorithms(True), cuDNN deterministic, no benchmark
    )

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        print("FOA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    elif argv == '3':
        print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

    elif argv == '4':
        print("MIC + GCC + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False

    elif argv == '5':
        print("MIC + SALSA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False

    elif argv == '6':
        print("MIC + GCC + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True

    elif argv == '7':
        print("MIC + SALSA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True

    elif argv == '20':
        # Olivia: archived Stage 3 SELD settings (stage2_legacy protocol).
        print("OLIVIA: FOA + ACCDOA, STARSS23 + synth, effective batch 512\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/cluster/work/projects/nn8104k/jinyueg/ambi_ssl/data/SELD_combined'
        params['dataset_year'] = '2022'  # STARSS22/23: 13 classes, train folds 1-3, val/test fold 4
        params['batch_size'] = 512
        params['micro_batch_size'] = 32  # gradient accumulation; same update as one 512 batch

    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    # A run config (configs/seld/*.yaml, via $SELD_CONFIG) replaces the task-id defaults.
    run_cfg = {}
    if os.environ.get('SELD_CONFIG'):
        import yaml
        with open(os.environ['SELD_CONFIG']) as f:
            run_cfg = yaml.safe_load(f)
        if str(run_cfg['task_id']) != argv:
            raise ValueError('{} is for task id {}, not {}'.format(os.environ['SELD_CONFIG'], run_cfg['task_id'], argv))
        params.update(run_cfg.get('params', {}))

    # Per-run settings from the Slurm script (scripts/seld.sh) or the run config.
    run_dir = os.environ.get('SELD_RUN_DIR') or run_cfg.get('run_dir')
    if run_dir:
        params['feat_label_dir'] = os.path.join(run_dir, 'features')
        params['model_dir'] = os.path.join(run_dir, 'models')
        params['dcase_output_dir'] = os.path.join(run_dir, 'predictions')
    if os.environ.get('SELD_FOA_FIX') in ('0', '1'):
        params['sn3d_to_n3d'] = params['swap_vertical_to_ch1'] = os.environ['SELD_FOA_FIX'] == '1'
    params['pretrained_gram_ckpt'] = os.environ.get('GRAM_CKPT', params['pretrained_gram_ckpt'])
    # wandb logging is on only when the run has a name.
    wandb_cfg = run_cfg.get('wandb', {})
    params['wandb_name'] = os.environ.get('SELD_WANDB_NAME', wandb_cfg.get('name'))
    params['wandb_group'] = os.environ.get('SELD_WANDB_GROUP', wandb_cfg.get('group'))
    params['wandb_entity'] = os.environ.get('WANDB_ENTITY', wandb_cfg.get('entity', 'ritmo'))
    params['wandb_project'] = os.environ.get('WANDB_PROJECT', wandb_cfg.get('project', 'ambi-ssl'))
    if os.environ.get('SELD_QUICK_TEST') in ('0', '1'):
        params['quick_test'] = os.environ['SELD_QUICK_TEST'] == '1'
    if 'dataset_year' not in params:
        params['dataset_year'] = next((y for y in ('2020', '2021', '2022') if y in params['dataset_dir']), None)

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]     # CNN time pooling
    params['patience'] = int(params['nb_epochs'])     # Stop training if patience is reached

    if params['dataset_year'] == '2020':
        params['unique_classes'] = 14
    elif params['dataset_year'] == '2021':
        params['unique_classes'] = 12
    elif params['dataset_year'] == '2022':
        params['unique_classes'] = 13

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
