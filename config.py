import os.path as path
from utils.features import MFCC, FBANK

# raw data
DATASET_BASE_PATH = '/home/vano/wrkdir/datasets/UrbanSound8K/audio/'
TR_SUB_DIRS = ['fold1_dwnsmp', 'fold2_dwnsmp']
TST_SUB_DIRS = ['fold3_dwnsmp']
# features
FEAT_PATH = './data/'
TRAIN_FEAT = path.join(FEAT_PATH, 'train_set.h5')
TEST_FEAT = path.join(FEAT_PATH, 'test_set.h5')

NN_MODEL_PATH = path.join(FEAT_PATH, 'model')
NN_PARAM = {
            'activations': ['elu', 'elu', 'elu', 'elu', 'sigmoid'],
            'optimizer': 'adam',
            'learn_rate': 0.001,
            'batch': 50,
            'batch_type': 'seq_slide_wnd',  # mtag_oversmp, rnd_wnd, seq_slide_wnd
            'dropout': 0.5,
            'feature_maps': [32],
            'frames': 80,                   # number of spliced frames, i.e. frame context
            'loss': 'mse',       # mfom_eer_uvz, mfom_eer_ovo, mfom_eer_uvz, mse, categorical_crossentropy
            'n_epoch': 1,
            'out_dim': 10
            }


# training_iters = 500
# batch_size = 50
# display_step = 30
#
# n_input = 64
# n_steps = 80
# n_hidden = 100
