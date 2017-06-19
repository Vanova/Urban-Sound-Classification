"""
Ref.: https://github.com/Vanova/music-auto_tagging-keras
"""
import os
import os.path as path
import matplotlib.pyplot as plt
import numpy as np
import utils.urban_loader as urb_load
import config as cnf
from kmodel import model
import utils.urban_loader as uld
import trainer as TR

plt.style.use('ggplot')
np.random.seed(777)
overwrite = False

# feature extraction
if not path.isfile(cnf.TRAIN_FEAT) and not path.isfile(cnf.TEST_FEAT) or overwrite:
    urb_load.do_feature_extraction('fbank', cnf.FBANK, cnf.DATASET_BASE_PATH, cnf.TR_SUB_DIRS,
                                   feat_file=cnf.TRAIN_FEAT)
    urb_load.do_feature_extraction('fbank', cnf.FBANK, cnf.DATASET_BASE_PATH, cnf.TST_SUB_DIRS,
                                   feat_file=cnf.TEST_FEAT)

if not any(f.startswith(cnf.NN_FNAME) for f in os.listdir(cnf.NN_MODEL_PATH)) or overwrite:
    # batch generators
    train_gen = uld.MiniBatchGenerator(cnf.TRAIN_FEAT, cnf.NN_PARAM['frames'],
                                       cnf.NN_PARAM['batch'], cnf.NN_PARAM['batch_type'])
    test_gen = uld.MiniBatchGenerator(cnf.TEST_FEAT, cnf.NN_PARAM['frames'],
                                       cnf.NN_PARAM['batch'], batch_type='eval')

    # ===
    # CNN-RNN
    # ===
    nn = model.audio_crnn(cnf.NN_PARAM, nclass=cnf.NN_PARAM['out_dim'], input_shape=train_gen.batch_shape())
    # ===
    # CNN-RNN-attention
    # ===
    # nn = model.cnn_rnn_attention(cnf.NN_PARAM, nclass=cnf.NN_PARAM['out_dim'], input_shape=train_gen.batch_shape())

    # training
    TR.do_nn_train(nn, train_gen, test_gen, cnf.NN_PARAM, cnf.NN_MODEL_PATH)
    train_gen.stop()
    test_gen.stop()
