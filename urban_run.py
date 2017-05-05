"""
Ref.: https://github.com/Vanova/music-auto_tagging-keras
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import utils.urban_loader as urb_load
import config as cnf
from kmodel import model
import utils.urban_loader as uld
import trainer as TR

plt.style.use('ggplot')
np.random.seed(777)

# feature extraction
if not os.path.isfile(cnf.TRAIN_FEAT) and not os.path.isfile(cnf.TEST_FEAT):
    urb_load.do_feature_extraction('fbank', cnf.FBANK, cnf.DATASET_BASE_PATH, cnf.TR_SUB_DIRS,
                                   feat_file=cnf.TRAIN_FEAT)
    urb_load.do_feature_extraction('fbank', cnf.FBANK, cnf.DATASET_BASE_PATH, cnf.TST_SUB_DIRS,
                                   feat_file=cnf.TEST_FEAT)

# training
train_gen = uld.MiniBatchGenerator(cnf.TRAIN_FEAT, cnf.NN_PARAM['frames'],
                                   cnf.NN_PARAM['batch'], cnf.NN_PARAM['batch_type'])
test_gen = uld.MiniBatchGenerator(cnf.TEST_FEAT, cnf.NN_PARAM['frames'],
                                   cnf.NN_PARAM['batch'], batch_type='eval')

nn = model.audio_crnn(cnf.NN_PARAM, nclass=cnf.NN_PARAM['out_dim'], input_shape=train_gen.batch_shape())
TR.do_nn_train(nn, train_gen, test_gen, cnf.NN_PARAM, cnf.NN_MODEL_PATH)
train_gen.stop()
test_gen.stop()
