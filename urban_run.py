"""
Ref.: https://github.com/Vanova/music-auto_tagging-keras
"""
import os
import os.path as path
import matplotlib.pyplot as plt
import keras
import numpy as np
import config as cnf
from keras.models import load_model
import kmodel.model as urb_ml
import utils.urban_loader as urb_ld
import trainer as urb_tr
import utils.visialiser as vis

plt.style.use('ggplot')
np.random.seed(777)
overwrite = False

# feature extraction
if not path.isfile(cnf.TRAIN_FEAT) and not path.isfile(cnf.TEST_FEAT) or overwrite:
    urb_ld.do_feature_extraction('fbank', cnf.FBANK, cnf.DATASET_BASE_PATH, cnf.TR_SUB_DIRS,
                                   feat_file=cnf.TRAIN_FEAT)
    urb_ld.do_feature_extraction('fbank', cnf.FBANK, cnf.DATASET_BASE_PATH, cnf.TST_SUB_DIRS,
                                   feat_file=cnf.TEST_FEAT)

if not any(f.startswith(cnf.NN_FNAME) for f in os.listdir(cnf.NN_MODEL_PATH)) or overwrite:
    # batch generators
    train_gen = urb_ld.MiniBatchGenerator(cnf.TRAIN_FEAT, cnf.NN_PARAM['frames'],
                                       cnf.NN_PARAM['batch'], cnf.NN_PARAM['batch_type'])
    test_gen = urb_ld.MiniBatchGenerator(cnf.TEST_FEAT, cnf.NN_PARAM['frames'],
                                       cnf.NN_PARAM['batch'], batch_type='eval')

    # ===
    # CNN-RNN
    # ===
    nn = urb_ml.audio_crnn(cnf.NN_PARAM, nclass=cnf.NN_PARAM['out_dim'], input_shape=train_gen.batch_shape())
    # ===
    # CNN-RNN-attention
    # ===
    # nn = urb_model.cnn_rnn_attention(cnf.NN_PARAM, nclass=cnf.NN_PARAM['out_dim'], input_shape=train_gen.batch_shape())

    # training
    urb_tr.do_nn_train(nn, train_gen, test_gen, cnf.NN_PARAM, cnf.NN_MODEL_PATH, cnf.NN_FNAME)
    train_gen.stop()
    test_gen.stop()


# load network model
nn = load_model(path.join(cnf.NN_MODEL_PATH, 'crnn_0.0820_0.3483.h5'))
nn.summary()

# ===
# prepare data to visualize
# ===
# TODO return batch from audio file name
# load one batch of data
test_gen = urb_ld.MiniBatchGenerator(cnf.TEST_FEAT, cnf.NN_PARAM['frames'],
                                       cnf.NN_PARAM['batch'], batch_type='eval')
tst_file, X_tst_b, Y_tst_b = test_gen.batch().next()

# prepare visualiser
visual = vis.KNetworkVisualizer(nn)
for id, layer in enumerate(nn.layers):
    if vis.KNetworkVisualizer.is_visualizable(layer):
    # if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.MaxPooling2D) or \
    #         isinstance(layer, keras.layers.AveragePooling2D) or isinstance(layer, keras.layers.InputLayer) or \
    #         isinstance(layer, keras.layers.Dense):
        visual.layer_activations(batch=X_tst_b, lname=nn.layers[id].name)
# visual.layer_activations(batch=X_tst_b, lname=nn.layers[0].name)
# visual.all_activations(batch=X_tst_b)