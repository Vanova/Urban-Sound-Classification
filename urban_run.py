"""
Ref.: https://github.com/Vanova/music-auto_tagging-keras
"""
import matplotlib.pyplot as plt
import numpy as np
import utils.urban_loader as urb_load
import config as cnfg
from kmodel import cnn_rnn

plt.style.use('ggplot')

# Network Parameters
learning_rate = 0.001
training_iters = 500
batch_size = 50
display_step = 30

n_input = 64
n_steps = 80
n_hidden = 100
n_classes = 10

# TODO:
# -extract features
# -save to hdf5
# -train / test
# my feature extraction
# urb_load.my_feature_extraction(cnfg.DATASET_BASE_PATH, cnfg.TR_SUB_DIRS, feat_name='train_set.h5')
# urb_load.my_feature_extraction(cnfg.DATASET_BASE_PATH, cnfg.TST_SUB_DIRS, feat_name='test_set.h5')
#######################

X_train, Y_train = urb_load.extract_mfcc_features(cnfg.DATASET_BASE_PATH, cnfg.TR_SUB_DIRS,
                                                  frames=n_steps, bands=n_input, add_channel=True)
Y_train = urb_load.one_hot_encode(Y_train)

X_test, Y_test = urb_load.extract_mfcc_features(cnfg.DATASET_BASE_PATH, cnfg.TST_SUB_DIRS,
                                                frames=n_steps, bands=n_input, add_channel=True)
Y_test = urb_load.one_hot_encode(Y_test)

params = {'loss': 'mse', 'optimizer': 'adam', 'learn_rate': learning_rate}
# input_shape = (batch_size, X_train.shape[1], X_train.shape[2], X_train.shape[3])
input_shape = (50, 64, 80, 1)
model = cnn_rnn.audio_crnn(params, nclass=n_classes, input_shape=input_shape)

# train model
# TODO add fit_generator and batch gen
for iter in xrange(training_iters):
        print("Iter: %s" % iter)
        offset = (iter * batch_size) % (Y_train.shape[0] - batch_size)
        batch_x = X_train[offset:(offset + batch_size), :, :]
        batch_y = Y_train[offset:(offset + batch_size), :]
        v = model.train_on_batch(batch_x, batch_y)

        if iter % display_step == 0:
            # Calculate accuracy
            v = model.evaluate(X_test, Y_test)
            print "Iter " + str(iter) + ", Minibatch Loss= " + \
                  "{:.6f}".format(v[0]) + ", Test Accuracy= " + \
                  "{:.5f}".format(v[1])

v = model.evaluate(X_test, Y_test)
print('Test loss: ', round(v[0]), 3)
print('Test accuracy: ', round(v[1]), 3)
