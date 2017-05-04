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
# res_model = \
TR.do_nn_train(nn, train_gen, test_gen, cnf.NN_PARAM, cnf.NN_MODEL_FILE)

train_gen.stop()
test_gen.stop()
# ===
# Fit generator
# ===
# gen = urb_load.MiniBatchGenerator('./data/train_set.h5', 80, 50, batch_type="seq_slide_wnd")
# nn = model.audio_crnn(cnf.NN_PARAM, nclass=cnf.NN_OUT_LAB, input_shape=gen.batch_shape())
#
# h = nn.fit_generator(gen.batch(),
#                      samples_per_epoch=7000, nb_epoch=1)
# print(h.history)
# gen.stop()

# ===
# Train on batch
# ===
# gen = urb_load.MiniBatchGenerator('./data/train_set.h5', cnf.NN_PARAM['frames'],
#                                   cnf.NN_PARAM['batch'], batch_type=cnf.NN_PARAM['batch_type'])
# nn = model.audio_crnn(cnf.NN_PARAM, nclass=cnf.NN_OUT_LAB, input_shape=gen.batch_shape())
# tr_hist = []
# # for ep in xrange(cnf.NN_PARAM['n_epoch']):
# for ep in xrange(1):
#     tr_loss, acc, cnt = 0, 0, 0
#     for X_batch, Y_batch in gen.batch():
#         print("batch: %d" % cnt)
#         v = nn.train_on_batch(X_batch, Y_batch)
#         tr_loss += v[0]
#         acc += v[1]
#         cnt += 1
#         print(v[0], v[1])
#     tr_hist.append(tr_loss / cnt)
# print(acc / cnt, tr_hist[-1])
# gen.stop()


# TODO:
# - refactor batch generator
# + trainer
# - tester
# - custom evaluation on Test set!!!
# - fit_generator + tensorboard


# X_train, Y_train = urb_load.extract_mfcc_features(cnf.DATASET_BASE_PATH, cnf.TR_SUB_DIRS,
#                                                   frames=n_steps, bands=n_input, add_channel=True)
# Y_train = urb_load.one_hot_encode(Y_train)
#
# X_test, Y_test = urb_load.extract_mfcc_features(cnf.DATASET_BASE_PATH, cnf.TST_SUB_DIRS,
#                                                 frames=n_steps, bands=n_input, add_channel=True)
# Y_test = urb_load.one_hot_encode(Y_test)

# params = {'loss': 'mse', 'optimizer': 'adam', 'learn_rate': learning_rate}
# input_shape = (batch_size, X_train.shape[1], X_train.shape[2], X_train.shape[3])
# model = model.audio_crnn(params, nclass=n_classes, input_shape=input_shape)
#
# for iter in xrange(training_iters):
#         print("Iter: %s" % iter)
#         offset = (iter * batch_size) % (Y_train.shape[0] - batch_size)
#         batch_x = X_train[offset:(offset + batch_size), :, :]
#         batch_y = Y_train[offset:(offset + batch_size), :]
#         v = model.train_on_batch(batch_x, batch_y)
#
#         if iter % display_step == 0:
#             # Calculate accuracy
#             v = model.evaluate(X_test, Y_test)
#             print "Iter " + str(iter) + ", Minibatch Loss= " + \
#                   "{:.6f}".format(v[0]) + ", Test Accuracy= " + \
#                   "{:.5f}".format(v[1])
#
# v = model.evaluate(X_test, Y_test)
# print('Test loss: ', round(v[0]), 3)
# print('Test accuracy: ', round(v[1]), 3)
# model.save("crnn_%0.2f.pkl" % v[1])
# # 46,69 best
