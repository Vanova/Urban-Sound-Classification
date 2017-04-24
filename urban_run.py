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

n_input = 40
n_steps = 100
n_hidden = 100
n_classes = 10

# TODO:
# extract features
# save to hdf5
# train / test
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

params = {'loss': 'mse', 'optimizer': 'adam'}
input_shape = (batch_size, X_train.shape[1], X_train.shape[2], X_train.shape[3])
model = cnn_rnn.audio_crnn(params, input_shape)

# train model


# define network
# x = tf.placeholder("float", [None, n_steps, n_input])
# y = tf.placeholder("float", [None, n_classes])
#
# weight = tf.Variable(tf.random_normal([n_hidden, n_classes]))
# bias = tf.Variable(tf.random_normal([n_classes]))
#
# prediction = NL.RNN(x, weight, bias, n_hidden)
#
# # Define loss and optimizer
# loss_f = -tf.reduce_sum(y * tf.log(prediction))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_f)
#
# # Evaluate model
# correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
# # Initializing the variables
# init = tf.global_variables_initializer()
#
# with tf.Session() as session:
#     session.run(init)
#
#     for epoch in xrange(training_iters):
#         print("Iter: %s" % epoch)
#         offset = (epoch * batch_size) % (Y_train.shape[0] - batch_size)
#         batch_x = X_train[offset:(offset + batch_size), :, :]
#         batch_y = Y_train[offset:(offset + batch_size), :]
#         _, c = session.run([optimizer, loss_f], feed_dict={x: batch_x, y: batch_y})
#
#         if epoch % display_step == 0:
#             # Calculate batch accuracy
#             acc = session.run(accuracy, feed_dict={x: X_test, y: Y_test})
#             # Calculate batch loss
#             loss = session.run(loss_f, feed_dict={x: X_test, y: Y_test})
#             print "Iter " + str(epoch) + ", Minibatch Loss= " + \
#                   "{:.6f}".format(loss) + ", Test Accuracy= " + \
#                   "{:.5f}".format(acc)
#
#     print('Test accuracy: ', round(session.run(accuracy, feed_dict={x: X_test, y: Y_test}), 3))
