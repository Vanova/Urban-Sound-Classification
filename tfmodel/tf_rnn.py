"""
Based on refs:
https://github.com/aqibsaeed/Urban-Sound-Classification
https://github.com/Vanova/Multilabel-timeseries-classification-with-LSTM
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import utils.urban_loader as urb_load
import utils.nnet_lib as NL
import config as cnfg

plt.style.use('ggplot')

# Network Parameters
tf.reset_default_graph()
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
                                                  frames=n_steps, bands=n_input)
X_train = np.transpose(X_train, (0, 2, 1))
Y_train = urb_load.one_hot_encode(Y_train)

X_test, Y_test = urb_load.extract_mfcc_features(cnfg.DATASET_BASE_PATH, cnfg.TST_SUB_DIRS,
                                                frames=n_steps, bands=n_input)
X_test = np.transpose(X_test, (0, 2, 1))
Y_test = urb_load.one_hot_encode(Y_test)

# define network
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weight = tf.Variable(tf.random_normal([n_hidden, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

prediction = NL.RNN(x, weight, bias, n_hidden)

# Define loss and optimizer
loss_f = -tf.reduce_sum(y * tf.log(prediction))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_f)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    for iter in xrange(training_iters):
        print("Iter: %s" % iter)
        offset = (iter * batch_size) % (Y_train.shape[0] - batch_size)
        batch_x = X_train[offset:(offset + batch_size), :, :]
        batch_y = Y_train[offset:(offset + batch_size), :]
        _, c = session.run([optimizer, loss_f], feed_dict={x: batch_x, y: batch_y})

        if iter % display_step == 0:
            # Calculate batch accuracy
            acc = session.run(accuracy, feed_dict={x: X_test, y: Y_test})
            # Calculate batch loss
            loss = session.run(loss_f, feed_dict={x: X_test, y: Y_test})
            print "Iter " + str(iter) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Test Accuracy= " + \
                  "{:.5f}".format(acc)

    print('Test accuracy: ', round(session.run(accuracy, feed_dict={x: X_test, y: Y_test}), 3))