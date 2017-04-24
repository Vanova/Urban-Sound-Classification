import utils.urban_loader as urb_load
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import utils.nnet_lib as NL

DATASET_BASE_PATH = '/home/vano/wrkdir/datasets/UrbanSound8K/audio/'
#########
# load features
#########
# training set
tr_sub_dirs = ['fold1_dwnsmp']
X_train, Y_train = urb_load.extract_fbank_feat(DATASET_BASE_PATH, tr_sub_dirs)
Y_train = urb_load.one_hot_encode(Y_train)
# test set
ts_sub_dirs = ['fold2_dwnsmp']
X_test, Y_test = urb_load.extract_fbank_feat(DATASET_BASE_PATH, ts_sub_dirs)
Y_test = urb_load.one_hot_encode(Y_test)

# network parameters
frames = 41
bands = 60

feature_size = bands * frames  # 60x41 = 2460
num_labels = 10
num_channels = 2

batch_size = 50
kernel_size = 30
depth = 20
num_hidden = 200

learning_rate = 0.01
# total_iterations = 2000
total_iterations = 20
# ===
# net structure
# ===
X = tf.placeholder(tf.float32, shape=[None, bands, frames, num_channels])
Y = tf.placeholder(tf.float32, shape=[None, num_labels])

conv = NL.apply_convolution(X, kernel_size, num_channels, depth)

shape = conv.get_shape().as_list()
conv_flat = tf.reshape(conv, [-1, shape[1] * shape[2] * shape[3]])

f_weights = NL.weight_variable([shape[1] * shape[2] * depth, num_hidden])
f_biases = NL.bias_variable([num_hidden])
f = tf.nn.sigmoid(tf.add(tf.matmul(conv_flat, f_weights), f_biases))

out_weights = NL.weight_variable([num_hidden, num_labels])
out_biases = NL.bias_variable([num_labels])
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

# loss function
loss = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# optimization
cost_history = np.empty(shape=[1], dtype=float)
with tf.Session() as session:
    tf.global_variables_initializer().run()

    for itr in range(total_iterations):
        print("Iteration: %d" % itr)
        offset = (itr * batch_size) % (Y_train.shape[0] - batch_size)
        batch_x = X_train[offset:(offset + batch_size), :, :, :]
        batch_y = Y_train[offset:(offset + batch_size), :]

        _, c = session.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
        cost_history = np.append(cost_history, c)

    print('Test accuracy: ', round(session.run(accuracy, feed_dict={X: X_test[:1000], Y: Y_test[:1000]}), 3))
    fig = plt.figure(figsize=(15, 10))
    plt.plot(cost_history)
    plt.axis([0, total_iterations, 0, np.max(cost_history)])
    plt.show()

# TODO implement fix_generator + tensorboard
# TODO batch_generator