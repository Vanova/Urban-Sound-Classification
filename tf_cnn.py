import utils.urban_loader as urb_load
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import utils.nnet_helper as netH

DATASET_BASE_PATH = '/home/vano/wrkdir/Datasets/UrbanSound8K/audio/'
#########
# load features
#########
# training set
tr_sub_dirs = ['fold1_dwnsmp', 'fold2_dwnsmp']
tr_features, tr_labels = urb_load.extract_fbank_feat(DATASET_BASE_PATH, tr_sub_dirs)
tr_labels = urb_load.one_hot_encode(tr_labels)
# test set
ts_sub_dirs = ['fold3_dwnsmp']
ts_features, ts_labels = urb_load.extract_fbank_feat(DATASET_BASE_PATH, ts_sub_dirs)
ts_labels = urb_load.one_hot_encode(ts_labels)

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

# net structure
X = tf.placeholder(tf.float32, shape=[None, bands, frames, num_channels])
Y = tf.placeholder(tf.float32, shape=[None, num_labels])

cov = netH.apply_convolution(X, kernel_size, num_channels, depth)

shape = cov.get_shape().as_list()
cov_flat = tf.reshape(cov, [-1, shape[1] * shape[2] * shape[3]])

f_weights = netH.weight_variable([shape[1] * shape[2] * depth, num_hidden])
f_biases = netH.bias_variable([num_hidden])
f = tf.nn.sigmoid(tf.add(tf.matmul(cov_flat, f_weights), f_biases))

out_weights = netH.weight_variable([num_hidden, num_labels])
out_biases = netH.bias_variable([num_labels])
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
        offset = (itr * batch_size) % (tr_labels.shape[0] - batch_size)
        batch_x = tr_features[offset:(offset + batch_size), :, :, :]
        batch_y = tr_labels[offset:(offset + batch_size), :]

        _, c = session.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
        cost_history = np.append(cost_history, c)

    print('Test accuracy: ', round(session.run(accuracy, feed_dict={X: ts_features, Y: ts_labels}), 3))
    fig = plt.figure(figsize=(15, 10))
    plt.plot(cost_history)
    plt.axis([0, total_iterations, 0, np.max(cost_history)])
    plt.show()
