"""
Try out multi-tagging detection (DCASE challenge) based on
the Urban Sound Classification
"""
import visialiser as vis
import urban_loader
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

DATASET_BASE_PATH = '/home/vano/wrkdir/Datasets/UrbanSound8K/audio/'


def vis_examp():
    DATASET_PATH = "/home/vano/wrkdir/Datasets/UrbanSound8K/audio/fold1/"
    sound_file_paths = ["57320-0-0-7.wav", "24074-1-0-3.wav", "15564-2-0-1.wav"]
    sound_names = ["air conditioner", "car horn", "children playing"]
    # add dataset path to file names
    sound_file_paths = [DATASET_PATH + p for p in sound_file_paths]
    raw_sounds = vis.load_sound_files(sound_file_paths)
    # visualise different features
    vis.plot_waves(sound_names, raw_sounds)
    vis.plot_specgram(sound_names, raw_sounds)
    vis.plot_log_power_specgram(sound_names, raw_sounds)


# example visualisation
# vis_examp()

# feature extraction
# TODO saving to hdf5!!!
tr_sub_dirs = ['fold1_dwnsmp', 'fold2_dwnsmp']
ts_sub_dirs = ['fold3_dwnsmp']
tr_features, tr_labels = urban_loader.parse_audio_files(DATASET_BASE_PATH, tr_sub_dirs)
ts_features, ts_labels = urban_loader.parse_audio_files(DATASET_BASE_PATH, ts_sub_dirs)

tr_labels = urban_loader.one_hot_encode(tr_labels)
ts_labels = urban_loader.one_hot_encode(ts_labels)

###
# Training
###
# network settings
training_epochs = 5000
n_dim = tr_features.shape[1]
n_classes = 10
n_hidden_units_one = 280
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01

# network layers
X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)

W = tf.Variable(tf.random_normal([n_hidden_units_two, n_classes], mean=0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2, W) + b)

init = tf.global_variables_initializer()

# Xentropy cost function
cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training
cost_history = np.empty(shape=[1], dtype=float)
y_true, y_pred = None, None
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        _, cost = sess.run([optimizer, cost_function], feed_dict={X: tr_features, Y: tr_labels})
        cost_history = np.append(cost_history, cost)

    y_pred = sess.run(tf.argmax(y_, 1), feed_dict={X: ts_features})
    y_true = sess.run(tf.argmax(ts_labels, 1))
    print('Test accuracy: ', round(sess.run(accuracy, feed_dict={X: ts_features, Y: ts_labels}), 3))

fig = plt.figure(figsize=(10, 8))
plt.plot(cost_history)
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.axis([0, training_epochs, 0, np.max(cost_history)])
plt.show()

p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='micro')
print "F-Score:", round(f, 3)
