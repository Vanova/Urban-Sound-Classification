import utils.urban_loader as urb_load
import tensorflow as tf
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
total_iterations = 2000

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
