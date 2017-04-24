# -*- coding: utf-8 -*-
'''MusicTaggerCRNN model for Keras.

# Reference:

- [Music-auto_tagging-keras](https://github.com/keunwoochoi/music-auto_tagging-keras)

'''
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Permute, Input
from keras.optimizers import Adam, SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import GRU


def MusicTaggerCRNN(weights='msd', input_tensor=None,
                    include_top=True):
    '''Instantiate the MusicTaggerCRNN architecture,
    optionally loading weights pre-trained
    on Million Song Dataset. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    For preparing mel-spectrogram input, see
    `audio_conv_utils.py` in [applications](https://github.com/fchollet/keras/tree/master/keras/applications).
    You will need to install [Librosa](http://librosa.github.io/librosa/)
    to use it.

    # Arguments
        weights: one of `None` (random initialization)
            or "msd" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        include_top: whether to include the 1 fully-connected
            layer (output layer) at the top of the network.
            If False, the network outputs 32-dim features.


    # Returns
        A Keras model instance.
    '''
    if weights not in {'msd', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `msd` '
                         '(pre-training on Million Song Dataset).')

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (1, 96, 1366)
    else:
        input_shape = (96, 1366, 1)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            melgram_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            melgram_input = input_tensor

    # Determine input axis
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
    x = Dropout(0.1, name='dropout1')(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)
    x = Dropout(0.1, name='dropout2')(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)
    x = Dropout(0.1, name='dropout3')(x)

    # Conv block 4
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)
    x = Dropout(0.1, name='dropout4')(x)

    # reshaping
    if K.image_dim_ordering() == 'th':
        x = Permute((3, 1, 2))(x)
    x = Reshape((15, 128))(x)

    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)
    x = Dropout(0.3)(x)
    if include_top:
        x = Dense(50, activation='sigmoid', name='output')(x)

    # Create model
    model = Model(melgram_input, x)
    if weights is None:
        return model
    else:
        # Load input
        if K.image_dim_ordering() == 'tf':
            raise RuntimeError("Please set image_dim_ordering == 'th'."
                               "You can set it at ~/.keras/keras.json")

        model.load_weights('data/music_tagger_crnn_weights_%s.h5' % K._BACKEND,
                           by_name=True)
        return model


def audio_crnn(params, input_shape, nclass, include_top=True):
    """
    The CNN dim equation: (width - kernel_size + 2*pad)/stride +1
    input shape: [batch_sz; band; frame_wnd; channel]
    Ref: based on [Music-auto_tagging-keras](https://github.com/keunwoochoi/music-auto_tagging-keras)
    """
    # determine proper input shape
    print("DNN input shape", input_shape)
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        # time_axis = 3
        batch_sz, channels, bands, frames = input_shape
        assert channels >= 1
        nn_shape = (channels, bands, frames)
    else:
        channel_axis = 3 # TODO check for theano
        freq_axis = 1
        # time_axis = 2
        batch_sz, bands, frames, channels = input_shape
        assert channels >= 1
        nn_shape = (bands, frames, channels)

    # TODO NOTE: we do 3 convolutions for DCASE
    # Input block
    feat_input = Input(shape=nn_shape)
    # x = ZeroPadding2D(padding=(0, 37))(feat_input) # TODO check?
    x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(feat_input)

    # Conv block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x) # (20, 50, 64)
    x = Dropout(0.1, name='dropout1')(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 2), strides=(3, 2), name='pool2')(x)
    x = Dropout(0.1, name='dropout2')(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 2), strides=(4, 2), name='pool3')(x)
    x = Dropout(0.1, name='dropout3')(x)

    # Conv block 4
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='pool4')(x)
    x = Dropout(0.1, name='dropout4')(x)

    # reshaping
    if K.image_dim_ordering() == 'th':
        x = Permute((3, 1, 2))(x)
    # do convolutions, until all freqs shrink to one
    # x = Reshape((-1, 128))(x) # TODO check
    x = Reshape((10, 128))(x)


    # GRU block 1, 2, output
    # x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)
    x = Dropout(0.3)(x)
    if include_top:
        x = Dense(nclass, activation='sigmoid', name='output')(x)
    # Create model
    model = Model(feat_input, x)
    # ===
    # choose loss
    # ===
    loss = params['loss']
    # ===
    # choose optimizer
    # ===
    if params['optimizer'] == 'adam':
        optimizer = Adam(lr=params['learn_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif params['optimizer'] == 'sgd':
        optimizer = SGD(lr=params['learn_rate'], decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optimizer = params['optimizer']

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()
    return model
