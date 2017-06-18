'''
Reference: [Music-auto_tagging-keras](https://github.com/keunwoochoi/music-auto_tagging-keras)
Note: to make attention work install seq2seq and
recurrentShop library, see working versions on my github repository
'''
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Reshape, Permute, Input
from keras.optimizers import Adam, SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from seq2seq import AttentionSeq2Seq
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import GRU
import numpy as np
np.random.seed(777)


def audio_crnn(params, input_shape, nclass, include_top=True):
    """
    The CNN dim equation: (width - kernel_size + 2*pad)/stride +1
    input shape: [batch_sz; band; frame_wnd; channel]
    Based on idea from [Music-auto_tagging-keras](https://github.com/keunwoochoi/music-auto_tagging-keras)
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
    x = Convolution2D(params['feature_maps'], 3, 3, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x) # (20, 50, 64)
    x = Dropout(0.1, name='dropout1')(x)

    # Conv block 2
    x = Convolution2D(2 * params['feature_maps'], 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 2), strides=(4, 2), name='pool2')(x)
    x = Dropout(0.1, name='dropout2')(x)

    # Conv block 3
    x = Convolution2D(2 * params['feature_maps'], 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 1), strides=(4, 1), name='pool3')(x)
    x = Dropout(0.1, name='dropout3')(x)

    # Conv block 4
    x = Convolution2D(2 * params['feature_maps'], 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='pool4')(x)
    x = Dropout(0.1, name='dropout4')(x)

    # reshaping
    if K.image_dim_ordering() == 'th':
        x = Permute((3, 1, 2))(x)
    # do convolutions, until all freqs shrink to one
    x = Reshape((-1, 2 * params['feature_maps']))(x)

    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
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


def cnn_rnn_attention(params, input_shape, nclass, include_top=True):
    """
    The CNN dim equation: (width - kernel_size + 2*pad)/stride +1
    input shape: [batch_sz; band; frame_wnd; channel]
    Based on idea from [Music-auto_tagging-keras](https://github.com/keunwoochoi/music-auto_tagging-keras)
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
        channel_axis = 3
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
    x = Convolution2D(params['feature_maps'], 3, 3, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)  # (20, 50, 64)
    x = Dropout(0.1, name='dropout1')(x)

    # Conv block 2
    x = Convolution2D(2 * params['feature_maps'], 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 2), strides=(4, 2), name='pool2')(x)
    x = Dropout(0.1, name='dropout2')(x)

    # Conv block 3
    x = Convolution2D(2 * params['feature_maps'], 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 1), strides=(4, 1), name='pool3')(x)
    x = Dropout(0.1, name='dropout3')(x)

    # Conv block 4
    x = Convolution2D(2 * params['feature_maps'], 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='pool4')(x)
    x = Dropout(0.1, name='dropout4')(x)

    # reshaping
    if K.image_dim_ordering() == 'th':
        x = Permute((3, 1, 2))(x)
    # do convolutions, until all freqs shrink to one
    # x = Reshape((-1, 2 * params['feature_maps']))(x)
    x = Reshape((20, 2 * params['feature_maps']))(x)


    # TODO implement attention layer here
    # Create initial CNN model
    init_model = Model(feat_input, x)
    # Create attention model
    input_length = 20
    input_dim = 2 * params['feature_maps']
    att_model = AttentionSeq2Seq(output_dim=32, output_length=1, input_shape=(input_length, input_dim), bidirectional=True)
    # merge models
    full_model = Sequential()
    full_model.add(init_model)
    # append attention
    full_model.add(att_model)
    full_model.add(Dropout(0.3))
    full_model.add(Dense(nclass, activation='sigmoid', name='output'))
    # GRU block 1, 2, output
    # x = GRU(32, return_sequences=True, name='gru1')(x)
    # x = GRU(32, return_sequences=False, name='gru2')(x)
    # x = Dropout(0.3)(x)
    # if include_top:
    #     x = Dense(nclass, activation='sigmoid', name='output')(x)
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

    full_model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    full_model.summary()

    return full_model