import glob
import os
import librosa
import numpy as np
import soundfile as sf


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


def parse_audio_files(parent_dir, sub_dirs, file_ext='*.wav'):
    features, labels = np.empty((0, 193)), np.empty(0)
    counter = 0
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
                ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
                features = np.vstack([features, ext_features])
                labels = np.append(labels, fn.split('/')[-1].split('-')[1])
                counter += 1
                print("{} processed".format(fn))
            except Exception:
                print("Bad file {}".format(fn))
                pass
        print("Files processed from {}: {}".format(sub_dir, counter))
    return np.array(features), np.array(labels, dtype=np.int)


def extract_fbank_feat(parent_dir, sub_dirs, file_ext="*.wav", bands=60, frames=41):
    """
    :return: 4D array (features) and 1D array (labels);
    feature shape [smp; band; frames; channel]
    """
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        print("Processing: %s" % l)
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            print(fn)
            sound_clip, s = librosa.load(fn)
            label = fn.split('/')[-1].split('-')[1]
            for (start, end) in windows(sound_clip, window_size):
                if (len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
                    logspec = librosa.logamplitude(melspec)
                    logspec = logspec.T.flatten()[:, np.newaxis].T
                    log_specgrams.append(logspec)
                    labels.append(label)

    log_specgrams = np.asarray(log_specgrams). \
        reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    return np.array(features), np.array(labels, dtype=np.int)


def extract_mfcc_features(parent_dir, sub_dirs, file_ext="*.wav", bands=20, frames=41):
    """
    :return: 3D array (features) and 1D array (labels);
    feature shape [smp; band; frames]
    """
    window_size = 512 * (frames - 1)
    mfccs = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        print("Processing: %s" % l)
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            print("%s" % fn)
            sound_clip, s = librosa.load(fn)
            label = fn.split('/')[-1].split('-')[1]
            for (start, end) in windows(sound_clip, window_size):
                if (len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc=bands).T.flatten()[:, np.newaxis].T
                    mfccs.append(mfcc)
                    labels.append(label)
    features = np.asarray(mfccs).reshape(len(mfccs), bands, frames)
    return np.array(features), np.array(labels, dtype=np.int)


def windows(data, window_size):
    """
    Return chunk of sound wave signal
    data: 1D array wave signal
    window_size: wave chunk length
    """
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)



# class BatchGenerator(object):
#     def __init__(self):
#         pass
#
#
# class Dataset(object):
#     """
#     Load data from hdf5 format
#     """
#     def __init__(self):
#         pass
#
#
# def feature_extractor():
#     pass

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot = np.zeros((n_labels, n_unique_labels))
    one_hot[np.arange(n_labels), labels] = 1
    return one_hot
