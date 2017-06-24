import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import specgram

from utils.urban_loader import load_sound_files

plt.style.use('ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 13


def plot_waves(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ns = len(sound_names)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(ns, 1, i)
        librosa.display.waveplot(np.array(f), sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 1: Waveplot', x=0.5, y=0.915, fontsize=18)
    plt.show()


def plot_specgram(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ns = len(sound_names)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(ns, 1, i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram', x=0.5, y=0.915, fontsize=18)
    plt.show()


def plot_log_power_specgram(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ns = len(sound_names)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(ns, 1, i)
        D = librosa.logamplitude(np.abs(librosa.stft(f)) ** 2, ref_power=np.max)
        librosa.display.specshow(D, x_axis='time', y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram', x=0.5, y=0.915, fontsize=18)
    plt.show()


class KNetworkVisualizer(object):
    """
    Keras networks visualizer
    """

    def __init__(self, ):
        pass

    def plot_filters(self):
        pass

    def plot_activations(self, input):
        pass




if __name__ == '__main__':
    DATASET_PATH = "/home/vano/wrkdir/Datasets/UrbanSound8K/audio/fold1_dwnsmp/"

    sound_file_paths = ["7061-6-0-0.wav", "57320-0-0-7.wav", "24074-1-0-3.wav", "15564-2-0-1.wav"]
    sound_names = ["gun shot", "air conditioner", "car horn", "children playing"]

    # add dataset path to file names
    sound_file_paths = [DATASET_PATH + p for p in sound_file_paths]
    raw_sounds = load_sound_files(sound_file_paths)

    plot_waves(sound_names, raw_sounds)
    plot_specgram(sound_names, raw_sounds)
    plot_log_power_specgram(sound_names, raw_sounds)
