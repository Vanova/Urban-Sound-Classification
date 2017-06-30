import librosa
import matplotlib.pyplot as plt
import numpy as np
from keras import models
import keras.layers
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

    def __init__(self, model):
        self.net = model

    def filters(self):
        pass

    def layer_activations(self, batch, lname, show=True):
        # find layer by name
        layer_names = [l.name for l in self.net.layers]
        lid_plt = layer_names.index(lname)
        # extracts the outputs of the bottom layers
        outs = [layer.output for layer in self.net.layers[:lid_plt + 1]]

        # creates a model to forward
        nn2forward = models.Model(input=self.net.input, output=outs)
        # NOTE: activations contain intermediate as well
        acts = nn2forward.predict_on_batch(batch)

        if isinstance(self.net.layers[lid_plt], keras.layers.InputLayer):
            last_act = acts
        else:
            last_act = acts[-1]
        # name and shape of layer to plot
        print(lname, last_act.shape)
        act_plt = self._arrange_dim(last_act)

        plt.imshow(act_plt)
        plt.axis('off')
        plt.grid(False)
        plt.savefig(lname + '.png', dpi=400)
        if show:
            plt.show()

    def all_activations(self, batch):
        outs = [layer.output for layer in self.net.layers]
        # creates a model to forward
        nn_to_forward = models.Model(input=self.net.input, output=outs)
        acts = nn_to_forward.predict_on_batch(batch)

        layer_names = []
        acts_plt = []
        for id, layer in enumerate(self.net.layers):
            if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.MaxPooling2D) or \
                    isinstance(layer, keras.layers.AveragePooling2D):
                layer_names.append(layer.name)
                acts_plt.append(acts[id])

        # display activations
        images_per_row = 16
        for lname, a in zip(layer_names, acts_plt):
            # feature maps have shape: (smps, fbanks, frames, channels)
            if a.shape == 4:
                _, fbank_sz, frames_sz, chan_sz = a.shape
            else:
                raise NotImplementedError("[error] It can not visualize such a layer as a path...")
            print(lname, a.shape)
            # image grid of feature maps
            display_grid = self._tile_grid(a, images_per_row)

            # display the grid
            scale = 2
            plt.figure(
                figsize=(scale * 1. / frames_sz * display_grid.shape[1], scale * 1. / fbank_sz * display_grid.shape[0]))
            plt.title(lname)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.savefig(lname + '.png', dpi=400)
            # plt.show()

    def class_activation_maps(self):
        """
        Class activation maps
        """
        pass

    @staticmethod
    def is_visualizable(layer):
        if isinstance(layer, keras.layers.InputLayer) or \
                isinstance(layer, keras.layers.Conv2D) or \
                isinstance(layer, keras.layers.MaxPooling2D) or \
                isinstance(layer, keras.layers.AveragePooling2D) or \
                isinstance(layer, keras.layers.GRU) or \
                isinstance(layer, keras.layers.Dense):
            return True

    def _tile_grid(self, a, img_per_row, grid=False):
        sample_id = 4
        _, fbank_sz, frames_sz, chan_sz = a.shape
        n_cols = chan_sz / img_per_row
        display_grid = np.zeros((n_cols * fbank_sz, img_per_row * frames_sz))

        # tile each filter into big horizontal grid
        for row in range(n_cols):
            for col in range(img_per_row):
                channel_image = a[sample_id, :, :, row * img_per_row + col]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[row * fbank_sz: (row + 1) * fbank_sz,
                col * frames_sz: (col + 1) * frames_sz] = channel_image
        return display_grid

    def _arrange_dim(self, array):
        """
        array: ndarray
        :return: return matrix to visualize
        """
        if len(array.shape) == 2:
            # [smp, class], e.g. last layer
            return array[:, :]
        elif len(array.shape) == 3:
            # e.g. for GRU: [smp, time, out_dim]
            # (5, 20, 32)
            return array[:, :, 0]
        elif len(array.shape) == 4:
            # [smp, w, h, ch]
            return array[0, :, :, 0]

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
