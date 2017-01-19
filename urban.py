"""
Try out multi-tagging detection (DCASE challenge) based on
the Urban Sound Classification
"""
import visialiser as vis
import numpy as np
import urban_loader

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

# load dataset
sub_dirs = ['fold1', 'fold2', 'fold3']
features, labels = urban_loader.parse_audio_files(DATASET_BASE_PATH, sub_dirs)

labels = urban_loader.one_hot_encode(labels)

train_test_split = np.random.rand(len(features)) < 0.70
train_x = features[train_test_split]
train_y = labels[train_test_split]
test_x = features[~train_test_split]
test_y = labels[~train_test_split]




