import glob
import os
import librosa
import numpy as np
import keras.backend as K
import h5py
import features as F

np.random.seed(777)


class HDFWriter(object):
    """
    Save audio features in single hdf5 file storage
    """

    def __init__(self, file_name):
        self.hdf = h5py.File(file_name, "w")

    def append(self, file_id, tag, feat):
        """
        file_id: unique identifier of the audio file
        tag: hot-encoded 1D array, where '1' marks class on
        """
        data = self.hdf.create_dataset(name=file_id, data=feat)
        data.attrs['tag'] = tag

    def close(self):
        self.hdf.close()

    @staticmethod
    def load_data(file_name):
        """
        Load all datasets from hdf5 to the memory
        NOTE: not preferred to run for large dataset
        """
        hdf = h5py.File(file_name, "r")
        files = list(hdf.keys())
        print('Files in dataset: %d' % len(files))
        X, Y = [], []
        for fn in hdf:
            X.append(np.array(hdf[fn]))
            Y.append(hdf[fn].attrs['tag'])
        hdf.close()
        return np.array(X), np.array(Y)


# def batch_handler(type):
#     if type == "seq_slide_wnd":
#         pass
#     elif type == "rnd_wnd":
#         pass
#     elif type == "eval":
#         pass
#     elif type == "mtag_oversmp":
#         pass
#     elif type == "stag_oversmp":
#         pass
#     else:
#         raise ValueError("Unknown batch type [" + type + "]")


class MiniBatchGenerator(object):
    """
    Generate mini-batches from HDF5 data set.
    Input: hdf5 storage of datasets with attributes as labels:
    hdf5 datasets (2D[3D] arrays) [bands; frames; [channel]] with
    attributes as labels (1D array)
    """

    def __init__(self, file_name, window, batch_sz=1, batch_type="seq_slide_wnd", data_stat=None):
        """
        batch_type:
            "seq_slide_wnd" - cut sequential chunks from the taken file
            "rnd_wnd" - cut random chunk from the file, then choose another file
            "mtag_oversmp" - add more rare samples, regarding multiple tags statistics 'data_stat'
            "stag_oversmp" - add more rare samples, regarding single tags statistics 'data_stat'
            "eval" - slice one file and return samples to do evaluation
        data_stat: {'train':{
                        tags1:[file_id1, file_id2, file_id3, ...]
                        tags2:[file_id5, file_id6, file_id10, ...]
                        ...}
                    'test':{
                        tags1:[file_id1, file_id2, file_id3, ...]
                        tags2:[file_id5, file_id1, file_id3, ...]
                        ...}}
        """
        self.hdf = h5py.File(file_name, "r")
        self.batch_sz = batch_sz
        self.window = window
        self.batch_type = batch_type
        self.data_stat = data_stat

        self.fnames = list(self.hdf.keys())
        print('Files in dataset: %d' % len(self.fnames))

        if batch_type == "mtag_oversmp" and data_stat:
            self._files_oversampling()

    def batch(self):
        """
        Slice HDF5 datasets and return batch: [smp x bands x frames [x channel]]
        """
        if self.batch_type == "seq_slide_wnd":
            # iterate over datasets until we fill the batch
            # TODO: see dwt feature extraction to fix 'last' variable here: [Monday]
            # while 1:
            cnt = 0
            X, Y = [], []
            np.random.shuffle(self.fnames)
            for ifn, fn in enumerate(self.fnames):
                dset = np.array(self.hdf[fn], dtype=np.float32)
                dim, N, ch = dset.shape
                last = min(N // self.window * self.window, N)
                for start in xrange(0, last, self.window):
                    if cnt < self.batch_sz:
                        X.append(dset[:, start:start + self.window, :])
                        Y.append(self.hdf[fn].attrs['tag'])
                        cnt += 1
                    else:
                        if K.image_dim_ordering() == 'th': # permute
                            X = np.array(X)
                            X = np.transpose(X, (0, 3, 1, 2))
                            Y = np.array(Y)
                            Y = Y[:, np.newaxis, :]
                        yield X, Y
                        cnt = 0
                        X, Y = [], []
        elif self.batch_type == "rnd_wnd" or self.batch_type == "mtag_oversmp":
            cnt = 0
            X, Y = [], []
            np.random.shuffle(self.fnames)
            for ifn, fn in enumerate(self.fnames):
                if cnt < self.batch_sz:
                    dset = np.array(self.hdf[fn], dtype=np.float32)
                    dim, N, ch = dset.shape
                    start = np.random.randint(0, N - self.window)
                    X.append(dset[:, start:start + self.window, :])
                    Y.append(self.hdf[fn].attrs['tag'])
                    cnt += 1
                else:
                    yield np.array(X), np.array(Y)
                    cnt = 0
                    X, Y = [], []
        elif self.batch_type == "stratified":
            # TODO implement!!!
            pass
        elif self.batch_type == "eval":
            X, Y = [], []
            for ifn, fn in enumerate(self.fnames):
                dset = np.array(self.hdf[fn], dtype=np.float32)
                dim, N, ch = dset.shape
                last = min(N // self.window * self.window, N)
                if last == 0:
                    print("[INFO] file %s shorter than frame context: %d" % (fn, N))
                else:
                    for start in xrange(0, last, self.window):
                        X.append(dset[:, start:start + self.window, :])
                        Y.append(self.hdf[fn].attrs['tag'])
                    if K.image_dim_ordering() == 'th':  # permute
                        X = np.array(X)
                        X = np.transpose(X, (0, 3, 1, 2))
                        Y = np.array(Y)
                        Y = Y[:, np.newaxis, :]
                    yield fn, X, Y
                X, Y = [], []
        else:
            raise Exception("There is no such generator type...")

    def batch_shape(self):
        """
        NOTE: dataset always is in Tensorflow order initially [bands, frames, channels]
        :return:
            Tensorflow:
                3D data [batch_sz; band; frame_wnd; channel]
                2D data [batch_sz; band; frame_wnd]
            Theano:
                3D data [batch_sz; channel; band; frame_wnd]
                2D data [batch_sz; band; frame_wnd] ? check
        """
        sh = np.array(self.hdf[self.fnames[0]]).shape
        if len(sh) == 3:
            bands, _, channels = sh
            assert channels >= 1
            if K.image_dim_ordering() == 'th':
                # [batch_sz; channel; band; frame_wnd]
                return self.batch_sz, channels, bands, self.window
            else:
                # [batch_sz; band; frame_wnd; channel]
                return self.batch_sz, bands, self.window, channels
        if len(sh) == 2:
            bands, _ = sh
            return self.batch_sz, bands, self.window

    def stop(self):
        self.hdf.close()

    def _files_oversampling(self):
        """
        Add more rare classes to the file list
        """
        # training set data statistic
        mtag_stat = self.data_stat['train']  # {'tags': [file_names], ...}
        # TODO check different settings
        N = int(np.mean(map(len, mtag_stat.values())))  # number of samples to add
        # sampling files
        for ts, fs in mtag_stat.items():
            if ts.lower() == 's': continue
            if len(fs) < N:
                add = N - len(fs)
                c = np.random.choice(fs, add)
                self.fnames.extend(c)
        np.random.shuffle(self.fnames)

    def _uniformsmp(self, btch_sz):
        """
        Uniformly sample for each class, given tags statistic
        """
        # training set data statistic
        mtag_stat = self.data_stat['train']  # {'tags': [file_names], ...}
        N = mtag_stat.keys()
        perC = btch_sz // N
        if N > btch_sz:
            print("[info] batch size is smaller than number of classes...")
        # sampling files
        btch_fs = []
        for ts, fs in mtag_stat.items():
            if ts.lower() == 's': continue
            c = np.random.choice(fs, perC)
            btch_fs.append(c)
        return btch_fs


def do_feature_extraction(ftype, feat_params, parent_dir, sub_dirs, file_ext="*.wav", feat_file='./data/train_set.h5'):
    fn_lab_pairs = parse_audio(parent_dir, sub_dirs, file_ext)
    # prepare extractor
    extractor = F.prepare_extractor(ftype, feat_params)
    writer = HDFWriter(file_name=feat_file)

    for fn, lab in fn_lab_pairs.items():
        x, fs = librosa.load(fn)
        feat = extractor.extract(x, fs)
        file_id = os.path.basename(fn).split(".")[0]  # file name without ext
        # binary tag
        tag_id = tag_hot_encoding(lab, 10)
        # dump features
        writer.append(file_id, tag_id, feat)
        print("Processed: %s" % fn)
    writer.close()
    print("Files processed: %d" % len(fn_lab_pairs))


def parse_audio(parent_dir, sub_dirs, file_ext):
    labels = {}
    for l, sub_dir in enumerate(sub_dirs):
        print("Processing: %s" % l)
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            # print("%s" % fn)
            labels[fn] = int(fn.split('/')[-1].split('-')[1])
    return labels


def tag_hot_encoding(tag, n_tags):
    """
    tag: integer label
    Return a unit vector (dim = # of tags) with a 1.0 in the jth
    position of the tag and zeroes elsewhere"""
    hots = np.zeros(n_tags, dtype=np.uint8)
    hots[tag] = 1
    return hots
