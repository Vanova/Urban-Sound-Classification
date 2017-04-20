import scipy
import numpy as np
from scipy import signal
from pywt import wavedec
import librosa

# TODO fix according to factory pattern: https://krzysztofzuraw.com/blog/2016/factory-pattern-python.html

def prepare_extractor(feats='mfcc', params=None):
    if feats == 'mfcc':
        return MFCCBaseExtractor(params)
    elif feats == 'fbank':
        return FbankBaseExtractor(params)
    elif feats == 'stft':
        return STFTBaseExtractor(params)
    elif feats == 'cqt':
        return CQTBaseExtractor(params)
    elif feats == 'cwt':
        return CWTBaseExtractor(params)
    elif feats == 'dwt':
        return DWTBaseExtractor(params)
    elif feats == 'raw_signal':
        return RawSignalBaseExtractor(params)
    else:
        raise ValueError("Unknown feature type [" + feats + "]")


class BaseExtractor(object):
    eps = np.spacing(1)

    def feat_dim(self):
        return

    def extract(self, x, smp_rate):
        return

    def _window(self, wtype, smp_sz):
        window = None
        # Windowing function
        if wtype == 'hamming_asymmetric':
            window = scipy.signal.hamming(smp_sz, sym=False)
        elif wtype == 'hamming_symmetric':
            window = scipy.signal.hamming(smp_sz, sym=True)
        elif wtype == 'hann_asymmetric':
            window = scipy.signal.hann(smp_sz, sym=False)
        elif wtype == 'hann_symmetric':
            window = scipy.signal.hann(smp_sz, sym=True)
        return window

    def _subsequence(self, x, wnd):
        """
        Chunk the sequence x with 50% overlapping
        """
        for index in xrange(0, len(x) - wnd + 1, wnd//2):
            yield x[index:index + wnd]


class MFCCBaseExtractor(BaseExtractor):
    def __init__(self, params=None):
        self.params = params

    def extract(self, x, smp_rate):
        """
        x: wave 1D signal
        return: 2D array, [frames; dimension]
        """
        # Extract features, Mel Frequency Cepstral Coefficients
        wnd = self._window(wtype=self.params['window'], smp_sz=self.params['n_fft'])
        # calculate static mfss coefficients
        stft = np.abs(librosa.stft(x + self.eps, n_fft=self.params['n_fft'], win_length=self.params['win_length'],
                                   hop_length=self.params['hop_length'], window=wnd)) ** 2

        mel_basis = librosa.filters.mel(sr=smp_rate, n_fft=self.params['n_fft'], n_mels=self.params['n_mels'],
                                        fmin=self.params['fmin'], fmax=self.params['fmax'], htk=self.params['htk'])
        stft_windowed = np.dot(mel_basis, stft)
        mfcc = librosa.feature.mfcc(S=librosa.logamplitude(stft_windowed))

        emfcc = mfcc[:, :, np.newaxis]  # np.expand_dims(logmel, axis=2)
        sh = emfcc.shape
        # consider delta features as the 2nd "image" channel
        if self.params['include_delta']:
            # append delta features as an additional channel: [bands; frames; channels]
            buf = np.zeros(sh)
            emfcc = np.concatenate((emfcc, buf), axis=2)
            emfcc[:, :, 1] = librosa.feature.delta(emfcc[:, :, 0], **self.params['mfcc_delta'])
        if self.params['include_acceleration']:
            buf = np.zeros(sh)
            emfcc = np.concatenate((emfcc, buf), axis=2)
            emfcc[:, :, 2] = librosa.feature.delta(emfcc[:, :, 1], **self.params['mfcc_acceleration'])
        return emfcc


class FbankBaseExtractor(BaseExtractor):
    """
    Log-Mel filter bank feature extractor, steps:
    1. Frame the signal into short frames.
    2. For each frame calculate the periodogram estimate of the power spectrum.
    3. Apply the mel filterbank to the power spectra, sum the energy in each filter.
    4. Take the logarithm of all filterbank energies.
    """

    def __init__(self, params=None):
        self.params = params

    def extract(self, x, smp_rate):
        """
        return: 3D array, features [bands; frames; channels]
        """
        # Extract features, Mel Frequency Cepstral Coefficients
        wnd = self._window(wtype=self.params['window'], smp_sz=self.params['n_fft'])
        # calculate static mfss coefficients
        stft = np.abs(librosa.stft(x + self.eps, n_fft=self.params['n_fft'], win_length=self.params['win_length'],
                                   hop_length=self.params['hop_length'], window=wnd)) ** 2

        mel_basis = librosa.filters.mel(sr=smp_rate, n_fft=self.params['n_fft'], n_mels=self.params['bands'],
                                        fmin=self.params['fmin'], fmax=self.params['fmax'], htk=self.params['htk'])
        melspec = np.dot(mel_basis, stft)
        logmel = librosa.logamplitude(melspec)
        elogmel = logmel[:, :, np.newaxis]  # np.expand_dims(logmel, axis=2)
        sh = elogmel.shape
        # consider delta features as the 2nd "image" channel
        if self.params['include_delta']:
            # append delta features as an additional channel: [bands; frames; channels]
            buf = np.zeros(sh)
            elogmel = np.concatenate((elogmel, buf), axis=2)
            elogmel[:, :, 1] = librosa.feature.delta(elogmel[:, :, 0], **self.params['delta'])
        if self.params['include_acceleration']:
            buf = np.zeros(sh)
            elogmel = np.concatenate((elogmel, buf), axis=2)
            elogmel[:, :, 2] = librosa.feature.delta(elogmel[:, :, 1], **self.params['acceleration'])
        return elogmel


class STFTBaseExtractor(BaseExtractor):
    def __init__(self, params=None):
        self.params = params

    def extract(self, x, smp_rate):
        """
        return: 3D array, features [bands; frames; channels] for Tensorflow
        """
        # Extract features, Mel Frequency Cepstral Coefficients
        wnd = self._window(wtype=self.params['window'], smp_sz=self.params['n_fft'])
        # calculate static mfss coefficients
        stft = np.abs(librosa.stft(x + self.eps, n_fft=self.params['n_fft'], win_length=self.params['win_length'],
                                   hop_length=self.params['hop_length'], window=wnd)) ** 2
        estft = np.expand_dims(stft, axis=2)
        sh = estft.shape
        # consider delta features as the 2nd "image" channel
        if self.params['include_delta']:
            # append delta features as an additional channel: [bands; frames; channels]
            buf = np.zeros(sh)
            estft = np.concatenate((estft, buf), axis=2)
            estft[:, :, 1] = librosa.feature.delta(estft[:, :, 0])
        if self.params['include_acceleration']:
            buf = np.zeros(sh)
            estft = np.concatenate((estft, buf), axis=2)
            estft[:, :, 2] = librosa.feature.delta(estft[:, :, 1])
        return estft


class CQTBaseExtractor(BaseExtractor):
    def __init__(self, params=None):
        self.params = params

    def extract(self, x, smp_rate):
        # TODO check librosa version
        cqt = librosa.cqt(x, sr=smp_rate, fmin=librosa.note_to_hz('C1'), n_bins=self.params['bands'],
                          bins_per_octave=self.params['bins_po'])
        # cqt = librosa.cqt(x, sr=smp_rate, fmin=librosa.note_to_hz('C1'), n_bins=self.params['bands'] * 2,
        #                       bins_per_octave=self.params['bins_po'] * 2)
        # amp = np.abs(cqt)
        # db_cqt = 10.0 * np.log10(np.maximum(1e-10, amp))
        edb = np.expand_dims(cqt, axis=2)
        return edb


class CWTBaseExtractor(BaseExtractor):
    def __init__(self, params=None):
        self.params = params

    def extract(self, x, smp_rate):
        """
        NOTE: it is VERY COSTLY, cos produce data shape [bands; len(x)]
        return: 3D array, features [bands; frames; channels] for Tensorflow
        """
        widths = np.arange(1, self.params['bands'])
        cwt_dat = signal.cwt(x, signal.ricker, widths)
        ecwt_dat = np.expand_dims(cwt_dat, axis=2)
        return ecwt_dat


class DWTBaseExtractor(BaseExtractor):
    def __init__(self, params=None):
        self.params = params

    def extract(self, x, smp_rate):
        level = self.params['level']
        frames = np.array(list(self._subsequence(x, self.params['n_dwt'])))
        # dec = wavedec(frames, self.params['wavelet'], level=level)
        dec = wavedec(frames, self.params['wavelet'], level=level)
        dec = [d.T for d in dec]
        dwt_dat = np.vstack(dec)
        # amp = np.abs(dwt_dat) ** 2
        # db = 10.0 * np.log10(np.maximum(1e-10, amp))
        edwt_dat = np.expand_dims(dwt_dat, axis=2)
        return edwt_dat


class RawSignalBaseExtractor(BaseExtractor):
    def __init__(self, params=None):
        pass

    def extract(self, x, smp_rate):
        pass
