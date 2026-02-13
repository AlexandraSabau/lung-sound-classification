import numpy as np
import librosa

# EXACT ca în notebook
N_MELS = 128
MAX_LEN = 256

def audio_to_melspec(path, n_mels=N_MELS, max_len=MAX_LEN):
    y, sr = librosa.load(path, sr=22050)
    y = librosa.util.normalize(y)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)

    if S_db.shape[1] < max_len:
        pad_width = max_len - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode="constant")
    else:
        S_db = S_db[:, :max_len]

    return S_db  # (128,256)

def wav_to_model_input(wav_path, return_spec=False):
    spec = audio_to_melspec(wav_path)          # (128,256)
    x = spec[np.newaxis, ..., np.newaxis]      # (1,128,256,1)

    if return_spec:
        return x, spec   # spec e S_db (dB), padded/trunchiat fix ca pentru model
    return x
