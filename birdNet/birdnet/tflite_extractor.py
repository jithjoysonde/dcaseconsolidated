import os
import argparse
from glob import glob
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import joblib

try:
    import tensorflow as tf
except Exception as e:
    raise RuntimeError('TensorFlow required to run TFLite extractor: %s' % e)


def prepare_input(interpreter, wav, sr, n_mels=128, win=1200, hop=240):
    input_details = interpreter.get_input_details()[0]
    shape = input_details['shape']
    # if model expects waveform: shape like [1, N]
    if len(shape) == 2 or (len(shape) == 1):
        target = int(shape[-1])
        if wav.shape[0] < target:
            pad = np.zeros(target - wav.shape[0], dtype=np.float32)
            wavp = np.concatenate([wav, pad])
        else:
            wavp = wav[:target]
        return wavp.reshape(1, -1).astype(np.float32)
    # assume spectrogram input [1, frames, n_mels, 1]
    target_frames = int(shape[1])
    target_mels = int(shape[2])
    S = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=win, hop_length=hop, n_mels=target_mels, power=2.0)
    S = librosa.power_to_db(S, ref=np.max)
    S = (S - S.mean()) / (S.std() + 1e-9)
    S = S.T.astype(np.float32)
    if S.shape[0] < target_frames:
        pad = np.zeros((target_frames - S.shape[0], S.shape[1]), dtype=np.float32)
        S = np.vstack([S, pad])
    else:
        S = S[:target_frames]
    return S[np.newaxis, :, :, np.newaxis]


def main(args):
    interpreter = tf.lite.Interpreter(model_path=args.tflite)
    interpreter.allocate_tensors()
    wavs = sorted(glob(os.path.join(args.wav_dir, '**', '*.wav'), recursive=True))
    os.makedirs(args.out_dir, exist_ok=True)
    meta = []
    for p in tqdm(wavs):
        try:
            y, sr = sf.read(p, dtype='float32')
            if sr != args.sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=args.sr)
            inp = prepare_input(interpreter, y, args.sr, n_mels=args.n_mels, win=args.win, hop=args.hop)
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'], inp)
            interpreter.invoke()
            out = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
            base = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(args.out_dir, base + '.npy')
            np.save(out_path, out.squeeze())
            meta.append((p, out_path))
        except Exception as e:
            print('skip', p, e)
    joblib.dump(meta, os.path.join(args.out_dir, 'meta.pkl'))
    print('Saved', len(meta), 'embeddings to', args.out_dir)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--wav-dir', required=True)
    p.add_argument('--tflite', required=True)
    p.add_argument('--out-dir', default='birdnet/features')
    p.add_argument('--sr', type=int, default=48000)
    p.add_argument('--n-mels', type=int, default=128)
    p.add_argument('--hop', type=int, default=240)
    p.add_argument('--win', type=int, default=1200)
    args = p.parse_args()
    main(args)
