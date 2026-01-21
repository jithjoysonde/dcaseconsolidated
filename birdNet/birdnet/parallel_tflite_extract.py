import os
from glob import glob
import argparse
import multiprocessing as mp
import numpy as np
import soundfile as sf
import joblib

def worker_init(tflite_path, sr, n_mels, win, hop):
    global INTERPRETER, SR, N_MELS, WIN, HOP, TORCH_AVAILABLE, torch, torchaudio, mel_transform, db_transform, device
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError('TensorFlow is required in worker: %s' % e)
    INTERPRETER = tf.lite.Interpreter(model_path=tflite_path)
    INTERPRETER.allocate_tensors()
    SR = sr
    N_MELS = n_mels
    WIN = win
    HOP = hop
    # optional GPU preprocessing using torch/torchaudio
    try:
        import torch
        import torchaudio
        TORCH_AVAILABLE = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=WIN, hop_length=HOP, n_mels=N_MELS, power=2.0).to(device)
        db_transform = torchaudio.transforms.AmplitudeToDB(stype='power').to(device)
        torch.set_num_threads(1)
    except Exception:
        TORCH_AVAILABLE = False


def process_file(path_out):
    wav_path, out_dir = path_out
    try:
        # read
        y, sr = sf.read(wav_path, dtype='float32')
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != SR:
            # fallback to librosa if torch not available
            if TORCH_AVAILABLE:
                import torchaudio
                y = torchaudio.functional.resample(torch.tensor(y), sr, SR).numpy()
            else:
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=SR)

        # prepare input for interpreter
        input_details = INTERPRETER.get_input_details()[0]
        in_shape = input_details['shape']
        if len(in_shape) == 2 or (len(in_shape) == 1):
            # model expects raw waveform
            target = int(in_shape[-1])
            wav = y
            if wav.shape[0] < target:
                pad = np.zeros(target - wav.shape[0], dtype=np.float32)
                wav = np.concatenate([wav, pad])
            else:
                wav = wav[:target]
            inp = wav.reshape(1, -1).astype(np.float32)
        else:
            # compute mel spectrogram (use torch if available)
            if TORCH_AVAILABLE:
                import torch
                wav_t = torch.from_numpy(y).to(device)
                spec = mel_transform(wav_t)
                spec_db = db_transform(spec)
                S = spec_db.cpu().numpy()
                # S shape (n_mels, frames)
                S = S.T.astype(np.float32)
            else:
                import librosa
                S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=WIN, hop_length=HOP, n_mels=N_MELS, power=2.0)
                import librosa
                S = librosa.power_to_db(S, ref=np.max)
                S = S.T.astype(np.float32)
            target_frames = int(in_shape[1])
            if S.shape[0] < target_frames:
                pad = np.zeros((target_frames - S.shape[0], S.shape[1]), dtype=np.float32)
                S = np.vstack([S, pad])
            else:
                S = S[:target_frames]
            inp = S[np.newaxis, :, :, np.newaxis].astype(np.float32)

        INTERPRETER.set_tensor(input_details['index'], inp)
        INTERPRETER.invoke()
        out = INTERPRETER.get_tensor(INTERPRETER.get_output_details()[0]['index'])
        base = os.path.splitext(os.path.basename(wav_path))[0]
        out_path = os.path.join(out_dir, base + '.npy')
        np.save(out_path, out.squeeze())
        return (wav_path, out_path)
    except Exception as e:
        return (wav_path, None, str(e))


def main(args):
    wavs = sorted(glob(os.path.join(args.wav_dir, '**', '*.wav'), recursive=True))
    os.makedirs(args.out_dir, exist_ok=True)
    # prepare inputs list
    inputs = [(p, args.out_dir) for p in wavs]
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(processes=args.workers, initializer=worker_init,
                    initargs=(args.tflite, args.sr, args.n_mels, args.win, args.hop))
    results = pool.map(process_file, inputs)
    pool.close(); pool.join()
    meta = [r for r in results if r and len(r) >= 2 and r[1] is not None]
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
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    main(args)
