import os
import sys
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import joblib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import BirdNetExtractor

# Enable GPU for TensorFlow
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f'Found {len(gpus)} GPU(s): {gpus}')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print('No GPU found, using CPU')
except Exception as e:
    print(f'GPU setup error: {e}')


def main(args):
    extractor = BirdNetExtractor(args.model, sr=args.sr, n_mels=args.n_mels,
                                 hop=args.hop, win=args.win, embedding_layer=args.embedding_layer)
    wavs = sorted(glob(os.path.join(args.wav_dir, '**', '*.wav'), recursive=True))
    
    meta = []
    save_in_place = args.save_in_place if hasattr(args, 'save_in_place') else False
    
    for p in tqdm(wavs):
        try:
            # Extract with sliding window (3s windows, 3s hop = no overlap for speed)
            # This gives us ~1 embedding per 3 seconds of audio
            emb = extractor.extract(p, window_size=3.0, hop_size=3.0)
            
            if save_in_place:
                # Save feature file next to audio with _BDnet.npy suffix
                base = os.path.splitext(p)[0]  # Remove .wav extension
                out_path = base + '_BDnet.npy'
            else:
                # Maintain folder structure relative to input directory
                rel_path = os.path.relpath(p, args.wav_dir)
                rel_dir = os.path.dirname(rel_path)
                base = os.path.splitext(os.path.basename(p))[0]
                
                # Create output directory maintaining structure
                out_subdir = os.path.join(args.out_dir, rel_dir)
                os.makedirs(out_subdir, exist_ok=True)
                
                # Save feature file
                out_path = os.path.join(out_subdir, base + '_BDnet.npy')
            
            # Save as (num_windows, 6522) array
            np.save(out_path, emb)
            meta.append((p, out_path))
        except Exception as e:
            print('skip', p, e)
    
    # Save metadata
    if not save_in_place:
        os.makedirs(args.out_dir, exist_ok=True)
        joblib.dump(meta, os.path.join(args.out_dir, 'meta.pkl'))
        print('Saved', len(meta), 'embeddings to', args.out_dir)
    else:
        print('Saved', len(meta), 'embeddings in-place alongside audio files')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--wav-dir', required=True)
    p.add_argument('--model', required=True, help='Path to SavedModel dir or .tflite file or folder')
    p.add_argument('--out-dir', default='birdnet/features', 
                   help='Output directory (ignored if --save-in-place is True)')
    p.add_argument('--save-in-place', action='store_true',
                   help='Save features next to audio files as audiofilename_BDnet.npy')
    p.add_argument('--sr', type=int, default=48000)
    p.add_argument('--n-mels', type=int, default=128)
    p.add_argument('--hop', type=int, default=240)
    p.add_argument('--win', type=int, default=1200)
    p.add_argument('--embedding-layer', default=None)
    args = p.parse_args()
    main(args)
# birdnet/extract_embeddings.py
import os
import argparse
import numpy as np
from glob import glob
from birdnet.feature_extractor import BirdNetExtractor
from tqdm import tqdm
import joblib

def main(args):
    extractor = BirdNetExtractor(args.model, sr=args.sr, n_mels=args.n_mels,
                                 hop=args.hop, win=args.win, embedding_layer=args.embedding_layer)
    wavs = sorted(glob(os.path.join(args.wav_dir, '**', '*.wav'), recursive=True))
    os.makedirs(args.out_dir, exist_ok=True)
    meta = []
    for p in tqdm(wavs):
        try:
            emb = extractor.extract(p)
            base = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(args.out_dir, base + '.npy')
            np.save(out_path, emb)
            meta.append((p, out_path))
        except Exception as e:
            print("skip", p, e)
    joblib.dump(meta, os.path.join(args.out_dir, 'meta.pkl'))
    print("Saved", len(meta), "embeddings to", args.out_dir)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--wav-dir', required=True)
    p.add_argument('--model', required=True, help='Path to Keras SavedModel or .h5')
    p.add_argument('--out-dir', default='birdnet/features')
    p.add_argument('--sr', type=int, default=48000)
    p.add_argument('--n-mels', type=int, default=128)
    p.add_argument('--hop', type=int, default=240)
    p.add_argument('--win', type=int, default=1200)
    p.add_argument('--embedding-layer', default=None)
    args = p.parse_args()
    main(args)