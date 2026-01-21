#!/usr/bin/env python
"""
Extract BirdNet embeddings from audio segments (sliding window approach).
This creates features for each time window in the audio file.
"""

import os
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import joblib
import soundfile as sf

from birdnet.feature_extractor import BirdNetExtractor


def extract_segments(extractor, wav_path, segment_len=3.0, hop_len=1.5):
    """
    Extract embeddings from overlapping segments of an audio file.
    
    Args:
        extractor: BirdNetExtractor instance
        wav_path: Path to audio file
        segment_len: Length of each segment in seconds
        hop_len: Hop length between segments in seconds
    
    Returns:
        List of (embedding, start_time, end_time) tuples
    """
    # Load audio
    y, sr = sf.read(wav_path, dtype='float32')
    
    # Handle stereo
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    
    # Resample if needed
    if sr != extractor.sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=extractor.sr)
        sr = extractor.sr
    
    # Calculate segment parameters in samples
    segment_samples = int(segment_len * sr)
    hop_samples = int(hop_len * sr)
    
    segments = []
    start_sample = 0
    
    while start_sample + segment_samples <= len(y):
        end_sample = start_sample + segment_samples
        segment = y[start_sample:end_sample]
        
        # Calculate time in seconds
        start_time = start_sample / sr
        end_time = end_sample / sr
        
        # Extract features for this segment
        try:
            # Preprocess segment
            S, y_seg = extractor.preprocess(segment, sr)
            
            if extractor.backend == 'tf':
                inp = np.expand_dims(S, axis=0)[..., np.newaxis]
                emb = extractor.embedding_model(inp, training=False).numpy().squeeze()
            else:
                inp = extractor._prepare_for_tflite(S, y_seg)
                extractor.interpreter.set_tensor(extractor.input_details[0]['index'], inp)
                extractor.interpreter.invoke()
                emb = extractor.interpreter.get_tensor(extractor.output_details[0]['index']).squeeze()
            
            segments.append((emb, start_time, end_time))
        except Exception as e:
            print(f"Warning: Failed to extract segment at {start_time:.2f}s: {e}")
        
        start_sample += hop_samples
    
    return segments


def main(args):
    extractor = BirdNetExtractor(args.model, sr=args.sr, n_mels=args.n_mels,
                                 hop=args.hop, win=args.win, embedding_layer=args.embedding_layer)
    
    wavs = sorted(glob(os.path.join(args.wav_dir, '**', '*.wav'), recursive=True))
    os.makedirs(args.out_dir, exist_ok=True)
    
    all_metadata = []
    
    for wav_path in tqdm(wavs, desc="Processing audio files"):
        try:
            segments = extract_segments(extractor, wav_path, 
                                       segment_len=args.segment_len,
                                       hop_len=args.hop_len)
            
            # Maintain folder structure relative to input directory
            rel_path = os.path.relpath(wav_path, args.wav_dir)
            rel_dir = os.path.dirname(rel_path)
            base = os.path.splitext(os.path.basename(wav_path))[0]
            
            # Create output directory maintaining structure
            out_subdir = os.path.join(args.out_dir, rel_dir)
            os.makedirs(out_subdir, exist_ok=True)
            
            # Save each segment's embedding
            for idx, (emb, start_time, end_time) in enumerate(segments):
                segment_name = f"{base}_seg{idx:04d}"
                out_path = os.path.join(out_subdir, f'{segment_name}.npy')
                np.save(out_path, emb)
                
                all_metadata.append({
                    'wav_path': wav_path,
                    'feature_path': out_path,
                    'filename': base,
                    'segment_id': idx,
                    'start_time': start_time,
                    'end_time': end_time
                })
        
        except Exception as e:
            print(f'Error processing {wav_path}: {e}')
    
    # Save metadata
    joblib.dump(all_metadata, os.path.join(args.out_dir, 'segments_meta.pkl'))
    print(f'Saved {len(all_metadata)} segment embeddings to {args.out_dir}')
    print(f'From {len(wavs)} audio files')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Extract BirdNet embeddings from audio segments')
    p.add_argument('--wav-dir', required=True, help='Directory containing WAV files')
    p.add_argument('--model', required=True, help='Path to BirdNet model (TFLite or SavedModel)')
    p.add_argument('--out-dir', default='birdnet/features', help='Output directory')
    p.add_argument('--sr', type=int, default=48000, help='Sample rate')
    p.add_argument('--n-mels', type=int, default=128, help='Number of mel bands')
    p.add_argument('--hop', type=int, default=240, help='Hop length for mel spectrogram')
    p.add_argument('--win', type=int, default=1200, help='Window size for mel spectrogram')
    p.add_argument('--segment-len', type=float, default=3.0, help='Segment length in seconds')
    p.add_argument('--hop-len', type=float, default=1.5, help='Hop between segments in seconds')
    p.add_argument('--embedding-layer', default=None, help='Embedding layer name')
    
    args = p.parse_args()
    main(args)
