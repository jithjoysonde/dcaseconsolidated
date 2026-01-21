"""
BirdNet Test Dataset for loading pre-computed BirdNet embeddings for testing.
"""
import os
from glob import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class BirdNetTestSet(Dataset):
    def __init__(self, path: dict = {}, features: dict = {}, train_param: dict = {}, eval_param: dict = {}):
        """
        Test dataset that loads pre-computed BirdNet embeddings from _BDnet.npy files for testing.

        Args:
            path: Dictionary with test_dir
            features: Dictionary with feature configuration
            train_param: Dictionary with training parameters
            eval_param: Dictionary with evaluation parameters
        """
        self.path = path
        self.features = features
        self.train_param = train_param
        self.eval_param = eval_param

        # BirdNet uses 3s non-overlapping windows = 0.333 windows/second
        self.fps = 1.0  # frames per second (windows per second)
        self.seg_len_frames = int(self.train_param.seg_len * self.fps)

        # Get all CSV files in test directory
        extension = "*.csv"
        if self.path.get('test_dir') and self.path.test_dir != "null":
            self.all_csv_files = [
                file
                for path_dir, _, _ in os.walk(self.path.test_dir)
                for file in glob(os.path.join(path_dir, extension))
            ]
        else:
            self.all_csv_files = []
        self.all_csv_files = sorted(self.all_csv_files)

        print(f"Found {len(self.all_csv_files)} test CSV files")

    def __len__(self):
        return len(self.all_csv_files)

    def __getitem__(self, idx):
        feat_file = self.all_csv_files[idx]
        X_pos, X_neg, X_query, strt_index_query, audio_path = self.read_file(feat_file)

        # Return in the same format as PrototypeTestSet
        hop_seg = int(round(self.eval_param.hop_seg * self.fps))
        return (
            (
                X_pos.astype(np.float32),
                X_neg.astype(np.float32),
                X_query.astype(np.float32),
                hop_seg,
            ),
            strt_index_query,
            audio_path,
        )

    def find_positive_label(self, df):
        """Find the positive class column (contains 'Q' for query)"""
        for col in df.columns:
            if "Q" in col:
                return col
        else:
            raise ValueError("Error: Expect validation set event name to contain 'Q'")

    def read_file(self, file):
        """Read CSV file and extract BirdNet embeddings for positive, negative, and query samples"""
        df_eval = pd.read_csv(file, header=0, index_col=False)
        key = self.find_positive_label(df_eval)

        # Load BirdNet embeddings
        audio_path = file.replace("csv", "wav")
        embedding_file = file.replace(".csv", "_BDnet.npy")

        if not os.path.exists(embedding_file):
            raise FileNotFoundError(f"BirdNet embedding file not found: {embedding_file}")

        embeddings = np.load(embedding_file)  # Shape: (time, 6522)

        # Get positive segments (Q column)
        pos_segments = df_eval[df_eval[key] == "POS"]
        if len(pos_segments) == 0:
            # If no POS segments, use all segments as query
            pos_segments = df_eval

        # Get negative segments (everything not POS)
        neg_segments = df_eval[df_eval[key] != "POS"]

        # Extract embeddings for positive samples (support set)
        X_pos = self.extract_embeddings_for_segments(embeddings, pos_segments, self.train_param.n_shot)

        # Extract embeddings for negative samples
        X_neg = self.extract_embeddings_for_segments(embeddings, neg_segments, self.eval_param.samples_neg)

        # Extract embeddings for query samples (remaining segments)
        query_segments = df_eval  # Use all segments as potential queries
        X_query, strt_index_query = self.extract_query_embeddings(embeddings, query_segments)

        return X_pos, X_neg, X_query, strt_index_query, audio_path

    def extract_embeddings_for_segments(self, embeddings, segments, max_samples):
        """Extract embeddings for given segments"""
        if len(segments) == 0:
            # Return empty array with correct shape
            return np.zeros((max_samples, self.seg_len_frames, embeddings.shape[1]), dtype=np.float32)

        extracted = []
        for _, segment in segments.iterrows():
            start_time = max(0, segment['Starttime'] - 0.025)  # 25ms margin
            end_time = segment['Endtime'] + 0.025

            start_frame = int(np.floor(start_time * self.fps))
            end_frame = int(np.floor(end_time * self.fps))

            # Ensure we have at least seg_len_frames
            if end_frame - start_frame < self.seg_len_frames:
                end_frame = min(embeddings.shape[0], start_frame + self.seg_len_frames)

            if start_frame >= embeddings.shape[0] or end_frame <= start_frame:
                continue

            segment_emb = embeddings[start_frame:end_frame]
            if len(segment_emb) >= self.seg_len_frames:
                # Take first seg_len_frames
                segment_emb = segment_emb[:self.seg_len_frames]
            elif len(segment_emb) > 0:
                # Pad if necessary
                padding = np.zeros((self.seg_len_frames - len(segment_emb), embeddings.shape[1]))
                segment_emb = np.vstack([segment_emb, padding])
            else:
                continue

            extracted.append(segment_emb)

            if len(extracted) >= max_samples:
                break

        if len(extracted) == 0:
            # Return zeros if no valid segments
            return np.zeros((max_samples, self.seg_len_frames, embeddings.shape[1]), dtype=np.float32)

        # Pad or truncate to max_samples
        if len(extracted) < max_samples:
            padding_shape = (max_samples - len(extracted), self.seg_len_frames, embeddings.shape[1])
            padding = np.zeros(padding_shape, dtype=np.float32)
            extracted.extend([padding[i] for i in range(max_samples - len(extracted))])

        return np.array(extracted[:max_samples])

    def extract_query_embeddings(self, embeddings, segments):
        """Extract embeddings for query samples"""
        query_embeddings = []
        strt_indices = []

        for idx, segment in segments.iterrows():
            start_time = max(0, segment['Starttime'] - 0.025)
            end_time = segment['Endtime'] + 0.025

            start_frame = int(np.floor(start_time * self.fps))
            end_frame = int(np.floor(end_time * self.fps))

            if end_frame - start_frame < self.seg_len_frames:
                end_frame = min(embeddings.shape[0], start_frame + self.seg_len_frames)

            if start_frame >= embeddings.shape[0] or end_frame <= start_frame:
                continue

            segment_emb = embeddings[start_frame:end_frame]
            if len(segment_emb) >= self.seg_len_frames:
                segment_emb = segment_emb[:self.seg_len_frames]
            elif len(segment_emb) > 0:
                padding = np.zeros((self.seg_len_frames - len(segment_emb), embeddings.shape[1]))
                segment_emb = np.vstack([segment_emb, padding])
            else:
                continue

            query_embeddings.append(segment_emb)
            strt_indices.append(start_frame)

        if len(query_embeddings) == 0:
            # Return at least one zero embedding
            query_embeddings = [np.zeros((self.seg_len_frames, embeddings.shape[1]), dtype=np.float32)]
            strt_indices = [0]

        return np.array(query_embeddings), strt_indices