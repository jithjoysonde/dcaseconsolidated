"""
BirdNet Dataset for loading pre-computed BirdNet embeddings.
Based on PrototypeDynamicArrayDataSetWithEval but loads .npy files instead of computing PCEN.
"""
import sys
import os
from glob import glob
from itertools import chain
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


class BirdNetDataset(Dataset):
    def __init__(self, path: dict = {}, features: dict = {}, train_param: dict = {}):
        """
        Dataset that loads pre-computed BirdNet embeddings from _BDnet.npy files.
        
        Args:
            path: Dictionary with train_dir, eval_dir, test_dir
            features: Dictionary with feature configuration
            train_param: Dictionary with training parameters
        """
        print("Building BirdNet dataset with pre-computed embeddings")
        self.path = path
        self.features = features
        self.train_param = train_param
        self.samples_per_cls = train_param.n_shot * 2
        self.seg_len = train_param.seg_len
        
        # BirdNet uses 3s non-overlapping windows = 0.333 windows/second
        self.fps = 1.0  # frames per second (windows per second)
        
        print("Building BirdNet dataset for training")
        
        (
            self.all_train_csv_files,
            self.all_eval_csv_files,
            self.extra_csv_files,
        ) = self.get_all_csv_files()
        self.all_csv_files = (
            self.all_train_csv_files + self.all_eval_csv_files + self.extra_csv_files
        )
        self.train_classes = []
        self.eval_classes = []
        self.extra_train_classes = []
        
        self.length = int(3600 * 8 / self.train_param.seg_len)
        
        self.meta = {}
        self.birdnet_emb = {}  # Store BirdNet embeddings (time, 1280)
        
        # Buffers for consistent sampling within a batch
        self.segment_buffer = {}
        self.start_end_buffer = {}
        self.cnt = 0
        self.segment_level_training_length = 1.5
        self.batchsize = self.train_param.k_way * self.train_param.n_shot * 2
        
        self.build_meta()
        self.load_birdnet_embeddings()
        # Filter classes that have insufficient positive samples for few-shot learning
        delete_keys = []
        for k in self.meta.keys():
            # Check if class has enough positive samples (at least n_shot)
            if len(self.meta[k]["info"]) < self.train_param.n_shot:
                delete_keys.append(k)
                continue
            
            # If using negative contrast, check for negative samples
            if self.train_param.negative_train_contrast:
                neg_info = self.meta[k]["neg_info"]
                if not neg_info:  # No negative samples
                    delete_keys.append(k)
                    continue
                
                # Check if negative duration is sufficient (at least seg_len)
                total_neg_duration = sum(self.meta[k]["neg_duration"])
                if total_neg_duration < self.train_param.seg_len:
                    delete_keys.append(k)
        
        print(f"Deleting {len(delete_keys)} classes due to insufficient samples (need at least {self.train_param.n_shot} positive samples)")
        for k in delete_keys:
            del self.meta[k]
        
        # Update class lists to only include remaining classes
        self.train_classes = [c for c in self.train_classes if c in self.meta]
        self.eval_classes = [c for c in self.eval_classes if c in self.meta]
        self.extra_train_classes = [c for c in self.extra_train_classes if c in self.meta]
        
        self.classes = list(self.meta.keys())
        self.classes2int = self.get_class2int()
        print("Training classes:", self.classes2int)
        self.classes_duration = self.get_class_durations()
        
        # Update length to match number of available classes
        # self.length = len(self.classes)  # WRONG: This should be a large number for episodic sampling
        # Keep the original length calculation for episodic training
        self.length = int(3600 * 8 / self.train_param.seg_len)
        self.classes_duration = self.get_class_durations()
        
        self.train_eval_class_idxs = [
            self.classes2int[x]
            for x in self.train_classes + self.eval_classes
            if (x in self.classes)
        ]
        self.extra_train_class_idxs = [
            self.classes2int[x] for x in self.extra_train_classes if (x in self.classes)
        ]
        
        # Add eval_class_idxs attribute for compatibility with validation dataset interface
        self.eval_class_idxs = [
            self.classes2int[x] for x in self.eval_classes if (x in self.classes)
        ]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.cnt % self.batchsize == 0:
            self.segment_buffer = {}
            self.start_end_buffer = {}

        class_name = self.classes[idx]
        segment = self.select_positive(class_name)
        self.cnt += 1

        if not self.train_param.negative_train_contrast:
            return segment.astype(np.float32), self.classes2int[class_name], class_name
        else:
            segment_neg = self.select_negative(class_name)
            return (
                segment.astype(np.float32),
                segment_neg.astype(np.float32),
                self.classes2int[class_name] * 2,
                self.classes2int[class_name] * 2 + 1,
                class_name,
            )

    def load_birdnet_embeddings(self):
        """Load pre-computed BirdNet embeddings from _BDnet.npy files."""
        print("Loading pre-computed BirdNet embeddings...")
        for file in tqdm(self.all_csv_files):
            audio_path = file.replace("csv", "wav")
            emb_path = audio_path.replace(".wav", "_BDnet.npy")
            
            if not os.path.exists(emb_path):
                print(f"Warning: Missing BirdNet embedding for {audio_path}")
                print(f"Expected at: {emb_path}")
                continue
                
            try:
                # Load BirdNet embeddings (shape: (time_frames, 1280))
                emb = np.load(emb_path)
                self.birdnet_emb[audio_path] = emb
            except Exception as e:
                print(f"Error loading {emb_path}: {e}")

    def select_negative(self, class_name):
        """Select a negative segment for contrastive learning."""
        segment_idx = np.random.randint(len(self.meta[class_name]["neg_info"]))
        start, end = self.meta[class_name]["neg_info"][segment_idx]

        while end - start < 0.1:
            segment_idx = np.random.randint(len(self.meta[class_name]["neg_info"]))
            start, end = self.meta[class_name]["neg_info"][segment_idx]

        segment = self.select_segment(
            start,
            end,
            self.birdnet_emb[self.meta[class_name]["neg_file"][segment_idx]],
            seg_len=int(self.seg_len * self.fps),
            class_name=class_name,
        )
        return segment

    def select_positive(self, class_name):
        """Select a positive segment for the given class."""
        if len(self.meta[class_name]["info"]) == 0:
            # Fallback: return zeros if no positive samples
            print(f"Warning: No positive samples for class {class_name}, returning zeros")
            return np.zeros((int(self.seg_len * self.fps), self.birdnet_emb[list(self.birdnet_emb.keys())[0]].shape[1]), dtype=np.float32)
        
        if class_name in self.extra_train_classes:
            if class_name not in self.segment_buffer.keys():
                segment_idx = np.random.randint(len(self.meta[class_name]["info"]))
                self.segment_buffer[class_name] = segment_idx
                start, end = self.meta[class_name]["info"][segment_idx]
                if end - start > self.segment_level_training_length:
                    start = np.random.uniform(
                        0, end - self.segment_level_training_length
                    )
                    end = start + self.segment_level_training_length
                self.start_end_buffer[class_name] = (start, end)
            else:
                segment_idx = self.segment_buffer[class_name]
                start, end = self.start_end_buffer[class_name]
        else:
            segment_idx = np.random.randint(len(self.meta[class_name]["info"]))
            start, end = self.meta[class_name]["info"][segment_idx]

        segment = self.select_segment(
            start,
            end,
            self.birdnet_emb[self.meta[class_name]["file"][segment_idx]],
            seg_len=int(self.seg_len * self.fps),
            class_name=class_name,
        )
        return segment

    def select_segment(self, start, end, embeddings, seg_len=17, class_name=None):
        """
        Extract a segment from BirdNet embeddings.
        
        Args:
            start, end: Time in seconds
            embeddings: BirdNet embeddings array (time_frames, 6522)
            seg_len: Target segment length in frames (windows)
        
        Returns:
            segment: (seg_len, 6522) array
        """
        start, end = int(start * self.fps), int(end * self.fps)
        if start < 0:
            start = 0
        
        # Ensure end doesn't exceed embeddings length
        if end > len(embeddings):
            end = len(embeddings)
        
        total_duration = end - start
        if total_duration < seg_len:
            x = embeddings[start:end]
            if total_duration > 0:
                tile_times = int(np.ceil(seg_len / total_duration))
                x = np.tile(x, (tile_times, 1))
                x = x[:seg_len]
            else:
                # If segment is empty, create zeros
                x = np.zeros((seg_len, embeddings.shape[1]), dtype=embeddings.dtype)
        else:
            rand_start = int(np.random.uniform(low=start, high=end - seg_len))
            x = embeddings[rand_start:rand_start + seg_len]
        
        # Pad if necessary
        if x.shape[0] < seg_len:
            x = np.pad(x, ((0, seg_len - x.shape[0]), (0, 0)), mode='constant')
        
        assert x.shape[0] == seg_len, f"Shape mismatch: {x.shape} vs {seg_len}"
        return x

    def remove_short_negative_duration(self):
        """Remove classes with insufficient negative samples or no positive samples."""
        delete_keys = []
        for k in self.meta.keys():
            # Check if class has positive samples
            if not self.meta[k]["info"]:
                delete_keys.append(k)
                continue
            
            # Check if class has negative samples
            neg_info = self.meta[k]["neg_info"]
            if not neg_info:  # Skip if no negative samples
                delete_keys.append(k)
                continue
            
            # Check if negative duration is sufficient
            end_time = [x[1] for x in neg_info]
            start_time = [x[0] for x in neg_info]
            duration = max([x - y for x, y in zip(end_time, start_time)])
            if duration < 0.3:
                delete_keys.append(k)
        print(f"Deleting {len(delete_keys)} classes due to short negative length or no positive samples")
        for k in delete_keys:
            del self.meta[k]

    def get_class_durations(self):
        """Get total duration for each class."""
        durations = []
        for cls in self.classes:
            durations.append(np.sum(self.meta[cls]["duration"]))
        return durations

    def get_all_csv_files(self):
        """Get all CSV annotation files from train, eval, and test directories."""
        extension = "*.csv"
        
        train_files = [
            file
            for path_dir, _, _ in os.walk(self.path.train_dir)
            for file in glob(os.path.join(path_dir, extension))
        ] if self.path.train_dir else []
        
        eval_files = [
            file
            for path_dir, _, _ in os.walk(self.path.eval_dir)
            for file in glob(os.path.join(path_dir, extension))
        ] if self.path.eval_dir else []
        
        extra_files = []
        if hasattr(self.path, 'test_dir') and self.path.test_dir and self.path.test_dir != "null":
            extra_files = [
                file
                for path_dir, _, _ in os.walk(self.path.test_dir)
                for file in glob(os.path.join(path_dir, extension))
            ]
        
        return train_files, eval_files, extra_files

    def get_glob_cls_name(self, file):
        """Extract class name from file path."""
        split_list = file.split("/")
        return split_list[-2]

    def get_df_pos(self, file):
        """Get positive annotations from CSV file."""
        df = pd.read_csv(file, header=0, index_col=False)
        return df[(df == "POS").any(axis=1)]

    def get_cls_list(self, df_pos, glob_cls_name, start_time):
        """Get class list from annotations."""
        if "CALL" in df_pos.columns:
            cls_list = [glob_cls_name] * len(start_time)
        else:
            cls_list = [
                df_pos.columns[(df_pos == "POS").loc[index]].values
                for index, row in df_pos.iterrows()
            ]
            cls_list = list(chain.from_iterable(cls_list))
        return cls_list

    def get_time(self, df):
        """Get onset and offset times with 25ms margin."""
        start_time = df["Starttime"].values
        end_time = df["Endtime"].values
        return start_time - 0.025, end_time + 0.025

    def get_class2int(self):
        """Map class names to integers."""
        return {cls: i for i, cls in enumerate(self.classes)}

    def build_meta(self):
        """Build metadata from CSV annotation files."""
        for file in tqdm(self.all_csv_files):
            df_pos = self.get_df_pos(file)
            glob_cls_name = self.get_glob_cls_name(file)
            start_time, end_time = self.get_time(df_pos)
            cls_list = self.get_cls_list(df_pos, glob_cls_name, start_time)
            self.update_meta(start_time, end_time, cls_list, file)

    def update_meta(self, start_time, end_time, cls_list, csv_file):
        """Update metadata with annotation information."""
        audio_path = csv_file.replace("csv", "wav")
        glob_cls_name = self.get_glob_cls_name(csv_file)
        
        # Determine if this is a training or evaluation file
        if "Training_Set" in csv_file:
            # Add individual classes to train_classes
            for cls in set(cls_list):
                if cls not in self.train_classes:
                    self.train_classes.append(cls)
        elif "Validation_Set" in csv_file or "Evaluation_Set" in csv_file:
            # Add individual classes to eval_classes
            for cls in set(cls_list):
                if cls not in self.eval_classes:
                    self.eval_classes.append(cls)
        else:
            # Add individual classes to extra_train_classes
            for cls in set(cls_list):
                if cls not in self.extra_train_classes:
                    self.extra_train_classes.append(cls)
        
        end = 0
        for cls, start, stop in zip(cls_list, start_time, end_time):
            if cls not in self.meta:
                self.meta[cls] = {
                    "info": [],
                    "duration": [],
                    "file": [],
                    "neg_info": [],
                    "neg_file": [],
                    "neg_duration": [],
                    "neg_start_time": 0,
                }
            
            self.meta[cls]["info"].append((start, stop))
            self.meta[cls]["duration"].append(stop - start)
            self.meta[cls]["file"].append(audio_path)
            end = max(end, stop)
        
        # Calculate negative regions for each class found in this file
        for cls in set(cls_list):
            if audio_path in self.birdnet_emb:
                total_duration = len(self.birdnet_emb[audio_path]) / self.fps
            else:
                total_duration = end + 10  # Default estimate
            
            # Use the end time from positive segments as start of negative region
            neg_start = self.meta[cls].get("neg_start_time", end)
            neg_end = total_duration
            if neg_end > neg_start:  # Only add if there's actual negative duration
                self.meta[cls]["neg_info"].append((neg_start, neg_end))
                self.meta[cls]["neg_file"].append(audio_path)
                self.meta[cls]["neg_duration"].append(neg_end - neg_start)
            self.meta[cls]["neg_start_time"] = neg_end


class BirdNetTestSet(Dataset):
    """Test dataset for BirdNet embeddings, similar to PrototypeTestSet but loads pre-computed embeddings."""
    
    def __init__(
        self,
        path: dict = {},
        features: dict = {},
        train_param: dict = {},
        eval_param: dict = {},
    ):
        self.path = path
        self.features = features
        self.train_param = train_param
        self.eval_param = eval_param
        
        # BirdNet uses 3s non-overlapping windows = 0.333 windows/second
        self.fps = 1.0  # frames per second (windows per second)
        
        extension = "*.csv"
        # Only process eval_dir if it's valid
        if self.path.eval_dir and self.path.eval_dir != "null":
            self.all_csv_files = [
                file
                for path_dir, _, _ in os.walk(self.path.eval_dir)
                for file in glob(os.path.join(path_dir, extension))
            ]
        else:
            self.all_csv_files = []
        self.all_csv_files = sorted(self.all_csv_files)

    def __len__(self):
        return len(self.all_csv_files)

    def __getitem__(self, idx):
        hop_seg = int(round(self.eval_param.hop_seg * self.fps))
        seg_len = int(round(self.eval_param.seg_len * self.fps))
        feat_file = self.all_csv_files[idx]
        X_pos, X_neg, X_query, strt_index_query, audio_path = self.read_file(feat_file)
        
        # For compatibility with the model, we need to provide the extended format
        # that includes negative segments and additional metadata
        # For BirdNet, we use the same hop_seg for both positive and negative
        hop_seg_neg = hop_seg
        
        # max_len is the maximum length of segments across pos/neg/query
        max_len = max(X_pos.shape[1], X_neg.shape[1], X_query.shape[1])
        
        # neg_min_length is the minimum length of negative segments
        neg_min_length = X_neg.shape[1]
        
        return (
            (
                X_pos.astype(np.float32),
                X_neg.astype(np.float32),
                X_query.astype(np.float32),
                X_pos.astype(np.float32),  # X_pos_neg - use same as X_pos for BirdNet
                X_neg.astype(np.float32),  # X_neg_neg - use same as X_neg for BirdNet
                X_query.astype(np.float32),  # X_query_neg - use same as X_query for BirdNet
                hop_seg,
                hop_seg_neg,
                max_len,
                neg_min_length,
            ),
            strt_index_query,
            audio_path,
            seg_len,
        )

    def find_positive_label(self, df):
        for col in df.columns:
            if "Q" in col:
                return col
        else:
            raise ValueError(
                "Error: Expect you change the validation set event name to Q_x"
            )

    def read_file(self, file):
        seg_len = int(round(self.eval_param.seg_len * self.fps))
        hop_seg = int(
            round(self.eval_param.hop_seg * self.fps)
        )  # TODO hard hop segment length
        hop_neg = 0
        hop_query = 0
        strt_index = 0

        audio_path = file.replace("csv", "wav")
        emb_path = audio_path.replace(".wav", "_BDnet.npy")
        
        # Load BirdNet embeddings
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"BirdNet embedding not found: {emb_path}")
        
        pcen = np.load(emb_path)  # Shape: (time_frames, 6522)
        
        df_eval = pd.read_csv(file, header=0, index_col=False)
        key = self.find_positive_label(df_eval)
        Q_list = df_eval[key].to_numpy()
        
        # Convert time to frames (BirdNet windows)
        start_time = [max(0, int(np.floor((start - 0.025) * self.fps))) for start in df_eval["Starttime"]]
        end_time = [min(pcen.shape[0], int(np.floor((end + 0.025) * self.fps))) for end in df_eval["Endtime"]]
        
        index_sup = np.where(Q_list == "POS")[0][: self.train_param.n_shot]

        strt_indx_query = end_time[index_sup[-1]]
        end_idx_neg = pcen.shape[0] - 1

        feat_neg, feat_pos, feat_query = [], [], []

        while end_idx_neg - (strt_index + hop_neg) > seg_len:
            patch_neg = pcen[
                int(strt_index + hop_neg) : int(strt_index + hop_neg + seg_len)
            ]
            feat_neg.append(patch_neg)
            hop_neg += hop_seg

        last_patch = pcen[end_idx_neg - seg_len : end_idx_neg]
        feat_neg.append(last_patch)

        # print("Creating Positive dataset")
        for index in index_sup:
            str_ind = int(start_time[index])
            end_ind = int(end_time[index])

            # Ensure we have at least one frame
            if str_ind >= end_ind:
                # If start >= end after flooring, use the start frame
                str_ind = max(0, str_ind)
                end_ind = min(pcen.shape[0], str_ind + 1)

            if end_ind - str_ind > seg_len:
                shift = 0
                while end_ind - (str_ind + shift) > seg_len:
                    patch_pos = pcen[
                        int(str_ind + shift) : int(str_ind + shift + seg_len)
                    ]
                    feat_pos.append(patch_pos)
                    shift += hop_seg
                last_patch_pos = pcen[end_ind - seg_len : end_ind]
                feat_pos.append(last_patch_pos)

            else:
                patch_pos = pcen[str_ind:end_ind]
                if patch_pos.shape[0] == 0:
                    # If no frames, use the nearest frame
                    nearest_frame = min(max(str_ind, 0), pcen.shape[0] - 1)
                    patch_pos = pcen[nearest_frame:nearest_frame+1]
                    if patch_pos.shape[0] == 0:
                        print(f"No frames available for segment at {str_ind}-{end_ind}, skipping")
                        continue
                repeat_num = int(seg_len / (patch_pos.shape[0])) + 1
                patch_new = np.tile(patch_pos, (repeat_num, 1))
                patch_new = patch_new[0 : int(seg_len)]
                feat_pos.append(patch_new)

        # print("Creating query dataset")

        while end_idx_neg - (strt_indx_query + hop_query) > seg_len:
            patch_query = pcen[
                int(strt_indx_query + hop_query) : int(
                    strt_indx_query + hop_query + seg_len
                )
            ]
            feat_query.append(patch_query)
            hop_query += hop_seg

        last_patch_query = pcen[end_idx_neg - seg_len : end_idx_neg]
        feat_query.append(last_patch_query)
        return (
            np.stack(feat_pos),
            np.stack(feat_neg),
            np.stack(feat_query),
            strt_indx_query,
            audio_path,
        )  # [n, seg_len, 6522]
