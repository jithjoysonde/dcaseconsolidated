#!/usr/bin/env python
"""
BirdNet Few-Shot Prediction Script
This script uses extracted BirdNet embeddings to perform few-shot classification.
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def load_features_with_labels(features_dir, class_csv=None, dataset_filter=None):
    """
    Load features from directory and match them with labels from CSV.
    
    Args:
        features_dir: Directory containing .npy feature files and meta.pkl
        class_csv: Path to CSV with columns [dataset, recording/class_code, class_code, class_name]
        dataset_filter: Optional list of dataset names to filter by (e.g., ['BV', 'HT'])
    
    Returns:
        X: Feature matrix (n_samples, feature_dim)
        y: Labels array
        filenames: List of filenames
    """
    meta_path = os.path.join(features_dir, 'meta.pkl')
    
    if os.path.exists(meta_path):
        meta = joblib.load(meta_path)
    else:
        # Fallback: glob for .npy files
        files = sorted([p for p in os.listdir(features_dir) if p.endswith('.npy')])
        meta = [(f.replace('.npy', '.wav'), os.path.join(features_dir, f)) for f in files]
    
    # Load class labels if provided
    labels_dict = {}
    if class_csv and os.path.exists(class_csv):
        df = pd.read_csv(class_csv)
        for _, row in df.iterrows():
            # Try to match by recording name
            if 'recording' in df.columns:
                labels_dict[row['recording']] = row['class_code'] if 'class_code' in df.columns else row.get('class_name', 'unknown')
            elif len(df.columns) >= 3:
                # Assume format: dataset, class_code, class_name
                # Use dataset as a prefix
                labels_dict[row.iloc[0]] = row.iloc[1]  # dataset -> class_code
    
    X, y, filenames = [], [], []
    for wav_path, feat_path in meta:
        if not os.path.exists(feat_path):
            continue
            
        feat = np.load(feat_path)
        filename = os.path.basename(wav_path).replace('.wav', '')
        
        # Extract dataset folder name
        dataset_name = os.path.basename(os.path.dirname(wav_path))
        
        # Apply dataset filter if specified
        if dataset_filter and dataset_name not in dataset_filter:
            continue
        
        # Determine label
        if class_csv:
            # Try matching by filename
            label = labels_dict.get(filename, dataset_name)
            # If not found, use dataset name as label
            if label == filename and filename not in labels_dict:
                label = dataset_name
        else:
            # Use parent directory as label
            label = dataset_name
        
        X.append(feat.flatten())
        y.append(label)
        filenames.append(filename)
    
    if len(X) == 0:
        raise RuntimeError(f'No features found in {features_dir}')
    
    return np.vstack(X), np.array(y), filenames


def sample_fewshot(X, y, n_shots=5, n_queries=10, rng=0):
    """
    Sample few-shot support and query sets.
    
    Args:
        X: Feature matrix
        y: Labels
        n_shots: Number of examples per class for support set
        n_queries: Number of examples per class for query set
        rng: Random seed
    
    Returns:
        support_X, support_y, query_X, query_y
    """
    np.random.seed(rng)
    labels = np.unique(y)
    support_X, support_y, query_X, query_y = [], [], [], []
    
    for lab in labels:
        ids = np.where(y == lab)[0]
        if len(ids) < n_shots + n_queries:
            print(f'Warning: Class {lab} has only {len(ids)} samples, need {n_shots + n_queries}')
            continue
        
        perm = np.random.permutation(ids)
        support_X.append(X[perm[:n_shots]])
        support_y += [lab] * n_shots
        query_X.append(X[perm[n_shots:n_shots + n_queries]])
        query_y += [lab] * n_queries
    
    if len(support_y) == 0:
        raise RuntimeError('Not enough data per class for shots/queries')
    
    return np.vstack(support_X), np.array(support_y), np.vstack(query_X), np.array(query_y)


def train_and_evaluate(support_X, support_y, query_X, query_y, classifier='knn'):
    """
    Train classifier on support set and evaluate on query set.
    
    Args:
        support_X: Support features
        support_y: Support labels
        query_X: Query features
        query_y: Query labels
        classifier: 'knn' or 'logreg'
    
    Returns:
        predictions, accuracy, model
    """
    if classifier == 'knn':
        model = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    else:
        model = LogisticRegression(max_iter=500, random_state=42)
    
    model.fit(support_X, support_y)
    predictions = model.predict(query_X)
    accuracy = accuracy_score(query_y, predictions)
    
    return predictions, accuracy, model


def predict_unlabeled(model, test_X, test_filenames, wav_dir=None, output_csv='predictions.csv'):
    """
    Make predictions on unlabeled test data and save to CSV with timestamps.
    
    Args:
        model: Trained classifier
        test_X: Test features
        test_filenames: Test filenames
        wav_dir: Directory containing wav files to get durations
        output_csv: Output CSV path
    
    Returns:
        predictions array
    """
    import soundfile as sf
    from glob import glob
    
    predictions = model.predict(test_X)
    
    # Prepare data for CSV
    rows = []
    for filename, pred_class in zip(test_filenames, predictions):
        # Try to get actual audio duration
        duration = None
        if wav_dir:
            # Search for the wav file
            wav_matches = glob(os.path.join(wav_dir, '**', f'{filename}.wav'), recursive=True)
            if wav_matches:
                try:
                    y, sr = sf.read(wav_matches[0])
                    duration = len(y) / sr
                except:
                    pass
        
        # Use actual duration or default to 5.0 seconds for whole file predictions
        end_time = duration if duration is not None else 5.0
        
        rows.append({
            'Audiofilename': filename,
            'Starttime': 0.0,
            'Endtime': end_time,
            'Q': pred_class  # Q is standard DCASE notation for query/predicted class
        })
    
    # Save predictions in DCASE format
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f'Predictions saved to {output_csv}')
    
    return predictions


def main(args):
    print("=" * 80)
    print("BirdNet Few-Shot Prediction")
    print("=" * 80)
    
    # Load training features
    print(f"\nLoading training features from {args.train_dir}...")
    train_X, train_y, train_files = load_features_with_labels(
        args.train_dir, args.train_csv
    )
    print(f"Loaded {len(train_X)} training samples from {len(np.unique(train_y))} classes")
    print(f"Classes: {np.unique(train_y)}")
    
    # Load validation features if provided
    if args.val_dir and os.path.exists(args.val_dir):
        print(f"\nLoading validation features from {args.val_dir}...")
        val_X, val_y, val_files = load_features_with_labels(
            args.val_dir, args.val_csv
        )
        print(f"Loaded {len(val_X)} validation samples from {len(np.unique(val_y))} classes")
        
        # Evaluate on validation set
        print("\n" + "=" * 80)
        print("Training and evaluating on validation set")
        print("=" * 80)
        
        for clf_name in ['knn', 'logreg']:
            print(f"\n{clf_name.upper()} Classifier:")
            predictions, accuracy, model = train_and_evaluate(
                train_X, train_y, val_X, val_y, classifier=clf_name
            )
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(val_y, predictions))
    
    # Few-shot evaluation if requested
    if args.few_shot:
        print("\n" + "=" * 80)
        print(f"Few-Shot Evaluation ({args.n_shots}-shot)")
        print("=" * 80)
        
        try:
            support_X, support_y, query_X, query_y = sample_fewshot(
                train_X, train_y, 
                n_shots=args.n_shots, 
                n_queries=args.n_queries, 
                rng=args.seed
            )
            
            for clf_name in ['knn', 'logreg']:
                print(f"\n{clf_name.upper()} {args.n_shots}-shot:")
                predictions, accuracy, _ = train_and_evaluate(
                    support_X, support_y, query_X, query_y, classifier=clf_name
                )
                print(f"Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Few-shot evaluation failed: {e}")
    
    # Predict on test set if provided
    if args.test_dir and os.path.exists(args.test_dir):
        print("\n" + "=" * 80)
        print("Predicting on test set")
        print("=" * 80)
        
        test_X, test_y, test_files = load_features_with_labels(args.test_dir)
        print(f"Loaded {len(test_X)} test samples")
        
        # Train on all training data
        model = KNeighborsClassifier(n_neighbors=1, metric='cosine')
        model.fit(train_X, train_y)
        
        # Make predictions
        predictions = predict_unlabeled(
            model, test_X, test_files,
            wav_dir=args.test_wav_dir,
            output_csv=args.output_csv
        )
        
        print(f"\nPredicted class distribution:")
        unique, counts = np.unique(predictions, return_counts=True)
        for cls, cnt in zip(unique, counts):
            print(f"  {cls}: {cnt}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BirdNet Few-Shot Prediction')
    
    # Data directories
    parser.add_argument('--train-dir', default='birdnet/features/train',
                        help='Training features directory')
    parser.add_argument('--val-dir', default='birdnet/features/val',
                        help='Validation features directory')
    parser.add_argument('--test-dir', default='birdnet/features/test',
                        help='Test features directory')
    parser.add_argument('--test-wav-dir', default='/data/msc-proj/Evaluation_Set_DSAI_2025_2026',
                        help='Test wav files directory for getting durations')
    
    # Class CSV files
    parser.add_argument('--train-csv', default='/data/msc-proj/Training_Set_classes.csv',
                        help='Training class labels CSV')
    parser.add_argument('--val-csv', default='/data/msc-proj/Validation_Set_classes_DSAI_2025_2026.csv',
                        help='Validation class labels CSV')
    
    # Few-shot settings
    parser.add_argument('--few-shot', action='store_true',
                        help='Run few-shot evaluation on training set')
    parser.add_argument('--n-shots', type=int, default=5,
                        help='Number of shots per class')
    parser.add_argument('--n-queries', type=int, default=10,
                        help='Number of queries per class')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output
    parser.add_argument('--output-csv', default='birdnet/predictions.csv',
                        help='Output predictions CSV')
    
    args = parser.parse_args()
    main(args)
