#!/usr/bin/env python
"""
BirdNet Segment-Level Prediction Script
Performs detection on audio segments with actual time windows.
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_segment_features(features_dir, class_csv=None):
    """
    Load segment features with time information.
    
    Returns:
        X: Feature matrix
        metadata: List of dicts with filename, start_time, end_time, label
    """
    meta_path = os.path.join(features_dir, 'segments_meta.pkl')
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Segments metadata not found: {meta_path}")
    
    metadata = joblib.load(meta_path)
    
    # Load class labels if provided
    labels_dict = {}
    if class_csv and os.path.exists(class_csv):
        df = pd.read_csv(class_csv)
        for _, row in df.iterrows():
            if 'recording' in df.columns:
                key = row['recording']
                label = row.get('class_code', row.get('class_name', 'unknown'))
                labels_dict[key] = label
            elif len(df.columns) >= 2:
                # dataset, class_code format
                labels_dict[row.iloc[0]] = row.iloc[1]
    
    X = []
    enriched_metadata = []
    
    for meta in metadata:
        if not os.path.exists(meta['feature_path']):
            continue
        
        feat = np.load(meta['feature_path'])
        X.append(feat.flatten())
        
        # Determine label
        filename = meta['filename']
        dataset_name = os.path.basename(os.path.dirname(meta['wav_path']))
        label = labels_dict.get(filename, dataset_name)
        
        enriched_metadata.append({
            'filename': filename,
            'segment_id': meta['segment_id'],
            'start_time': meta['start_time'],
            'end_time': meta['end_time'],
            'label': label,
            'wav_path': meta['wav_path']
        })
    
    return np.vstack(X), enriched_metadata


def train_classifier(X_train, y_train, classifier='knn'):
    """Train a classifier."""
    if classifier == 'knn':
        model = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    else:
        model = LogisticRegression(max_iter=500, random_state=42)
    
    model.fit(X_train, y_train)
    return model


def predict_segments(model, X_test, test_metadata, confidence_threshold=0.5, output_csv='predictions.csv'):
    """
    Predict on test segments and output detections with time windows.
    
    Args:
        model: Trained classifier
        X_test: Test features
        test_metadata: List of metadata dicts
        confidence_threshold: Confidence threshold for predictions (if using proba)
        output_csv: Output CSV path
    """
    predictions = model.predict(X_test)
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X_test)
        max_probas = np.max(probas, axis=1)
    else:
        max_probas = np.ones(len(predictions))  # Default confidence
    
    # Create output dataframe
    results = []
    for pred, proba, meta in zip(predictions, max_probas, test_metadata):
        results.append({
            'Audiofilename': meta['filename'],
            'Starttime': round(meta['start_time'], 3),
            'Endtime': round(meta['end_time'], 3),
            'Q': pred,
            'Confidence': round(proba, 3)
        })
    
    df = pd.DataFrame(results)
    
    # Optional: Filter by confidence threshold
    if confidence_threshold > 0:
        print(f"Filtering predictions with confidence >= {confidence_threshold}")
        df_filtered = df[df['Confidence'] >= confidence_threshold]
        print(f"Kept {len(df_filtered)}/{len(df)} predictions")
    else:
        df_filtered = df
    
    # Sort by filename and start time
    df_filtered = df_filtered.sort_values(['Audiofilename', 'Starttime'])
    
    # Save
    df_filtered.to_csv(output_csv, index=False)
    print(f'Predictions saved to {output_csv}')
    
    return df_filtered


def merge_detections(df, max_gap=1.0):
    """
    Merge consecutive detections of the same class with small gaps.
    
    Args:
        df: DataFrame with columns [Audiofilename, Starttime, Endtime, Q]
        max_gap: Maximum gap in seconds to merge
    
    Returns:
        Merged DataFrame
    """
    merged = []
    
    for filename in df['Audiofilename'].unique():
        file_df = df[df['Audiofilename'] == filename].sort_values('Starttime')
        
        for class_label in file_df['Q'].unique():
            class_df = file_df[file_df['Q'] == class_label].sort_values('Starttime')
            
            if len(class_df) == 0:
                continue
            
            current_start = class_df.iloc[0]['Starttime']
            current_end = class_df.iloc[0]['Endtime']
            
            for i in range(1, len(class_df)):
                row = class_df.iloc[i]
                gap = row['Starttime'] - current_end
                
                if gap <= max_gap:
                    # Merge with current detection
                    current_end = row['Endtime']
                else:
                    # Save current detection and start new one
                    merged.append({
                        'Audiofilename': filename,
                        'Starttime': round(current_start, 3),
                        'Endtime': round(current_end, 3),
                        'Q': class_label
                    })
                    current_start = row['Starttime']
                    current_end = row['Endtime']
            
            # Save last detection
            merged.append({
                'Audiofilename': filename,
                'Starttime': round(current_start, 3),
                'Endtime': round(current_end, 3),
                'Q': class_label
            })
    
    return pd.DataFrame(merged)


def main(args):
    print("=" * 80)
    print("BirdNet Segment-Level Detection")
    print("=" * 80)
    
    # Load training data
    print(f"\nLoading training segments from {args.train_dir}...")
    X_train, train_metadata = load_segment_features(args.train_dir, args.train_csv)
    y_train = np.array([m['label'] for m in train_metadata])
    
    print(f"Loaded {len(X_train)} training segments")
    print(f"Classes: {np.unique(y_train)}")
    print(f"Segments per class: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    # Train classifier
    print(f"\nTraining {args.classifier} classifier...")
    model = train_classifier(X_train, y_train, classifier=args.classifier)
    
    # Evaluate on validation if provided
    if args.val_dir and os.path.exists(args.val_dir):
        print(f"\nEvaluating on validation set...")
        try:
            X_val, val_metadata = load_segment_features(args.val_dir, args.val_csv)
            y_val = np.array([m['label'] for m in val_metadata])
            
            val_preds = model.predict(X_val)
            val_acc = accuracy_score(y_val, val_preds)
            print(f"Validation Accuracy: {val_acc:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_val, val_preds))
        except Exception as e:
            print(f"Validation evaluation failed: {e}")
    
    # Predict on test set
    if args.test_dir and os.path.exists(args.test_dir):
        print("\n" + "=" * 80)
        print("Predicting on test set")
        print("=" * 80)
        
        X_test, test_metadata = load_segment_features(args.test_dir)
        print(f"Loaded {len(X_test)} test segments")
        print(f"From {len(set(m['filename'] for m in test_metadata))} audio files")
        
        # Predict
        df_pred = predict_segments(model, X_test, test_metadata,
                                   confidence_threshold=args.confidence_threshold,
                                   output_csv=args.output_csv)
        
        # Optionally merge detections
        if args.merge_detections:
            print(f"\nMerging consecutive detections (max gap: {args.merge_gap}s)...")
            df_merged = merge_detections(df_pred, max_gap=args.merge_gap)
            merged_csv = args.output_csv.replace('.csv', '_merged.csv')
            df_merged.to_csv(merged_csv, index=False)
            print(f"Merged predictions saved to {merged_csv}")
            print(f"Reduced from {len(df_pred)} to {len(df_merged)} detections")
        
        # Print summary
        print("\n" + "=" * 80)
        print("Detection Summary")
        print("=" * 80)
        print(f"\nTotal detections: {len(df_pred)}")
        print(f"\nDetections per class:")
        for cls, cnt in df_pred['Q'].value_counts().items():
            print(f"  {cls}: {cnt}")
        print(f"\nDetections per file:")
        for fname, cnt in df_pred['Audiofilename'].value_counts().head(10).items():
            print(f"  {fname}: {cnt}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BirdNet Segment-Level Detection')
    
    # Data directories
    parser.add_argument('--train-dir', default='birdnet/features/train',
                        help='Training segment features directory')
    parser.add_argument('--val-dir', default='birdnet/features/val',
                        help='Validation segment features directory')
    parser.add_argument('--test-dir', default='birdnet/features/test',
                        help='Test segment features directory')
    
    # Class CSV files
    parser.add_argument('--train-csv', default='/data/msc-proj/Training_Set_classes.csv',
                        help='Training class labels CSV')
    parser.add_argument('--val-csv', default='/data/msc-proj/Validation_Set_classes_DSAI_2025_2026.csv',
                        help='Validation class labels CSV')
    
    # Model settings
    parser.add_argument('--classifier', default='knn', choices=['knn', 'logreg'],
                        help='Classifier type')
    parser.add_argument('--confidence-threshold', type=float, default=0.0,
                        help='Confidence threshold for predictions (0 = keep all)')
    
    # Post-processing
    parser.add_argument('--merge-detections', action='store_true',
                        help='Merge consecutive detections of the same class')
    parser.add_argument('--merge-gap', type=float, default=1.0,
                        help='Maximum gap in seconds to merge detections')
    
    # Output
    parser.add_argument('--output-csv', default='birdnet/predictions_segments.csv',
                        help='Output predictions CSV')
    
    args = parser.parse_args()
    main(args)
