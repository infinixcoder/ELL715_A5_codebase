"""
Training Script for Viola-Jones Face Detector

This script trains the complete cascade of classifiers using the generated dataset.
"""

import numpy as np
import os
from haar_features import HaarFeatureGenerator
from cascade_classifier import train_cascade
from utils import save_model, print_metrics
import time


def main():
    """Main training function."""
    
    print("="*70)
    print(" "*20 + "VIOLA-JONES FACE DETECTOR")
    print(" "*25 + "TRAINING SCRIPT")
    print("="*70)
    
    # Configuration
    WINDOW_SIZE = 16
    MIN_FEATURE_SIZE = 2
    MAX_FEATURE_SIZE = 16
    
    NUM_STAGES = 5
    FEATURES_PER_STAGE = [10, 20, 30, 40, 50]  # Progressive complexity
    TARGET_DETECTION_RATE = 0.995  # 99.5% detection rate per stage
    TARGET_FP_RATE = 0.5  # 50% false positive rate per stage
    
    VALIDATION_SPLIT = 0.2  # Use 20% of training data for validation
    
    # Load training data
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    if not os.path.exists("data/train_faces.npy"):
        print("Error: Training data not found!")
        print("Please run 'python dataset_generator.py' first to generate the dataset.")
        return
    
    train_faces = np.load("data/train_faces.npy")
    train_labels = np.load("data/train_labels.npy")
    
    print(f"Training data loaded:")
    print(f"  Total samples: {len(train_faces)}")
    print(f"  Face samples: {np.sum(train_labels == 1)}")
    print(f"  Non-face samples: {np.sum(train_labels == 0)}")
    print(f"  Image size: {train_faces.shape[1]}x{train_faces.shape[2]}")
    
    # Split into train and validation
    n_samples = len(train_faces)
    n_validation = int(n_samples * VALIDATION_SPLIT)
    
    # Shuffle data
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    
    val_indices = indices[:n_validation]
    train_indices = indices[n_validation:]
    
    validation_faces = train_faces[val_indices]
    validation_labels = train_labels[val_indices]
    
    train_faces = train_faces[train_indices]
    train_labels = train_labels[train_indices]
    
    print(f"\nData split:")
    print(f"  Training: {len(train_faces)} samples")
    print(f"  Validation: {len(validation_faces)} samples")
    
    # Generate Haar features
    print("\n" + "="*70)
    print("GENERATING HAAR FEATURES")
    print("="*70)
    
    feature_generator = HaarFeatureGenerator(window_size=WINDOW_SIZE)
    print(f"Generating features with:")
    print(f"  Window size: {WINDOW_SIZE}x{WINDOW_SIZE}")
    print(f"  Min feature size: {MIN_FEATURE_SIZE}")
    print(f"  Max feature size: {MAX_FEATURE_SIZE}")
    
    features = feature_generator.generate_all_features(
        min_feature_size=MIN_FEATURE_SIZE,
        max_feature_size=MAX_FEATURE_SIZE
    )
    
    print(f"\nGenerated {len(features)} Haar features")
    
    # Extract feature values for all training samples
    print("\nExtracting feature values for training samples...")
    start_time = time.time()
    train_feature_matrix = feature_generator.compute_features_for_images(train_faces)
    train_time = time.time() - start_time
    
    print(f"Training feature extraction completed in {train_time:.2f} seconds")
    print(f"Feature matrix shape: {train_feature_matrix.shape}")
    
    # Extract feature values for validation samples
    print("\nExtracting feature values for validation samples...")
    start_time = time.time()
    val_feature_matrix = feature_generator.compute_features_for_images(validation_faces)
    val_time = time.time() - start_time
    
    print(f"Validation feature extraction completed in {val_time:.2f} seconds")
    print(f"Feature matrix shape: {val_feature_matrix.shape}")
    
    # Save feature data
    os.makedirs("data", exist_ok=True)
    np.save("data/train_feature_matrix.npy", train_feature_matrix)
    np.save("data/val_feature_matrix.npy", val_feature_matrix)
    print("\nFeature matrices saved to data/ directory")
    
    # Train cascade
    print("\n" + "="*70)
    print("TRAINING CASCADE OF CLASSIFIERS")
    print("="*70)
    
    start_time = time.time()
    
    cascade = train_cascade(
        train_feature_matrix,
        train_labels,
        val_feature_matrix,
        validation_labels,
        num_stages=NUM_STAGES,
        features_per_stage=FEATURES_PER_STAGE,
        target_detection_rate=TARGET_DETECTION_RATE,
        target_fp_rate=TARGET_FP_RATE,
        verbose=True
    )
    
    training_time = time.time() - start_time
    
    print(f"\nTotal training time: {training_time/60:.2f} minutes")
    
    # Evaluate on validation set
    print("\n" + "="*70)
    print("VALIDATION SET EVALUATION")
    print("="*70)
    
    val_results = cascade.evaluate(val_feature_matrix, validation_labels)
    
    print_metrics(val_results, "Overall Validation Metrics")
    
    print("\nPer-stage metrics:")
    for stage_metric in val_results['stage_metrics']:
        print(f"  Stage {stage_metric['stage'] + 1}:")
        print(f"    Samples reaching: {stage_metric['samples_reaching']}")
        print(f"    Detection rate: {stage_metric['detection_rate']:.4f}")
        print(f"    False positive rate: {stage_metric['false_positive_rate']:.4f}")
    
    # Save model
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    
    os.makedirs("models", exist_ok=True)
    
    model_data = {
        'cascade': cascade,
        'feature_generator': feature_generator,
        'window_size': WINDOW_SIZE,
        'num_stages': NUM_STAGES,
        'features_per_stage': FEATURES_PER_STAGE,
        'validation_metrics': val_results
    }
    
    save_model(model_data, "models/viola_jones_cascade.pkl")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: models/viola_jones_cascade.pkl")
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Validation accuracy: {val_results['accuracy']:.4f}")
    print(f"Validation recall: {val_results['recall']:.4f}")
    print("\nNext steps:")
    print("  1. Run 'python test.py' to evaluate on test set")
    print("  2. Run 'python detect_faces.py --image_path <image>' to detect faces")


if __name__ == "__main__":
    main()
