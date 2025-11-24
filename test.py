"""
Testing Script for Viola-Jones Face Detector

This script evaluates the trained cascade classifier on the test dataset.
"""

import numpy as np
import os
from utils import load_model, print_metrics, compute_metrics
import time


def main():
    """Main testing function."""
    
    print("="*70)
    print(" "*20 + "VIOLA-JONES FACE DETECTOR")
    print(" "*25 + "TESTING SCRIPT")
    print("="*70)
    
    # Load test data
    print("\n" + "="*70)
    print("LOADING TEST DATASET")
    print("="*70)
    
    if not os.path.exists("data/test_faces.npy"):
        print("Error: Test data not found!")
        print("Please run 'python dataset_generator.py' first to generate the dataset.")
        return
    
    test_faces = np.load("data/test_faces.npy")
    test_labels = np.load("data/test_labels.npy")
    
    print(f"Test data loaded:")
    print(f"  Total samples: {len(test_faces)}")
    print(f"  Face samples: {np.sum(test_labels == 1)}")
    print(f"  Non-face samples: {np.sum(test_labels == 0)}")
    print(f"  Image size: {test_faces.shape[1]}x{test_faces.shape[2]}")
    
    # Load trained model
    print("\n" + "="*70)
    print("LOADING TRAINED MODEL")
    print("="*70)
    
    if not os.path.exists("models/viola_jones_cascade.pkl"):
        print("Error: Trained model not found!")
        print("Please run 'python train.py' first to train the model.")
        return
    
    model_data = load_model("models/viola_jones_cascade.pkl")
    
    cascade = model_data['cascade']
    feature_generator = model_data['feature_generator']
    
    print(f"\nModel information:")
    print(f"  Number of stages: {len(cascade.stages)}")
    print(f"  Window size: {model_data['window_size']}")
    print(f"  Total features: {len(feature_generator.features)}")
    
    # Extract features from test data
    print("\n" + "="*70)
    print("EXTRACTING FEATURES FROM TEST DATA")
    print("="*70)
    
    print("Computing Haar features for test samples...")
    start_time = time.time()
    test_feature_matrix = feature_generator.compute_features_for_images(test_faces)
    extraction_time = time.time() - start_time
    
    print(f"Feature extraction completed in {extraction_time:.2f} seconds")
    print(f"Feature matrix shape: {test_feature_matrix.shape}")
    
    # Evaluate cascade
    print("\n" + "="*70)
    print("EVALUATING CASCADE CLASSIFIER")
    print("="*70)
    
    start_time = time.time()
    test_results = cascade.evaluate(test_feature_matrix, test_labels)
    eval_time = time.time() - start_time
    
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    
    # Print overall results
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    
    print_metrics(test_results, "Overall Test Metrics")
    
    # Per-stage analysis
    print("\n" + "-"*70)
    print("PER-STAGE ANALYSIS")
    print("-"*70)
    
    for stage_metric in test_results['stage_metrics']:
        print(f"\nStage {stage_metric['stage'] + 1}:")
        print(f"  Samples reaching this stage: {stage_metric['samples_reaching']}")
        print(f"  Detection rate: {stage_metric['detection_rate']:.4f} ({stage_metric['detection_rate']*100:.2f}%)")
        print(f"  False positive rate: {stage_metric['false_positive_rate']:.4f} ({stage_metric['false_positive_rate']*100:.2f}%)")
    
    # Additional statistics
    print("\n" + "-"*70)
    print("DETECTION STATISTICS")
    print("-"*70)
    
    predictions, rejection_stages = cascade.predict_with_stage_info(test_feature_matrix)
    
    # Analyze rejection stages
    rejected_indices = np.where(predictions == 0)[0]
    if len(rejected_indices) > 0:
        rejection_stage_counts = np.bincount(rejection_stages[rejected_indices] + 1, 
                                            minlength=len(cascade.stages) + 1)
        
        print("\nRejection distribution:")
        for stage_idx in range(len(cascade.stages)):
            count = rejection_stage_counts[stage_idx]
            percentage = 100 * count / len(rejected_indices)
            print(f"  Stage {stage_idx + 1}: {count} samples ({percentage:.2f}%)")
    
    # Face vs non-face breakdown
    print("\n" + "-"*70)
    print("CLASS-WISE PERFORMANCE")
    print("-"*70)
    
    # Face samples (positive class)
    face_indices = np.where(test_labels == 1)[0]
    face_predictions = predictions[face_indices]
    face_accuracy = np.mean(face_predictions == 1)
    
    print(f"\nFace samples (positive class):")
    print(f"  Total: {len(face_indices)}")
    print(f"  Correctly detected: {np.sum(face_predictions == 1)}")
    print(f"  Missed: {np.sum(face_predictions == 0)}")
    print(f"  Detection rate: {face_accuracy:.4f} ({face_accuracy*100:.2f}%)")
    
    # Non-face samples (negative class)
    non_face_indices = np.where(test_labels == 0)[0]
    non_face_predictions = predictions[non_face_indices]
    non_face_accuracy = np.mean(non_face_predictions == 0)
    
    print(f"\nNon-face samples (negative class):")
    print(f"  Total: {len(non_face_indices)}")
    print(f"  Correctly rejected: {np.sum(non_face_predictions == 0)}")
    print(f"  False positives: {np.sum(non_face_predictions == 1)}")
    print(f"  Rejection rate: {non_face_accuracy:.4f} ({non_face_accuracy*100:.2f}%)")
    
    # Save test results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    results_summary = {
        'test_metrics': test_results,
        'predictions': predictions,
        'rejection_stages': rejection_stages,
        'evaluation_time': eval_time,
        'feature_extraction_time': extraction_time
    }
    
    np.save("data/test_results.npy", results_summary)
    print("Test results saved to data/test_results.npy")
    
    # Summary
    print("\n" + "="*70)
    print("TESTING COMPLETE!")
    print("="*70)
    print(f"\nFinal Test Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall: {test_results['recall']:.4f}")
    print(f"F1-Score: {test_results['f1_score']:.4f}")
    print(f"\nTotal evaluation time: {eval_time:.2f} seconds")
    print(f"Average time per sample: {eval_time/len(test_faces)*1000:.2f} ms")


if __name__ == "__main__":
    main()
