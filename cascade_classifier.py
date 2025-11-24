"""
Cascade of Classifiers Module for Viola-Jones Face Detector

This module implements the cascade structure where multiple AdaBoost classifiers
are arranged in series to quickly reject non-face regions while maintaining
high detection rate.
"""

import numpy as np
from adaboost import AdaBoost
from tqdm import tqdm


class CascadeStage:
    """
    A single stage in the cascade of classifiers.
    
    Each stage is an AdaBoost classifier with an adjusted threshold
    to achieve high detection rate and low false positive rate.
    """
    
    def __init__(self, classifier, threshold):
        """
        Initialize cascade stage.
        
        Args:
            classifier: AdaBoost classifier
            threshold: Classification threshold (adjusted for high detection rate)
        """
        self.classifier = classifier
        self.threshold = threshold
    
    def predict(self, feature_matrix):
        """
        Predict using this stage.
        
        Args:
            feature_matrix: numpy array of shape (n_samples, n_features)
            
        Returns:
            numpy array of predictions (1 for pass, 0 for reject)
        """
        # Get confidence scores
        scores = self.classifier.predict_proba(feature_matrix)
        
        # Apply threshold
        predictions = (scores >= self.threshold).astype(np.int32)
        
        return predictions
    
    def evaluate_on_validation(self, feature_matrix, labels):
        """
        Evaluate stage performance on validation set.
        
        Args:
            feature_matrix: Validation features
            labels: Validation labels
            
        Returns:
            dict with detection_rate and false_positive_rate
        """
        predictions = self.predict(feature_matrix)
        
        # Detection rate (true positive rate)
        n_positives = np.sum(labels == 1)
        detected = np.sum((predictions == 1) & (labels == 1))
        detection_rate = detected / n_positives if n_positives > 0 else 0
        
        # False positive rate
        n_negatives = np.sum(labels == 0)
        false_positives = np.sum((predictions == 1) & (labels == 0))
        false_positive_rate = false_positives / n_negatives if n_negatives > 0 else 0
        
        return {
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'n_positives': n_positives,
            'n_detected': detected,
            'n_negatives': n_negatives,
            'n_false_positives': false_positives
        }


class CascadeClassifier:
    """
    Cascade of AdaBoost classifiers for fast face detection.
    
    The cascade works as a degenerate decision tree where:
    - Each stage is more complex than the previous
    - A negative outcome at any stage immediately rejects the sample
    - All stages must output positive for final acceptance
    """
    
    def __init__(self):
        """Initialize empty cascade."""
        self.stages = []
    
    def add_stage(self, stage):
        """
        Add a stage to the cascade.
        
        Args:
            stage: CascadeStage object
        """
        self.stages.append(stage)
    
    def predict(self, feature_matrix):
        """
        Predict using full cascade.
        
        Sample must pass all stages to be classified as face.
        
        Args:
            feature_matrix: numpy array of shape (n_samples, n_features)
            
        Returns:
            numpy array of predictions (1 for face, 0 for not-face)
        """
        n_samples = feature_matrix.shape[0]
        predictions = np.ones(n_samples, dtype=np.int32)
        
        # Apply each stage sequentially
        for stage in self.stages:
            # Only process samples that passed previous stages
            active_indices = np.where(predictions == 1)[0]
            
            if len(active_indices) == 0:
                break
            
            # Get predictions for active samples
            stage_predictions = stage.predict(feature_matrix[active_indices])
            
            # Update overall predictions
            predictions[active_indices] = stage_predictions
        
        return predictions
    
    def predict_with_stage_info(self, feature_matrix):
        """
        Predict and return which stage rejected each sample.
        
        Args:
            feature_matrix: numpy array of shape (n_samples, n_features)
            
        Returns:
            tuple: (predictions, rejection_stages)
                predictions: final predictions
                rejection_stages: stage index where sample was rejected (-1 if passed all)
        """
        n_samples = feature_matrix.shape[0]
        predictions = np.ones(n_samples, dtype=np.int32)
        rejection_stages = np.full(n_samples, -1, dtype=np.int32)
        
        for stage_idx, stage in enumerate(self.stages):
            active_indices = np.where(predictions == 1)[0]
            
            if len(active_indices) == 0:
                break
            
            stage_predictions = stage.predict(feature_matrix[active_indices])
            
            # Mark rejected samples
            rejected = active_indices[stage_predictions == 0]
            rejection_stages[rejected] = stage_idx
            
            predictions[active_indices] = stage_predictions
        
        return predictions, rejection_stages
    
    def evaluate(self, feature_matrix, labels):
        """
        Evaluate cascade on data.
        
        Args:
            feature_matrix: numpy array of shape (n_samples, n_features)
            labels: Ground truth labels
            
        Returns:
            dict with overall metrics and per-stage metrics
        """
        predictions = self.predict(feature_matrix)
        
        # Overall metrics
        accuracy = np.mean(predictions == labels)
        
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Per-stage metrics
        stage_metrics = []
        active_samples = np.ones(len(labels), dtype=bool)
        
        for stage_idx, stage in enumerate(self.stages):
            if np.sum(active_samples) == 0:
                break
            
            stage_preds = stage.predict(feature_matrix[active_samples])
            
            # Calculate metrics for samples reaching this stage
            stage_labels = labels[active_samples]
            
            stage_tp = np.sum((stage_preds == 1) & (stage_labels == 1))
            stage_fp = np.sum((stage_preds == 1) & (stage_labels == 0))
            stage_fn = np.sum((stage_preds == 0) & (stage_labels == 1))
            
            stage_detection_rate = stage_tp / np.sum(stage_labels == 1) if np.sum(stage_labels == 1) > 0 else 0
            stage_fp_rate = stage_fp / np.sum(stage_labels == 0) if np.sum(stage_labels == 0) > 0 else 0
            
            stage_metrics.append({
                'stage': stage_idx,
                'samples_reaching': np.sum(active_samples),
                'detection_rate': stage_detection_rate,
                'false_positive_rate': stage_fp_rate
            })
            
            # Update active samples
            active_samples[active_samples] = (stage_preds == 1)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'stage_metrics': stage_metrics
        }


def train_cascade(feature_matrix, labels, 
                 validation_feature_matrix, validation_labels,
                 num_stages=5,
                 features_per_stage=None,
                 target_detection_rate=0.995,
                 target_fp_rate=0.5,
                 verbose=True):
    """
    Train a cascade of classifiers.
    
    Training procedure:
    1. For each stage:
        a. Train AdaBoost classifier with increasing number of features
        b. Adjust threshold to meet target detection rate
        c. Collect false positives from current cascade for next stage training
        d. Add stage to cascade
    2. Continue until overall target is met or max stages reached
    
    Args:
        feature_matrix: Training features
        labels: Training labels
        validation_feature_matrix: Validation features for threshold tuning
        validation_labels: Validation labels
        num_stages: Number of cascade stages
        features_per_stage: List of feature counts per stage (default: increasing)
        target_detection_rate: Minimum detection rate per stage (default: 0.995)
        target_fp_rate: Maximum false positive rate per stage (default: 0.5)
        verbose: Print progress
        
    Returns:
        CascadeClassifier object
    """
    if features_per_stage is None:
        # Default: increasing number of features
        features_per_stage = [10 * (i + 1) for i in range(num_stages)]
    
    cascade = CascadeClassifier()
    
    # Start with full training set
    current_train_features = feature_matrix.copy()
    current_train_labels = labels.copy()
    
    if verbose:
        print("="*60)
        print("TRAINING CASCADE OF CLASSIFIERS")
        print("="*60)
        print(f"Number of stages: {num_stages}")
        print(f"Features per stage: {features_per_stage}")
        print(f"Target detection rate: {target_detection_rate}")
        print(f"Target FP rate per stage: {target_fp_rate}")
        print()
    
    for stage_idx in range(num_stages):
        if verbose:
            print(f"\n{'='*60}")
            print(f"TRAINING STAGE {stage_idx + 1}/{num_stages}")
            print(f"{'='*60}")
            print(f"Training samples: {len(current_train_labels)}")
            print(f"  Positive: {np.sum(current_train_labels == 1)}")
            print(f"  Negative: {np.sum(current_train_labels == 0)}")
        
        # Train AdaBoost classifier for this stage
        n_estimators = min(features_per_stage[stage_idx], current_train_features.shape[1])
        classifier = AdaBoost(n_estimators=n_estimators)
        classifier.fit(current_train_features, current_train_labels, verbose=verbose)
        
        # Adjust threshold to meet target detection rate on validation set
        if verbose:
            print(f"\nAdjusting threshold for stage {stage_idx + 1}...")
        
        val_scores = classifier.predict_proba(validation_feature_matrix)
        
        # Find threshold that gives target detection rate
        pos_scores = val_scores[validation_labels == 1]
        pos_scores_sorted = np.sort(pos_scores)
        
        # Threshold at (1 - target_detection_rate) percentile
        threshold_idx = int(len(pos_scores_sorted) * (1 - target_detection_rate))
        threshold_idx = max(0, min(threshold_idx, len(pos_scores_sorted) - 1))
        threshold = pos_scores_sorted[threshold_idx]
        
        # Create stage
        stage = CascadeStage(classifier, threshold)
        
        # Evaluate on validation set
        val_metrics = stage.evaluate_on_validation(validation_feature_matrix, validation_labels)
        
        if verbose:
            print(f"Stage {stage_idx + 1} threshold: {threshold:.4f}")
            print(f"Validation metrics:")
            print(f"  Detection rate: {val_metrics['detection_rate']:.4f}")
            print(f"  False positive rate: {val_metrics['false_positive_rate']:.4f}")
        
        # Add stage to cascade
        cascade.add_stage(stage)
        
        # Prepare training data for next stage using false positives
        # Apply current cascade to training negatives
        if stage_idx < num_stages - 1:
            if verbose:
                print(f"\nCollecting false positives for stage {stage_idx + 2} training...")
            
            # Get predictions from current cascade on negative samples
            neg_indices = np.where(labels == 0)[0]
            neg_features = feature_matrix[neg_indices]
            
            cascade_preds = cascade.predict(neg_features)
            false_positive_indices = neg_indices[cascade_preds == 1]
            
            # New training set: all positives + false positives
            pos_indices = np.where(labels == 1)[0]
            new_train_indices = np.concatenate([pos_indices, false_positive_indices])
            
            current_train_features = feature_matrix[new_train_indices]
            current_train_labels = labels[new_train_indices]
            
            if verbose:
                print(f"  False positives collected: {len(false_positive_indices)}")
                print(f"  Next stage training size: {len(new_train_indices)}")
    
    if verbose:
        print("\n" + "="*60)
        print("CASCADE TRAINING COMPLETE")
        print(f"Total stages: {len(cascade.stages)}")
        print("="*60)
    
    return cascade


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    
    n_train = 2000
    n_val = 500
    n_features = 200
    
    # Training data
    train_features = np.random.randn(n_train, n_features)
    train_labels = ((train_features[:, 0] + train_features[:, 1] > 0) & 
                   (train_features[:, 2] < 1)).astype(np.int32)
    
    # Validation data
    val_features = np.random.randn(n_val, n_features)
    val_labels = ((val_features[:, 0] + val_features[:, 1] > 0) & 
                 (val_features[:, 2] < 1)).astype(np.int32)
    
    print("Synthetic data created")
    print(f"Training: {n_train} samples, {np.sum(train_labels==1)} positive")
    print(f"Validation: {n_val} samples, {np.sum(val_labels==1)} positive")
    
    # Train cascade
    cascade = train_cascade(
        train_features, train_labels,
        val_features, val_labels,
        num_stages=3,
        features_per_stage=[5, 10, 15],
        verbose=True
    )
    
    # Evaluate
    results = cascade.evaluate(val_features, val_labels)
    print("\n\nValidation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
