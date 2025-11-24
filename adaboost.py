"""
AdaBoost Algorithm Module for Viola-Jones Face Detector

This module implements the AdaBoost algorithm for training weak classifiers
and combining them into a strong classifier. Each weak classifier is based on
a single Haar feature.
"""

import numpy as np
from tqdm import tqdm


class WeakClassifier:
    """
    Weak classifier based on a single Haar feature.
    
    The classifier thresholds a single feature value to classify as face/not-face:
        h_j(x) = 1 if p_j * f_j(x) < p_j * theta_j, else 0
    where p_j indicates the direction of the inequality.
    """
    
    def __init__(self, feature_idx, threshold, parity, error):
        """
        Initialize weak classifier.
        
        Args:
            feature_idx: Index of the Haar feature to use
            threshold: Classification threshold
            parity: Direction of inequality (1 or -1)
            error: Training error of this classifier
        """
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.parity = parity
        self.error = error
    
    def classify(self, feature_values):
        """
        Classify samples based on feature values.
        
        Args:
            feature_values: numpy array of feature values
            
        Returns:
            numpy array of predictions (1 for face, 0 for not-face)
        """
        predictions = np.ones(len(feature_values), dtype=np.int32)
        
        if self.parity == 1:
            predictions[feature_values < self.threshold] = 0
        else:
            predictions[feature_values >= self.threshold] = 0
        
        return predictions
    
    def __repr__(self):
        return f"WeakClassifier(feature={self.feature_idx}, threshold={self.threshold:.2f}, parity={self.parity}, error={self.error:.4f})"


class AdaBoost:
    """
    AdaBoost algorithm for training a strong classifier from weak classifiers.
    
    Each boosting round selects the best single-feature weak classifier
    and updates sample weights.
    """
    
    def __init__(self, n_estimators=10):
        """
        Initialize AdaBoost.
        
        Args:
            n_estimators: Number of weak classifiers to train (T rounds)
        """
        self.n_estimators = n_estimators
        self.classifiers = []
        self.alphas = []
    
    def train_weak_classifier(self, features, labels, weights, feature_idx):
        """
        Train a single weak classifier for one feature.
        
        Finds optimal threshold and parity to minimize weighted error.
        
        Args:
            features: Feature values for this feature (1D array)
            labels: Ground truth labels (1 for face, 0 for not-face)
            weights: Sample weights (1D array)
            feature_idx: Index of this feature
            
        Returns:
            WeakClassifier object
        """
        n_samples = len(features)
        
        # Sort features and corresponding labels/weights
        sorted_indices = np.argsort(features)
        sorted_features = features[sorted_indices]
        sorted_labels = labels[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # Compute cumulative sums for positive and negative classes
        pos_weights = sorted_weights * sorted_labels
        neg_weights = sorted_weights * (1 - sorted_labels)
        
        total_pos = np.sum(pos_weights)
        total_neg = np.sum(neg_weights)
        
        cum_pos = np.cumsum(pos_weights)
        cum_neg = np.cumsum(neg_weights)
        
        # For each possible threshold, compute error for both parities
        min_error = float('inf')
        best_threshold = 0
        best_parity = 1
        
        for i in range(n_samples):
            # Threshold between current and next feature value
            if i < n_samples - 1:
                threshold = (sorted_features[i] + sorted_features[i + 1]) / 2
            else:
                threshold = sorted_features[i] + 1
            
            # Error for parity = 1 (predict 0 if feature < threshold)
            # Misclassified: negatives below threshold + positives above threshold
            error_1 = cum_neg[i] + (total_pos - cum_pos[i])
            
            # Error for parity = -1 (predict 0 if feature >= threshold)
            # Misclassified: positives below threshold + negatives above threshold
            error_minus1 = cum_pos[i] + (total_neg - cum_neg[i])
            
            # Choose best parity
            if error_1 < min_error:
                min_error = error_1
                best_threshold = threshold
                best_parity = 1
            
            if error_minus1 < min_error:
                min_error = error_minus1
                best_threshold = threshold
                best_parity = -1
        
        return WeakClassifier(feature_idx, best_threshold, best_parity, min_error)
    
    def fit(self, feature_matrix, labels, verbose=True):
        """
        Train AdaBoost classifier.
        
        Implementation follows the algorithm from Viola-Jones paper:
        1. Initialize weights uniformly
        2. For T rounds:
            a. Normalize weights
            b. Train weak classifier for each feature, select best
            c. Update weights based on errors
        3. Return strong classifier as weighted combination
        
        Args:
            feature_matrix: numpy array of shape (n_samples, n_features)
            labels: numpy array of shape (n_samples,)
            verbose: Whether to print progress
            
        Returns:
            self
        """
        n_samples, n_features = feature_matrix.shape
        
        # Initialize weights
        # w_1,i = 1/(2m) for negatives, 1/(2l) for positives
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        
        weights = np.zeros(n_samples, dtype=np.float64)
        weights[labels == 1] = 1.0 / (2 * n_pos)
        weights[labels == 0] = 1.0 / (2 * n_neg)
        
        if verbose:
            print(f"\nTraining AdaBoost with {self.n_estimators} estimators...")
            print(f"Training samples: {n_samples} ({n_pos} positive, {n_neg} negative)")
            print(f"Number of features: {n_features}")
        
        # Boosting rounds
        iterator = tqdm(range(self.n_estimators), desc="AdaBoost rounds") if verbose else range(self.n_estimators)
        
        for t in iterator:
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Train weak classifier for each feature and select best
            best_classifier = None
            min_error = float('inf')
            
            # Sample features if too many (for efficiency)
            if n_features > 5000:
                feature_indices = np.random.choice(n_features, 5000, replace=False)
            else:
                feature_indices = np.arange(n_features)
            
            for feature_idx in feature_indices:
                classifier = self.train_weak_classifier(
                    feature_matrix[:, feature_idx],
                    labels,
                    weights,
                    feature_idx
                )
                
                if classifier.error < min_error:
                    min_error = classifier.error
                    best_classifier = classifier
            
            # Avoid division by zero
            epsilon = best_classifier.error
            epsilon = max(epsilon, 1e-10)
            epsilon = min(epsilon, 1 - 1e-10)
            
            # Calculate beta and alpha
            beta = epsilon / (1 - epsilon)
            alpha = np.log(1 / beta)
            
            # Store classifier and weight
            self.classifiers.append(best_classifier)
            self.alphas.append(alpha)
            
            # Update weights
            predictions = best_classifier.classify(feature_matrix[:, best_classifier.feature_idx])
            errors = (predictions != labels).astype(np.float64)
            
            # w_t+1,i = w_t,i * beta^(1-e_i)
            weights = weights * np.power(beta, 1 - errors)
            
            if verbose and (t + 1) % max(1, self.n_estimators // 10) == 0:
                train_predictions = self.predict(feature_matrix)
                train_accuracy = np.mean(train_predictions == labels)
                print(f"  Round {t+1}/{self.n_estimators}: error={epsilon:.4f}, alpha={alpha:.4f}, train_acc={train_accuracy:.4f}")
        
        return self
    
    def predict(self, feature_matrix):
        """
        Predict using strong classifier (weighted majority vote).
        
        h(x) = 1 if sum(alpha_t * h_t(x)) >= 0.5 * sum(alpha_t), else 0
        
        Args:
            feature_matrix: numpy array of shape (n_samples, n_features)
            
        Returns:
            numpy array of predictions
        """
        n_samples = feature_matrix.shape[0]
        
        # Compute weighted sum of weak classifier outputs
        scores = np.zeros(n_samples, dtype=np.float64)
        
        for classifier, alpha in zip(self.classifiers, self.alphas):
            predictions = classifier.classify(feature_matrix[:, classifier.feature_idx])
            scores += alpha * predictions
        
        # Threshold at half of total alpha
        threshold = 0.5 * np.sum(self.alphas)
        predictions = (scores >= threshold).astype(np.int32)
        
        return predictions
    
    def predict_proba(self, feature_matrix):
        """
        Predict confidence scores (normalized weighted sum).
        
        Args:
            feature_matrix: numpy array of shape (n_samples, n_features)
            
        Returns:
            numpy array of confidence scores (0 to 1)
        """
        n_samples = feature_matrix.shape[0]
        
        scores = np.zeros(n_samples, dtype=np.float64)
        
        for classifier, alpha in zip(self.classifiers, self.alphas):
            predictions = classifier.classify(feature_matrix[:, classifier.feature_idx])
            scores += alpha * predictions
        
        # Normalize by total alpha
        total_alpha = np.sum(self.alphas)
        if total_alpha > 0:
            scores = scores / total_alpha
        
        return scores
    
    def evaluate(self, feature_matrix, labels):
        """
        Evaluate classifier on data.
        
        Args:
            feature_matrix: numpy array of shape (n_samples, n_features)
            labels: Ground truth labels
            
        Returns:
            dict with accuracy, precision, recall, f1_score
        """
        predictions = self.predict(feature_matrix)
        
        accuracy = np.mean(predictions == labels)
        
        # Calculate precision, recall, F1
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 100
    
    # Create synthetic feature matrix
    feature_matrix = np.random.randn(n_samples, n_features)
    
    # Create labels based on first few features
    labels = ((feature_matrix[:, 0] + feature_matrix[:, 1] > 0) & 
              (feature_matrix[:, 2] < 1)).astype(np.int32)
    
    print("Synthetic data created:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Positive class: {np.sum(labels == 1)}")
    print(f"  Negative class: {np.sum(labels == 0)}")
    
    # Train AdaBoost
    ada = AdaBoost(n_estimators=20)
    ada.fit(feature_matrix, labels, verbose=True)
    
    # Evaluate
    results = ada.evaluate(feature_matrix, labels)
    print("\nTraining Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1-Score: {results['f1_score']:.4f}")
