"""
Viola-Jones Face Detector Implementation
Complete Main Module

This module provides a high-level interface to the Viola-Jones face detector.
"""

import numpy as np
from haar_features import HaarFeatureGenerator
from cascade_classifier import CascadeClassifier, train_cascade
from integral_image import IntegralImage
from utils import load_model, save_model


class ViolaJonesDetector:
    """
    Complete Viola-Jones face detector.
    
    This class provides a unified interface for training and using
    the Viola-Jones face detection algorithm.
    """
    
    def __init__(self, window_size=16):
        """
        Initialize detector.
        
        Args:
            window_size: Size of detection window (default: 16x16)
        """
        self.window_size = window_size
        self.feature_generator = HaarFeatureGenerator(window_size=window_size)
        self.cascade = None
        self.trained = False
    
    def generate_features(self, min_feature_size=2, max_feature_size=None):
        """
        Generate Haar features.
        
        Args:
            min_feature_size: Minimum feature size
            max_feature_size: Maximum feature size (default: window_size)
        """
        if max_feature_size is None:
            max_feature_size = self.window_size
        
        self.feature_generator.generate_all_features(
            min_feature_size=min_feature_size,
            max_feature_size=max_feature_size
        )
        
        return len(self.feature_generator.features)
    
    def extract_features(self, images):
        """
        Extract feature values from images.
        
        Args:
            images: numpy array of shape (n_samples, height, width)
            
        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        return self.feature_generator.compute_features_for_images(images)
    
    def train(self, train_images, train_labels, 
             val_images, val_labels,
             num_stages=5,
             features_per_stage=None,
             target_detection_rate=0.995,
             target_fp_rate=0.5,
             verbose=True):
        """
        Train the cascade classifier.
        
        Args:
            train_images: Training images
            train_labels: Training labels
            val_images: Validation images
            val_labels: Validation labels
            num_stages: Number of cascade stages
            features_per_stage: List of features per stage
            target_detection_rate: Target detection rate per stage
            target_fp_rate: Target false positive rate per stage
            verbose: Print progress
        """
        # Extract features
        if verbose:
            print("Extracting training features...")
        train_features = self.extract_features(train_images)
        
        if verbose:
            print("Extracting validation features...")
        val_features = self.extract_features(val_images)
        
        # Train cascade
        self.cascade = train_cascade(
            train_features,
            train_labels,
            val_features,
            val_labels,
            num_stages=num_stages,
            features_per_stage=features_per_stage,
            target_detection_rate=target_detection_rate,
            target_fp_rate=target_fp_rate,
            verbose=verbose
        )
        
        self.trained = True
    
    def predict(self, images):
        """
        Predict face/non-face for images.
        
        Args:
            images: numpy array of images
            
        Returns:
            numpy array of predictions
        """
        if not self.trained:
            raise ValueError("Model not trained! Call train() first or load a trained model.")
        
        features = self.extract_features(images)
        return self.cascade.predict(features)
    
    def detect_single_scale(self, image, stride=1):
        """
        Detect faces in an image at single scale.
        
        Args:
            image: Grayscale image
            stride: Sliding window stride
            
        Returns:
            List of (x, y, w, h) detections
        """
        if not self.trained:
            raise ValueError("Model not trained! Call train() first or load a trained model.")
        
        detections = []
        h, w = image.shape
        
        for y in range(0, h - self.window_size + 1, stride):
            for x in range(0, w - self.window_size + 1, stride):
                window = image[y:y+self.window_size, x:x+self.window_size]
                
                # Extract features
                ii = IntegralImage(window)
                feature_values = np.array([
                    feature.compute(ii) for feature in self.feature_generator.features
                ]).reshape(1, -1)
                
                # Predict
                prediction = self.cascade.predict(feature_values)[0]
                
                if prediction == 1:
                    detections.append((x, y, self.window_size, self.window_size))
        
        return detections
    
    def save(self, filepath):
        """
        Save trained detector to file.
        
        Args:
            filepath: Path to save file
        """
        if not self.trained:
            raise ValueError("Model not trained! Nothing to save.")
        
        model_data = {
            'cascade': self.cascade,
            'feature_generator': self.feature_generator,
            'window_size': self.window_size,
            'trained': self.trained
        }
        
        save_model(model_data, filepath)
    
    @classmethod
    def load(cls, filepath):
        """
        Load trained detector from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            ViolaJonesDetector object
        """
        model_data = load_model(filepath)
        
        detector = cls(window_size=model_data['window_size'])
        detector.cascade = model_data['cascade']
        detector.feature_generator = model_data['feature_generator']
        detector.trained = model_data.get('trained', True)
        
        return detector


# Example usage
if __name__ == "__main__":
    print("Viola-Jones Face Detector")
    print("="*50)
    print("\nThis is the main module. To use the detector:")
    print("  1. Generate dataset: python dataset_generator.py")
    print("  2. Train model: python train.py")
    print("  3. Test model: python test.py")
    print("  4. Detect faces: python detect_faces.py --image_path <path>")
    print("\nOr use the ViolaJonesDetector class programmatically:")
    print("""
    from viola_jones_detector import ViolaJonesDetector
    
    # Create detector
    detector = ViolaJonesDetector(window_size=16)
    
    # Generate features
    detector.generate_features()
    
    # Train
    detector.train(train_images, train_labels, val_images, val_labels)
    
    # Save
    detector.save('my_model.pkl')
    
    # Load
    detector = ViolaJonesDetector.load('my_model.pkl')
    
    # Predict
    predictions = detector.predict(test_images)
    """)
