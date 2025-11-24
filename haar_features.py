"""
Haar Features Module for Viola-Jones Face Detector

This module implements Haar-like rectangle features used in the Viola-Jones algorithm.
Features include two-rectangle, three-rectangle, and four-rectangle features at
multiple scales and locations.
"""

import numpy as np
from integral_image import IntegralImage


class HaarFeature:
    """Represents a single Haar-like feature."""
    
    def __init__(self, feature_type, x, y, width, height, orientation='horizontal'):
        """
        Initialize a Haar feature.
        
        Args:
            feature_type: Type of feature (2, 3, or 4 for number of rectangles)
            x: Top-left x coordinate
            y: Top-left y coordinate
            width: Feature width
            height: Feature height
            orientation: 'horizontal' or 'vertical' (for 2 and 3-rectangle features)
        """
        self.feature_type = feature_type
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.orientation = orientation
    
    def compute(self, integral_image):
        """
        Compute feature value using integral image.
        
        Args:
            integral_image: IntegralImage object
            
        Returns:
            Feature value (float)
        """
        if self.feature_type == 2:
            return integral_image.get_two_rectangle_feature(
                self.x, self.y, self.width, self.height, self.orientation
            )
        elif self.feature_type == 3:
            return integral_image.get_three_rectangle_feature(
                self.x, self.y, self.width, self.height, self.orientation
            )
        elif self.feature_type == 4:
            return integral_image.get_four_rectangle_feature(
                self.x, self.y, self.width, self.height
            )
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
    
    def __repr__(self):
        return f"HaarFeature(type={self.feature_type}, pos=({self.x},{self.y}), size=({self.width}x{self.height}), orient={self.orientation})"


class HaarFeatureGenerator:
    """Generate all possible Haar features for a given window size."""
    
    def __init__(self, window_size=16):
        """
        Initialize feature generator.
        
        Args:
            window_size: Size of detection window (default: 16x16)
        """
        self.window_size = window_size
        self.features = []
    
    def generate_all_features(self, min_feature_size=2, max_feature_size=None):
        """
        Generate all possible Haar features within the window.
        
        This creates an overcomplete set of rectangle features at multiple
        scales and locations (over 180,000 for 24x24 window).
        
        Args:
            min_feature_size: Minimum size of a feature rectangle
            max_feature_size: Maximum size of a feature (default: window_size)
            
        Returns:
            List of HaarFeature objects
        """
        if max_feature_size is None:
            max_feature_size = self.window_size
        
        features = []
        
        # Generate two-rectangle features (horizontal and vertical)
        features.extend(self._generate_two_rectangle_features(min_feature_size, max_feature_size))
        
        # Generate three-rectangle features (horizontal and vertical)
        features.extend(self._generate_three_rectangle_features(min_feature_size, max_feature_size))
        
        # Generate four-rectangle features
        features.extend(self._generate_four_rectangle_features(min_feature_size, max_feature_size))
        
        self.features = features
        return features
    
    def _generate_two_rectangle_features(self, min_size, max_size):
        """Generate all two-rectangle features."""
        features = []
        
        # Horizontal two-rectangle features
        for h in range(min_size, max_size + 1):
            for w in range(min_size * 2, max_size + 1, 2):  # Width must be even
                for y in range(0, self.window_size - h + 1):
                    for x in range(0, self.window_size - w + 1):
                        features.append(HaarFeature(2, x, y, w, h, 'horizontal'))
        
        # Vertical two-rectangle features
        for w in range(min_size, max_size + 1):
            for h in range(min_size * 2, max_size + 1, 2):  # Height must be even
                for y in range(0, self.window_size - h + 1):
                    for x in range(0, self.window_size - w + 1):
                        features.append(HaarFeature(2, x, y, w, h, 'vertical'))
        
        return features
    
    def _generate_three_rectangle_features(self, min_size, max_size):
        """Generate all three-rectangle features."""
        features = []
        
        # Horizontal three-rectangle features
        for h in range(min_size, max_size + 1):
            for w in range(min_size * 3, max_size + 1, 3):  # Width must be divisible by 3
                for y in range(0, self.window_size - h + 1):
                    for x in range(0, self.window_size - w + 1):
                        features.append(HaarFeature(3, x, y, w, h, 'horizontal'))
        
        # Vertical three-rectangle features
        for w in range(min_size, max_size + 1):
            for h in range(min_size * 3, max_size + 1, 3):  # Height must be divisible by 3
                for y in range(0, self.window_size - h + 1):
                    for x in range(0, self.window_size - w + 1):
                        features.append(HaarFeature(3, x, y, w, h, 'vertical'))
        
        return features
    
    def _generate_four_rectangle_features(self, min_size, max_size):
        """Generate all four-rectangle features."""
        features = []
        
        for w in range(min_size * 2, max_size + 1, 2):  # Width must be even
            for h in range(min_size * 2, max_size + 1, 2):  # Height must be even
                for y in range(0, self.window_size - h + 1):
                    for x in range(0, self.window_size - w + 1):
                        features.append(HaarFeature(4, x, y, w, h))
        
        return features
    
    def compute_features_for_images(self, images):
        """
        Compute all feature values for a set of images.
        
        Args:
            images: numpy array of shape (n_samples, height, width)
            
        Returns:
            numpy array of shape (n_samples, n_features)
        """
        if not self.features:
            self.generate_all_features()
        
        n_samples = len(images)
        n_features = len(self.features)
        
        feature_matrix = np.zeros((n_samples, n_features), dtype=np.float64)
        
        for i, image in enumerate(images):
            ii = IntegralImage(image)
            for j, feature in enumerate(self.features):
                feature_matrix[i, j] = feature.compute(ii)
        
        return feature_matrix


def visualize_haar_features():
    """Visualize different types of Haar features."""
    import matplotlib.pyplot as plt
    
    # Create sample features
    features = [
        HaarFeature(2, 2, 2, 8, 4, 'horizontal'),
        HaarFeature(2, 2, 2, 4, 8, 'vertical'),
        HaarFeature(3, 1, 1, 12, 4, 'horizontal'),
        HaarFeature(3, 1, 1, 4, 12, 'vertical'),
        HaarFeature(4, 2, 2, 8, 8),
    ]
    
    titles = [
        'Two-Rectangle (Horizontal)',
        'Two-Rectangle (Vertical)',
        'Three-Rectangle (Horizontal)',
        'Three-Rectangle (Vertical)',
        'Four-Rectangle'
    ]
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    for idx, (feature, title) in enumerate(zip(features, titles)):
        canvas = np.zeros((16, 16))
        
        if feature.feature_type == 2:
            if feature.orientation == 'horizontal':
                hw = feature.width // 2
                canvas[feature.y:feature.y+feature.height, feature.x:feature.x+hw] = 1
                canvas[feature.y:feature.y+feature.height, feature.x+hw:feature.x+feature.width] = -1
            else:
                hh = feature.height // 2
                canvas[feature.y:feature.y+hh, feature.x:feature.x+feature.width] = 1
                canvas[feature.y+hh:feature.y+feature.height, feature.x:feature.x+feature.width] = -1
        
        elif feature.feature_type == 3:
            if feature.orientation == 'horizontal':
                tw = feature.width // 3
                canvas[feature.y:feature.y+feature.height, feature.x:feature.x+tw] = 1
                canvas[feature.y:feature.y+feature.height, feature.x+tw:feature.x+2*tw] = -1
                canvas[feature.y:feature.y+feature.height, feature.x+2*tw:feature.x+feature.width] = 1
            else:
                th = feature.height // 3
                canvas[feature.y:feature.y+th, feature.x:feature.x+feature.width] = 1
                canvas[feature.y+th:feature.y+2*th, feature.x:feature.x+feature.width] = -1
                canvas[feature.y+2*th:feature.y+feature.height, feature.x:feature.x+feature.width] = 1
        
        elif feature.feature_type == 4:
            hw = feature.width // 2
            hh = feature.height // 2
            canvas[feature.y:feature.y+hh, feature.x:feature.x+hw] = 1
            canvas[feature.y:feature.y+hh, feature.x+hw:feature.x+feature.width] = -1
            canvas[feature.y+hh:feature.y+feature.height, feature.x:feature.x+hw] = -1
            canvas[feature.y+hh:feature.y+feature.height, feature.x+hw:feature.x+feature.width] = 1
        
        axes[idx].imshow(canvas, cmap='RdBu', vmin=-1, vmax=1)
        axes[idx].set_title(title)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('haar_features_visualization.png', dpi=150, bbox_inches='tight')
    print("Haar features visualization saved to 'haar_features_visualization.png'")
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate features for 16x16 window
    generator = HaarFeatureGenerator(window_size=16)
    features = generator.generate_all_features(min_feature_size=2)
    
    print(f"Generated {len(features)} Haar features for 16x16 window")
    print(f"\nFirst 5 features:")
    for i, feat in enumerate(features[:5]):
        print(f"  {i+1}. {feat}")
    
    # Test feature computation
    test_image = np.random.rand(16, 16) * 255
    ii = IntegralImage(test_image)
    
    print(f"\nTest feature value: {features[0].compute(ii):.2f}")
    
    # Visualize features
    print("\nGenerating visualization...")
    visualize_haar_features()
