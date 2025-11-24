"""
Utility Functions for Viola-Jones Face Detector

This module contains helper functions for image processing, visualization,
and other utilities.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_image_grayscale(image_path):
    """
    Load image and convert to grayscale.
    
    Args:
        image_path: Path to image file
        
    Returns:
        numpy array (grayscale image)
    """
    img = Image.open(image_path).convert('L')
    return np.array(img)


def extract_patches(image, window_size, stride=1):
    """
    Extract all possible patches from an image using sliding window.
    
    Args:
        image: 2D numpy array
        window_size: Size of square window
        stride: Step size for sliding window
        
    Returns:
        tuple: (patches, positions)
            patches: numpy array of shape (n_patches, window_size, window_size)
            positions: list of (x, y) coordinates for each patch
    """
    h, w = image.shape
    patches = []
    positions = []
    
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            patch = image[y:y+window_size, x:x+window_size]
            patches.append(patch)
            positions.append((x, y))
    
    return np.array(patches), positions


def resize_image(image, scale):
    """
    Resize image by scale factor.
    
    Args:
        image: numpy array or PIL Image
        scale: Scale factor (e.g., 0.5 for half size)
        
    Returns:
        numpy array (resized image)
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    
    new_size = (int(image.width * scale), int(image.height * scale))
    resized = image.resize(new_size, Image.BILINEAR)
    
    return np.array(resized)


def non_maximum_suppression(detections, scores, overlap_threshold=0.3):
    """
    Apply non-maximum suppression to remove overlapping detections.
    
    Args:
        detections: List of (x, y, width, height) tuples
        scores: List of confidence scores
        overlap_threshold: IoU threshold for suppression
        
    Returns:
        List of indices to keep
    """
    if len(detections) == 0:
        return []
    
    detections = np.array(detections)
    scores = np.array(scores)
    
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 0] + detections[:, 2]
    y2 = detections[:, 1] + detections[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        # Compute IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        # Keep boxes with IoU below threshold
        indices = np.where(iou <= overlap_threshold)[0]
        order = order[indices + 1]
    
    return keep


def visualize_detections(image, detections, output_path=None, title="Face Detections"):
    """
    Visualize detected faces on image.
    
    Args:
        image: numpy array or path to image
        detections: List of (x, y, width, height) tuples
        output_path: Path to save output image (optional)
        title: Plot title
    """
    if isinstance(image, str):
        image = load_image_grayscale(image)
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image, cmap='gray')
    
    for (x, y, w, h) in detections:
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                 edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    
    ax.set_title(f"{title} ({len(detections)} faces detected)")
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Detection result saved to {output_path}")
    
    plt.show()


def save_model(model, filepath):
    """
    Save trained model to file.
    
    Args:
        model: Model object to save
        filepath: Path to save file
    """
    import joblib
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load trained model from file.
    
    Args:
        filepath: Path to model file
        
    Returns:
        Loaded model object
    """
    import joblib
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


def plot_training_history(history, output_path=None):
    """
    Plot training history (accuracy over stages).
    
    Args:
        history: Dict with training metrics
        output_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    stages = range(1, len(history['train_accuracy']) + 1)
    
    # Training accuracy
    axes[0, 0].plot(stages, history['train_accuracy'], 'b-o', label='Train')
    if 'val_accuracy' in history:
        axes[0, 0].plot(stages, history['val_accuracy'], 'r-o', label='Validation')
    axes[0, 0].set_xlabel('Stage')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy over Stages')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Detection rate
    axes[0, 1].plot(stages, history['detection_rate'], 'g-o')
    axes[0, 1].set_xlabel('Stage')
    axes[0, 1].set_ylabel('Detection Rate')
    axes[0, 1].set_title('Detection Rate over Stages')
    axes[0, 1].grid(True)
    
    # False positive rate
    axes[1, 0].plot(stages, history['false_positive_rate'], 'r-o')
    axes[1, 0].set_xlabel('Stage')
    axes[1, 0].set_ylabel('False Positive Rate')
    axes[1, 0].set_title('False Positive Rate over Stages')
    axes[1, 0].grid(True)
    
    # Number of features
    if 'num_features' in history:
        axes[1, 1].bar(stages, history['num_features'])
        axes[1, 1].set_xlabel('Stage')
        axes[1, 1].set_ylabel('Number of Features')
        axes[1, 1].set_title('Features per Stage')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {output_path}")
    
    plt.show()


def compute_metrics(predictions, labels):
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        
    Returns:
        Dict with metrics
    """
    accuracy = np.mean(predictions == labels)
    
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


def print_metrics(metrics, title="Metrics"):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dict with metrics
        title: Title to print
    """
    print(f"\n{title}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TN: {metrics['tn']}")


# Example usage
if __name__ == "__main__":
    # Test patch extraction
    test_image = np.random.rand(100, 100) * 255
    patches, positions = extract_patches(test_image, window_size=16, stride=8)
    print(f"Extracted {len(patches)} patches from 100x100 image")
    
    # Test NMS
    detections = [
        (10, 10, 20, 20),
        (12, 12, 20, 20),
        (50, 50, 20, 20),
    ]
    scores = [0.9, 0.8, 0.95]
    keep = non_maximum_suppression(detections, scores)
    print(f"\nNMS kept {len(keep)} out of {len(detections)} detections")
