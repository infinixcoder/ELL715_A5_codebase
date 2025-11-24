"""
Face Detection Script for Viola-Jones Face Detector

This script detects faces in arbitrary images using the trained cascade classifier.
Uses multi-scale detection with sliding window approach.
"""

import numpy as np
import os
import argparse
from PIL import Image
from utils import load_model, resize_image, non_maximum_suppression, visualize_detections
from integral_image import IntegralImage
import time


def detect_faces_multiscale(image, cascade, feature_generator, 
                            scale_factor=1.25, min_window_size=16, 
                            stride=2, confidence_threshold=0.5):
    """
    Detect faces in an image using multi-scale sliding window.
    
    Args:
        image: Grayscale image as numpy array
        cascade: Trained cascade classifier
        feature_generator: HaarFeatureGenerator object
        scale_factor: Scale factor for image pyramid
        min_window_size: Minimum detection window size
        stride: Sliding window stride
        confidence_threshold: Confidence threshold for detection
        
    Returns:
        tuple: (detections, scores) where detections are (x, y, w, h) tuples
    """
    window_size = feature_generator.window_size
    detections = []
    scores = []
    
    h, w = image.shape
    scale = 1.0
    
    print(f"\nDetecting faces in {w}x{h} image...")
    
    # Image pyramid - scale down the image
    while True:
        scaled_height = int(h / scale)
        scaled_width = int(w / scale)
        
        # Stop if image is too small
        if scaled_height < min_window_size or scaled_width < min_window_size:
            break
        
        # Resize image
        if scale == 1.0:
            scaled_image = image
        else:
            scaled_image = resize_image(image, 1.0 / scale)
        
        print(f"  Scanning at scale {scale:.2f} (size: {scaled_image.shape[1]}x{scaled_image.shape[0]})")
        
        # Sliding window
        for y in range(0, scaled_image.shape[0] - window_size + 1, stride):
            for x in range(0, scaled_image.shape[1] - window_size + 1, stride):
                # Extract window
                window = scaled_image[y:y+window_size, x:x+window_size]
                
                # Compute features
                ii = IntegralImage(window)
                feature_values = np.array([
                    feature.compute(ii) for feature in feature_generator.features
                ]).reshape(1, -1)
                
                # Classify with cascade
                prediction = cascade.predict(feature_values)[0]
                
                if prediction == 1:
                    # Get confidence score
                    score = cascade.stages[-1].classifier.predict_proba(feature_values)[0]
                    
                    if score >= confidence_threshold:
                        # Convert to original image coordinates
                        det_x = int(x * scale)
                        det_y = int(y * scale)
                        det_w = int(window_size * scale)
                        det_h = int(window_size * scale)
                        
                        detections.append((det_x, det_y, det_w, det_h))
                        scores.append(score)
        
        # Next scale
        scale *= scale_factor
    
    return detections, scores


def main():
    """Main detection function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Detect faces in images using Viola-Jones")
    parser.add_argument("--image_path", type=str, required=True, 
                       help="Path to input image")
    parser.add_argument("--scale_factor", type=float, default=1.25,
                       help="Scale factor for multi-scale detection (default: 1.25)")
    parser.add_argument("--min_window_size", type=int, default=16,
                       help="Minimum detection window size (default: 16)")
    parser.add_argument("--stride", type=int, default=2,
                       help="Sliding window stride (default: 2)")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                       help="Confidence threshold for detection (default: 0.5)")
    parser.add_argument("--nms_threshold", type=float, default=0.3,
                       help="IoU threshold for non-maximum suppression (default: 0.3)")
    parser.add_argument("--output_path", type=str, default="output/detected_faces.jpg",
                       help="Path to save output image (default: output/detected_faces.jpg)")
    
    args = parser.parse_args()
    
    print("="*70)
    print(" "*20 + "VIOLA-JONES FACE DETECTOR")
    print(" "*25 + "DETECTION SCRIPT")
    print("="*70)
    
    # Load image
    print("\n" + "="*70)
    print("LOADING IMAGE")
    print("="*70)
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return
    
    image = Image.open(args.image_path).convert('L')
    image_array = np.array(image)
    
    print(f"Image loaded: {args.image_path}")
    print(f"Image size: {image_array.shape[1]}x{image_array.shape[0]}")
    
    # Load model
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
    
    print(f"Model loaded successfully")
    print(f"  Number of stages: {len(cascade.stages)}")
    print(f"  Window size: {model_data['window_size']}")
    
    # Detection parameters
    print("\n" + "="*70)
    print("DETECTION PARAMETERS")
    print("="*70)
    print(f"  Scale factor: {args.scale_factor}")
    print(f"  Min window size: {args.min_window_size}")
    print(f"  Stride: {args.stride}")
    print(f"  Confidence threshold: {args.confidence_threshold}")
    print(f"  NMS threshold: {args.nms_threshold}")
    
    # Detect faces
    print("\n" + "="*70)
    print("DETECTING FACES")
    print("="*70)
    
    start_time = time.time()
    
    detections, scores = detect_faces_multiscale(
        image_array,
        cascade,
        feature_generator,
        scale_factor=args.scale_factor,
        min_window_size=args.min_window_size,
        stride=args.stride,
        confidence_threshold=args.confidence_threshold
    )
    
    detection_time = time.time() - start_time
    
    print(f"\nInitial detections: {len(detections)}")
    print(f"Detection time: {detection_time:.2f} seconds")
    
    # Apply non-maximum suppression
    if len(detections) > 0:
        print("\nApplying non-maximum suppression...")
        keep_indices = non_maximum_suppression(detections, scores, 
                                              overlap_threshold=args.nms_threshold)
        
        final_detections = [detections[i] for i in keep_indices]
        final_scores = [scores[i] for i in keep_indices]
        
        print(f"Final detections after NMS: {len(final_detections)}")
    else:
        final_detections = []
        final_scores = []
        print("\nNo faces detected!")
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Create output directory
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Visualize and save
    if len(final_detections) > 0:
        visualize_detections(
            image_array,
            final_detections,
            output_path=args.output_path,
            title=f"Face Detection Results"
        )
        
        # Print detection details
        print(f"\nDetected {len(final_detections)} face(s):")
        for i, ((x, y, w, h), score) in enumerate(zip(final_detections, final_scores)):
            print(f"  Face {i+1}: position=({x}, {y}), size={w}x{h}, confidence={score:.3f}")
    else:
        # Save image without detections
        image.save(args.output_path)
        print(f"No faces detected. Original image saved to {args.output_path}")
    
    # Summary
    print("\n" + "="*70)
    print("DETECTION COMPLETE!")
    print("="*70)
    print(f"Total faces detected: {len(final_detections)}")
    print(f"Total time: {detection_time:.2f} seconds")
    print(f"Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()
