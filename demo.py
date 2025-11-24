"""
Quick Demo Script for Viola-Jones Face Detector

This script provides a quick demonstration of the detector capabilities.
"""

import numpy as np
import os


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")
    
    required = ['numpy', 'PIL', 'matplotlib', 'sklearn', 'tqdm', 'joblib']
    missing = []
    
    for package in required:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\nAll dependencies installed!")
    return True


def check_data():
    """Check if dataset has been generated."""
    print("\nChecking dataset...")
    
    if os.path.exists("data/train_faces.npy") and os.path.exists("data/test_faces.npy"):
        train_faces = np.load("data/train_faces.npy")
        test_faces = np.load("data/test_faces.npy")
        
        print(f"  ✓ Training data: {len(train_faces)} samples")
        print(f"  ✓ Test data: {len(test_faces)} samples")
        return True
    else:
        print("  ✗ Dataset not found")
        print("\nGenerate dataset with: python dataset_generator.py")
        return False


def check_model():
    """Check if model has been trained."""
    print("\nChecking trained model...")
    
    if os.path.exists("models/viola_jones_cascade.pkl"):
        print("  ✓ Trained model found")
        return True
    else:
        print("  ✗ Trained model not found")
        print("\nTrain model with: python train.py")
        return False


def show_stats():
    """Show dataset and model statistics."""
    print("\n" + "="*60)
    print("VIOLA-JONES FACE DETECTOR - STATUS")
    print("="*60)
    
    # Check components
    has_deps = check_dependencies()
    has_data = check_data()
    has_model = check_model()
    
    # Summary
    print("\n" + "="*60)
    print("SETUP STATUS")
    print("="*60)
    
    if has_deps and has_data and has_model:
        print("✓ System ready for face detection!")
        print("\nNext steps:")
        print("  1. Test model: python test.py")
        print("  2. Detect faces: python detect_faces.py --image_path <path>")
    elif has_deps and has_data:
        print("⚠ Dependencies and data ready, but model not trained")
        print("\nNext step:")
        print("  Train model: python train.py")
    elif has_deps:
        print("⚠ Dependencies ready, but data and model missing")
        print("\nNext steps:")
        print("  1. Generate data: python dataset_generator.py")
        print("  2. Train model: python train.py")
    else:
        print("✗ Missing dependencies")
        print("\nNext step:")
        print("  Install dependencies: pip install -r requirements.txt")


def run_quick_test():
    """Run a quick test on synthetic data."""
    print("\n" + "="*60)
    print("RUNNING QUICK TEST ON SYNTHETIC DATA")
    print("="*60)
    
    try:
        from haar_features import HaarFeatureGenerator
        from adaboost import AdaBoost
        
        # Generate synthetic data
        print("\nGenerating synthetic test data...")
        np.random.seed(42)
        n_samples = 100
        n_features = 50
        
        feature_matrix = np.random.randn(n_samples, n_features)
        labels = (feature_matrix[:, 0] + feature_matrix[:, 1] > 0).astype(np.int32)
        
        print(f"  Samples: {n_samples}")
        print(f"  Features: {n_features}")
        print(f"  Positive: {np.sum(labels == 1)}")
        print(f"  Negative: {np.sum(labels == 0)}")
        
        # Train small AdaBoost
        print("\nTraining AdaBoost classifier (5 features)...")
        ada = AdaBoost(n_estimators=5)
        ada.fit(feature_matrix, labels, verbose=False)
        
        # Evaluate
        predictions = ada.predict(feature_matrix)
        accuracy = np.mean(predictions == labels)
        
        print(f"\nQuick test results:")
        print(f"  Training accuracy: {accuracy:.4f}")
        
        if accuracy > 0.7:
            print("  ✓ Basic functionality working!")
        else:
            print("  ⚠ Lower than expected accuracy")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during quick test: {e}")
        return False


def main():
    """Main demo function."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_quick_test()
    else:
        show_stats()
        
        print("\n" + "="*60)
        print("USAGE")
        print("="*60)
        print("\nFull workflow:")
        print("  1. python dataset_generator.py   # Generate dataset")
        print("  2. python train.py              # Train cascade")
        print("  3. python test.py               # Evaluate on test set")
        print("  4. python detect_faces.py --image_path <image>  # Detect faces")
        print("\nQuick test:")
        print("  python demo.py --test           # Run synthetic data test")
        print("\nFor detailed instructions, see INSTRUCTIONS.md")


if __name__ == "__main__":
    main()
