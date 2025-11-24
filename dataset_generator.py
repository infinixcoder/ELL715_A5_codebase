"""
Dataset Generation Module for Viola-Jones Face Detector

This module generates training and testing datasets by extracting 16x16 patches
from images in the faces94 dataset.

According to specifications:
- 'face' class: 16x16 patch from center of each image
- 'not-a-face' class: 5 random 16x16 patches from each image
- Training data: female/ and malestaff/ folders
- Testing data: male/ folder
"""

import os
import numpy as np
from PIL import Image
import random
from tqdm import tqdm


class DatasetGenerator:
    """Generate face and non-face patches from image dataset."""
    
    def __init__(self, patch_size=16, num_negative_patches=5):
        """
        Initialize dataset generator.
        
        Args:
            patch_size: Size of square patches to extract (default: 16)
            num_negative_patches: Number of random patches per image (default: 5)
        """
        self.patch_size = patch_size
        self.num_negative_patches = num_negative_patches
        
    def extract_center_patch(self, image):
        """
        Extract a patch from the center of the image.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            numpy array of shape (patch_size, patch_size)
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert('L'))  # Convert to grayscale
        
        h, w = image.shape
        center_y, center_x = h // 2, w // 2
        half_patch = self.patch_size // 2
        
        # Extract centered patch
        y_start = max(0, center_y - half_patch)
        y_end = min(h, center_y + half_patch)
        x_start = max(0, center_x - half_patch)
        x_end = min(w, center_x + half_patch)
        
        patch = image[y_start:y_end, x_start:x_end]
        
        # Resize if necessary
        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
            patch = np.array(Image.fromarray(patch).resize(
                (self.patch_size, self.patch_size), Image.BILINEAR
            ))
        
        return patch
    
    def extract_random_patches(self, image, num_patches):
        """
        Extract random patches from the image.
        
        Args:
            image: PIL Image or numpy array
            num_patches: Number of patches to extract
            
        Returns:
            list of numpy arrays, each of shape (patch_size, patch_size)
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert('L'))
        
        h, w = image.shape
        patches = []
        
        # Generate random positions avoiding the center
        for _ in range(num_patches):
            max_y = max(1, h - self.patch_size)
            max_x = max(1, w - self.patch_size)
            
            y = random.randint(0, max_y - 1) if max_y > 1 else 0
            x = random.randint(0, max_x - 1) if max_x > 1 else 0
            
            patch = image[y:y+self.patch_size, x:x+self.patch_size]
            
            # Resize if necessary
            if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                patch = np.array(Image.fromarray(patch).resize(
                    (self.patch_size, self.patch_size), Image.BILINEAR
                ))
            
            patches.append(patch)
        
        return patches
    
    def load_images_from_folder(self, folder_path):
        """
        Load all images from a folder and its subfolders.
        
        Args:
            folder_path: Path to folder containing images
            
        Returns:
            list of PIL Images
        """
        images = []
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    try:
                        img_path = os.path.join(root, file)
                        img = Image.open(img_path)
                        images.append(img)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
        
        return images
    
    def generate_dataset(self, folders, is_training=True):
        """
        Generate dataset from specified folders.
        
        Args:
            folders: List of folder paths to process
            is_training: Whether this is training data (affects progress description)
            
        Returns:
            tuple: (faces_array, labels_array) where labels are 1 for face, 0 for not-face
        """
        all_faces = []
        all_labels = []
        
        dataset_type = "Training" if is_training else "Testing"
        
        for folder in folders:
            if not os.path.exists(folder):
                print(f"Warning: Folder {folder} does not exist, skipping...")
                continue
            
            print(f"\nProcessing {folder}...")
            images = self.load_images_from_folder(folder)
            
            print(f"Found {len(images)} images")
            
            for img in tqdm(images, desc=f"{dataset_type} - {os.path.basename(folder)}"):
                # Extract center patch (face)
                face_patch = self.extract_center_patch(img)
                all_faces.append(face_patch)
                all_labels.append(1)  # Label 1 for face
                
                # Extract random patches (not-face)
                non_face_patches = self.extract_random_patches(img, self.num_negative_patches)
                all_faces.extend(non_face_patches)
                all_labels.extend([0] * len(non_face_patches))  # Label 0 for not-face
        
        # Convert to numpy arrays
        faces_array = np.array(all_faces, dtype=np.uint8)
        labels_array = np.array(all_labels, dtype=np.int32)
        
        return faces_array, labels_array


def main():
    """Main function to generate and save datasets."""
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize generator
    generator = DatasetGenerator(patch_size=16, num_negative_patches=5)
    
    # Define base path
    base_path = "faces94"
    
    # Training folders: female and malestaff
    train_folders = [
        os.path.join(base_path, "female"),
        os.path.join(base_path, "malestaff")
    ]
    
    # Testing folder: male
    test_folders = [
        os.path.join(base_path, "male")
    ]
    
    # Create output directory
    os.makedirs("data", exist_ok=True)
    
    # Generate training dataset
    print("="*60)
    print("GENERATING TRAINING DATASET")
    print("="*60)
    train_faces, train_labels = generator.generate_dataset(train_folders, is_training=True)
    
    print(f"\nTraining dataset generated:")
    print(f"  Total samples: {len(train_faces)}")
    print(f"  Face samples: {np.sum(train_labels == 1)}")
    print(f"  Non-face samples: {np.sum(train_labels == 0)}")
    print(f"  Shape: {train_faces.shape}")
    
    # Save training dataset
    np.save("data/train_faces.npy", train_faces)
    np.save("data/train_labels.npy", train_labels)
    print("\nTraining data saved to data/train_faces.npy and data/train_labels.npy")
    
    # Generate testing dataset
    print("\n" + "="*60)
    print("GENERATING TESTING DATASET")
    print("="*60)
    test_faces, test_labels = generator.generate_dataset(test_folders, is_training=False)
    
    print(f"\nTesting dataset generated:")
    print(f"  Total samples: {len(test_faces)}")
    print(f"  Face samples: {np.sum(test_labels == 1)}")
    print(f"  Non-face samples: {np.sum(test_labels == 0)}")
    print(f"  Shape: {test_faces.shape}")
    
    # Save testing dataset
    np.save("data/test_faces.npy", test_faces)
    np.save("data/test_labels.npy", test_labels)
    print("\nTesting data saved to data/test_faces.npy and data/test_labels.npy")
    
    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
