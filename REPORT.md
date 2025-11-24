# Viola-Jones Face Detector - Implementation Report

## Project Overview

This project implements the Viola-Jones face detection algorithm from scratch, following the specifications in the original paper. The implementation includes all major components: Haar features, integral images, AdaBoost classifier, and cascade architecture.

## Implementation Details

### 1. Dataset Generation (dataset_generator.py)

**Approach:**
- **Face class**: Extract 16×16 patch from the center of each image
- **Non-face class**: Extract 5 random 16×16 patches from each image
- **Training data**: Images from `female/` and `malestaff/` folders
- **Testing data**: Images from `male/` folder

**Statistics:**
- Training samples: [To be filled after running]
  - Face samples: 
  - Non-face samples: 
- Testing samples: [To be filled after running]
  - Face samples: 
  - Non-face samples: 

### 2. Haar Features (haar_features.py)

**Feature Types Implemented:**
1. **Two-rectangle features**: Horizontal and vertical
2. **Three-rectangle features**: Horizontal and vertical
3. **Four-rectangle features**: Diagonal patterns

**Configuration:**
- Window size: 16×16 pixels
- Minimum feature size: 2 pixels
- Maximum feature size: 16 pixels
- Total features generated: [To be filled after running]

**Feature Extraction:**
- All features are extracted at multiple scales and positions
- Features are computed using integral images for efficiency
- Each feature computes the difference between sums of white and grey rectangles

### 3. Integral Image (integral_image.py)

**Implementation:**
- Uses recurrence relations for O(1) computation per pixel:
  - `s(x,y) = s(x, y-1) + i(x, y)` (cumulative row sum)
  - `ii(x,y) = ii(x-1, y) + s(x, y)` (integral image)

**Efficiency:**
- Two-rectangle feature: 6 array references
- Three-rectangle feature: 8 array references  
- Four-rectangle feature: 9 array references

**Benefits:**
- Any rectangle sum computed in constant time
- Enables real-time feature extraction

### 4. AdaBoost Algorithm (adaboost.py)

**Implementation Details:**

**Weak Classifier:**
- Each weak classifier uses a single Haar feature
- Classification function: `h_j(x) = 1 if p_j*f_j(x) < p_j*θ_j, else 0`
- Threshold and parity optimized to minimize weighted error

**Training Procedure:**
1. Initialize weights: `w_1,i = 1/(2m)` for negatives, `1/(2l)` for positives
2. For T rounds:
   - Normalize weights to form probability distribution
   - Train weak classifier for each feature, select best (lowest error)
   - Calculate β_t = ε_t / (1 - ε_t) and α_t = log(1/β_t)
   - Update weights: `w_t+1,i = w_t,i * β_t^(1-e_i)`
3. Final classifier: Weighted majority vote of weak classifiers

**Strong Classifier:**
- `h(x) = 1 if Σ(α_t * h_t(x)) ≥ 0.5 * Σ(α_t), else 0`

### 5. Cascade of Classifiers (cascade_classifier.py)

**Architecture:**
- Degenerate decision tree structure
- Each stage is an AdaBoost classifier
- Progressive complexity: each stage has more features

**Configuration:**
- Number of stages: 5
- Features per stage: [10, 20, 30, 40, 50]
- Target detection rate: 99.5% per stage
- Target false positive rate: 50% per stage

**Training Strategy:**
1. Train initial AdaBoost classifier
2. Adjust threshold to meet target detection rate on validation set
3. Collect false positives by running partial cascade
4. Use false positives as "hard negatives" for next stage
5. Repeat until all stages trained

**Detection Process:**
- All sub-windows evaluated by first classifier
- Rejected windows immediately discarded
- Passing windows proceed to next stage
- Final detection requires passing all stages

### 6. Face Detection (detect_faces.py)

**Multi-scale Detection:**
- Image pyramid with configurable scale factor (default: 1.25)
- Sliding window with configurable stride (default: 2 pixels)
- Detects faces at multiple scales

**Post-processing:**
- Non-maximum suppression to remove overlapping detections
- IoU threshold for merging (default: 0.3)

**Parameters:**
- `scale_factor`: Controls scale increments (1.2-1.5 recommended)
- `stride`: Window step size (smaller = more accurate but slower)
- `confidence_threshold`: Minimum confidence for detection
- `nms_threshold`: Overlap threshold for NMS

## Results

### Test Set Performance

**Overall Metrics:**
- Accuracy: [To be filled after testing]
- Precision: [To be filled after testing]
- Recall: [To be filled after testing]
- F1-Score: [To be filled after testing]

**Per-Stage Metrics:**

| Stage | Detection Rate | False Positive Rate | Samples Reaching |
|-------|---------------|---------------------|------------------|
| 1     | [TBF]         | [TBF]              | [TBF]            |
| 2     | [TBF]         | [TBF]              | [TBF]            |
| 3     | [TBF]         | [TBF]              | [TBF]            |
| 4     | [TBF]         | [TBF]              | [TBF]            |
| 5     | [TBF]         | [TBF]              | [TBF]            |

### Detection Examples

[Insert images with detected faces here]

**Example 1: [Image name]**
- Faces detected: [number]
- Detection time: [time]

**Example 2: [Image name]**
- Faces detected: [number]
- Detection time: [time]

## Performance Analysis

### Training Time
- Feature extraction: [To be filled]
- Cascade training: [To be filled]
- Total: [To be filled]

### Detection Speed
- Average time per 16×16 window: [To be filled]
- Average time per image: [To be filled]

### Memory Usage
- Feature matrix size: [To be filled]
- Model size: [To be filled]

## Key Implementation Decisions

1. **Feature Selection**: Generated all possible Haar features within constraints for completeness
2. **Training Data Balance**: Equal weight initialization for positive and negative classes
3. **Cascade Depth**: 5 stages to balance accuracy and speed
4. **Progressive Complexity**: Increasing features per stage (10→50) for efficiency
5. **Threshold Tuning**: Used validation set to adjust thresholds per stage

## Challenges and Solutions

### Challenge 1: Large Feature Space
- **Problem**: Over 100,000 features for 16×16 window
- **Solution**: AdaBoost automatically selects most discriminative features

### Challenge 2: Class Imbalance
- **Problem**: 5× more negative samples than positive
- **Solution**: Proper weight initialization in AdaBoost

### Challenge 3: Training Time
- **Problem**: Feature extraction for all samples is slow
- **Solution**: Cached feature matrices, used validation sampling

### Challenge 4: False Positives
- **Problem**: High false positive rate in single classifier
- **Solution**: Cascade architecture progressively filters negatives

## Comparison with Original Paper

| Aspect | Original Paper | Our Implementation |
|--------|---------------|-------------------|
| Window size | 24×24 | 16×16 |
| Features | 180,000+ | [TBF] |
| Cascade stages | 38 | 5 |
| Detection rate | 93.7% | [TBF] |
| False positive rate | 1 in 14,084 | [TBF] |

## Conclusions

### Strengths
1. Complete implementation of all components from scratch
2. Efficient integral image computation
3. Proper AdaBoost with feature selection
4. Cascade architecture for fast rejection
5. Multi-scale detection capability

### Limitations
1. Smaller window size (16×16 vs 24×24) may reduce accuracy
2. Fewer cascade stages (5 vs 38) may increase false positives
3. No specialized optimizations for real-time performance
4. Limited to frontal face detection

### Future Improvements
1. Increase window size to 24×24 for better features
2. Add more cascade stages for lower false positive rate
3. Implement rotation and scale invariance
4. Add profile face detection
5. Optimize with parallel processing
6. Implement hard negative mining during training

## Code Structure

```
viola-jones-detector/
├── dataset_generator.py       # Dataset creation
├── haar_features.py          # Haar feature extraction
├── integral_image.py         # Integral image computation
├── adaboost.py              # AdaBoost algorithm
├── cascade_classifier.py     # Cascade training & prediction
├── viola_jones_detector.py   # Main detector class
├── train.py                 # Training script
├── test.py                  # Testing script
├── detect_faces.py          # Face detection script
├── utils.py                 # Utility functions
└── requirements.txt         # Dependencies
```

## References

1. Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. CVPR.
2. Viola, P., & Jones, M. (2004). Robust real-time face detection. International Journal of Computer Vision.

---

**Note**: Fill in the bracketed placeholders [TBF] after running experiments.
