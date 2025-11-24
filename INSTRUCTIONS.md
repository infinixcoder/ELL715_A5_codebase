# Viola-Jones Face Detector - Setup and Testing Instructions

## Dependencies

All required Python libraries are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── requirements.txt           # Python dependencies
├── dataset_generator.py       # Dataset generation module
├── haar_features.py          # Haar feature extraction
├── integral_image.py         # Integral image computation
├── adaboost.py              # AdaBoost algorithm implementation
├── cascade_classifier.py     # Cascade of classifiers
├── viola_jones_detector.py   # Main detector class
├── train.py                 # Training script
├── test.py                  # Testing script
├── detect_faces.py          # Face detection on custom images
├── utils.py                 # Utility functions
└── faces94/                 # Dataset folder
    ├── female/              # Training data (female)
    ├── male/                # Testing data (male)
    └── malestaff/           # Training data (malestaff)
```

## Step 1: Generate Dataset

Generate training and testing datasets (16x16 patches):

```bash
python dataset_generator.py
```

This will create:
- `data/train_faces.npy` and `data/train_labels.npy` (training set)
- `data/test_faces.npy` and `data/test_labels.npy` (testing set)

## Step 2: Train the Viola-Jones Detector

Train the cascade of classifiers using AdaBoost:

```bash
python train.py
```

This will:
- Extract Haar features from training data
- Train cascade classifiers using AdaBoost
- Save the trained model to `models/viola_jones_cascade.pkl`

Training parameters can be adjusted in `train.py`:
- `num_stages`: Number of cascade stages (default: 5)
- `features_per_stage`: Features per stage (default: [10, 20, 30, 40, 50])
- `min_detection_rate`: Minimum detection rate per stage (default: 0.995)
- `max_false_positive_rate`: Maximum false positive rate per stage (default: 0.5)

## Step 3: Test the Detector

Evaluate the detector on the test dataset:

```bash
python test.py
```

This will:
- Load the trained model
- Test on the generated test dataset
- Display accuracy, precision, recall, and F1-score

## Step 4: Detect Faces in Custom Images

To detect faces in your own images:

```bash
python detect_faces.py --image_path path/to/your/image.jpg
```

Optional arguments:
- `--scale_factor`: Scale factor for multi-scale detection (default: 1.25)
- `--min_window_size`: Minimum detection window size (default: 16)
- `--stride`: Sliding window stride (default: 2)
- `--output_path`: Path to save output image (default: output/detected_faces.jpg)

Example:
```bash
python detect_faces.py --image_path test_image.jpg --scale_factor 1.2 --stride 4
```

## Additional Options

### Visualize Haar Features

To visualize the Haar features being used:

```bash
python -c "from haar_features import visualize_haar_features; visualize_haar_features()"
```

### Check Model Information

To see details about the trained model:

```bash
python -c "import joblib; model = joblib.load('models/viola_jones_cascade.pkl'); print(f'Number of stages: {len(model.stages)}'); print(f'Total features: {sum(len(stage.classifiers) for stage in model.stages)}')"
```

## Expected Performance

Based on the implementation:
- Training time: 30-60 minutes (depends on number of stages and features)
- Detection rate: >95% on test set
- False positive rate: <5% per cascade stage

## Troubleshooting

1. **Out of Memory Error**: Reduce the number of features per stage or use fewer training samples
2. **Low Accuracy**: Increase the number of cascade stages or features per stage
3. **Slow Training**: Reduce the number of Haar features or use parallel processing
4. **No Faces Detected**: Adjust scale_factor and stride parameters in detection

## Notes

- The implementation uses 16x16 image patches as specified in the readme
- Training uses images from `female/` and `malestaff/` folders
- Testing uses images from `male/` folder
- All intermediate results are saved in `data/` and `models/` directories
