# Hand Gesture Recognition for Music Box Control

This project implements a machine learning system for recognizing hand gestures to control a music box. The system can recognize three distinct gestures:
- Thumb gesture: Start music
- Palm gesture: Stop music
- Fist gesture: Pause music

## Project Structure
```
hand_gesture_recognition/
├── data/
│   ├── raw/                 # Original captured images
│   ├── processed/           # Preprocessed images
│   └── augmented/           # Augmented dataset
├── src/
│   ├── data_collection/     # Scripts for capturing images
│   ├── preprocessing/       # Image preprocessing modules
│   ├── feature_extraction/  # Feature extraction modules
│   ├── models/             # ML model implementations
│   └── utils/              # Utility functions
├── notebooks/              # Jupyter notebooks for analysis
├── tests/                 # Unit tests
└── results/               # Model results and visualizations
```

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Data Collection:
```bash
python src/data_collection/capture_images.py
```

2. Preprocessing:
```bash
python src/preprocessing/preprocess_images.py
```

3. Model Training:
```bash
python src/models/train_model.py
```

4. Evaluation:
```bash
python src/models/evaluate_model.py
```

## Features
- Hand gesture recognition using traditional ML approaches (SVM or FFNN)
- Comprehensive data preprocessing pipeline
- Feature extraction using HOG, shape features, and statistical measures
- Model evaluation and visualization tools
- Real-time gesture recognition capability

## Model Performance
The system includes multiple models with their accuracies:
1. SVM: 97% accuracy
2. Random Forest: 95% accuracy
3. KNN: 94% accuracy
4. Decision Tree: 89% accuracy

## Version History

### v1.0.0 (Initial Release)
- Basic gesture recognition with white background
- Support for multiple ML models
- Real-time recognition capability
- Debug mode for confidence scores

### v1.1.0 (In Development)
- Improved background handling
- Hand detection and segmentation
- Better real-world environment support
- Enhanced visualization and feedback

## Requirements
- Python 3.8+
- OpenCV
- scikit-learn
- TensorFlow
- Other dependencies listed in requirements.txt

## License
MIT License
