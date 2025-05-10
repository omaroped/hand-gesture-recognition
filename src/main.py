import os
import cv2
import numpy as np
from data_collection.capture_images import HandGestureCapture
from preprocessing.preprocess_images import ImagePreprocessor
from feature_extraction.feature_extractor import FeatureExtractor
from models.model import GestureClassifier

def main():
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models/saved', exist_ok=True)
    
    # Check if we need to collect data
    if not os.path.exists('data/raw/thumb') or not os.listdir('data/raw/thumb'):
        print("No training data found. Starting data collection...")
        # Step 1: Data Collection
        print("\nStep 1: Data Collection")
        print("Press 'c' to capture an image, 'q' to quit")
        capture = HandGestureCapture()
        capture.run()
    
    # Step 2: Preprocessing
    print("\nStep 2: Preprocessing")
    preprocessor = ImagePreprocessor()
    preprocessor.process_dataset()
    
    # Step 3: Feature Extraction
    print("\nStep 3: Feature Extraction")
    extractor = FeatureExtractor()
    
    # Load and process all images
    X = []  # Features
    y = []  # Labels
    
    for gesture_idx, gesture in enumerate(['thumb', 'palm', 'fist']):
        gesture_dir = os.path.join('data/processed', gesture)
        for filename in os.listdir(gesture_dir):
            if not filename.endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            # Load image
            image_path = os.path.join(gesture_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue
            
            # Normalize image
            image = image / 255.0
            
            # Extract features
            features = extractor.extract_all_features(image)
            X.append(features)
            y.append(gesture_idx)
    
    X = np.array(X)
    y = np.array(y)
    
    # Step 4: Model Training
    print("\nStep 4: Model Training")
    
    # Train different models
    models = ['svm', 'random_forest', 'knn', 'decision_tree']
    best_model = None
    best_accuracy = 0
    
    for model_type in models:
        print(f"\nTraining {model_type} model...")
        classifier = GestureClassifier(model_type=model_type)
        classifier.train(X, y)
        
        # Save the model
        model_path = f'models/saved/{model_type}'
        classifier.save_model(model_path)
        
        # For testing, we'll use the first model as default
        if best_model is None:
            best_model = model_type
    
    print(f"\nAll models trained and saved. Using {best_model} as default model.")

if __name__ == "__main__":
    main() 