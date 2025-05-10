import cv2
import numpy as np
from preprocessing.preprocess_images import ImagePreprocessor
from feature_extraction.feature_extractor import FeatureExtractor
from models.model import GestureClassifier

def main():
    # Available models
    models = ['svm', 'random_forest', 'knn', 'decision_tree']
    
    # Print model options
    print("Available models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    # Get user choice
    while True:
        try:
            choice = int(input("\nSelect a model (1-4): "))
            if 1 <= choice <= 4:
                model_type = models[choice - 1]
                break
            else:
                print("Please enter a number between 1 and 4")
        except ValueError:
            print("Please enter a valid number")
    
    # Load the selected model
    print(f"\nLoading {model_type} model...")
    classifier = GestureClassifier(model_type=model_type)
    classifier.load_model(f'models/saved/{model_type}')
    print("Model loaded successfully!")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    preprocessor = ImagePreprocessor()
    extractor = FeatureExtractor()
    
    print("\nStarting real-time gesture recognition...")
    print("Press 'q' to quit")
    print("Press 'm' to switch models")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Preprocess the frame
        processed = preprocessor.preprocess_image(frame)
        
        if processed is not None:
            # Extract features
            features = extractor.extract_all_features(processed)
            
            # Make prediction
            prediction = classifier.predict([features])[0]
            
            # Map prediction to gesture name
            gesture_names = ['thumb', 'palm', 'fist']
            gesture_name = gesture_names[prediction]
            
            # Add prediction to display
            cv2.putText(display_frame, f"Model: {model_type}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Gesture: {gesture_name}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add action based on gesture
            if gesture_name == 'thumb':
                action = "Start Music"
            elif gesture_name == 'palm':
                action = "Stop Music"
            else:  # fist
                action = "Pause Music"
            
            cv2.putText(display_frame, f"Action: {action}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Gesture Recognition', display_frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            # Switch models
            cap.release()
            cv2.destroyAllWindows()
            main()  # Restart with model selection
            return
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 