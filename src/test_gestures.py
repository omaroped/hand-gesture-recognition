import cv2
import numpy as np
from collections import deque
from preprocessing.preprocess_images import ImagePreprocessor
from feature_extraction.feature_extractor import FeatureExtractor
from models.model import GestureClassifier

class GestureStateMachine:
    def __init__(self, min_frames=10, cooldown_frames=5):
        self.min_frames = min_frames  # Minimum frames to confirm a gesture
        self.cooldown_frames = cooldown_frames  # Frames to wait before accepting new gesture
        self.current_gesture = None
        self.gesture_frames = 0
        self.cooldown_counter = 0
        self.gesture_history = deque(maxlen=min_frames)
    
    def update(self, new_gesture, confidence):
        if confidence < 0.8:  # Confidence threshold
            return self.current_gesture
        
        self.gesture_history.append(new_gesture)
        
        # Count most common gesture in history
        if len(self.gesture_history) >= self.min_frames:
            most_common = max(set(self.gesture_history), key=self.gesture_history.count)
            count = self.gesture_history.count(most_common)
            
            if count >= self.min_frames * 0.8:  # 80% of frames must agree
                if most_common != self.current_gesture:
                    if self.cooldown_counter <= 0:
                        self.current_gesture = most_common
                        self.cooldown_counter = self.cooldown_frames
                    else:
                        self.cooldown_counter -= 1
                else:
                    self.cooldown_counter = self.cooldown_frames
        
        return self.current_gesture

def detect_motion(prev_frame, current_frame, threshold=30):
    if prev_frame is None:
        return True
    
    # Calculate absolute difference between frames
    diff = cv2.absdiff(prev_frame, current_frame)
    mean_diff = np.mean(diff)
    
    return mean_diff > threshold

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
    
    # Initialize components
    cap = cv2.VideoCapture(0)
    preprocessor = ImagePreprocessor()
    extractor = FeatureExtractor()
    state_machine = GestureStateMachine()
    
    # Motion detection variables
    prev_frame = None
    
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
        
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check for significant motion
        if not detect_motion(prev_frame, gray):
            # If no significant motion, keep previous prediction
            if state_machine.current_gesture is not None:
                gesture_name = state_machine.current_gesture
            else:
                gesture_name = "No motion"
        else:
            # Preprocess the frame
            processed = preprocessor.preprocess_image(frame)
            
            if processed is not None:
                # Extract features
                features = extractor.extract_all_features(processed)
                
                # Get prediction and probability
                prediction = classifier.predict([features])[0]
                probabilities = classifier.model.predict_proba(classifier.scaler.transform([features]))[0]
                confidence = np.max(probabilities)
                
                # Map prediction to gesture name
                gesture_names = ['thumb', 'palm', 'fist']
                gesture_name = gesture_names[prediction]
                
                # Update state machine
                gesture_name = state_machine.update(gesture_name, confidence)
        
        # Update previous frame
        prev_frame = gray
        
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
        elif gesture_name == 'fist':
            action = "Pause Music"
        else:
            action = "No Action"
        
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