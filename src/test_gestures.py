import cv2
import numpy as np
from preprocessing.preprocess_images import ImagePreprocessor
from feature_extraction.feature_extractor import FeatureExtractor
from models.model import GestureClassifier

def detect_hand(frame):
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    
    # Define skin color range in YCrCb
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([255, 173, 127], np.uint8)
    
    # Create a binary mask for skin color
    skin_mask = cv2.inRange(ycrcb, min_YCrCb, max_YCrCb)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    
    # Apply Gaussian blur to smooth the mask
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # Find the largest contour (assumed to be the hand)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding box of the hand
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add some padding around the hand
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame.shape[1] - x, w + 2 * padding)
    h = min(frame.shape[0] - y, h + 2 * padding)
    
    # Extract the hand region
    hand_region = frame[y:y+h, x:x+w]
    
    # Create a mask for the hand region
    hand_mask = skin_mask[y:y+h, x:x+w]
    
    # Create a white background
    white_bg = np.ones_like(hand_region) * 255
    
    # Use the mask to combine the hand with white background
    hand_segmented = cv2.bitwise_and(hand_region, hand_region, mask=hand_mask)
    hand_segmented = cv2.bitwise_or(hand_segmented, white_bg, mask=cv2.bitwise_not(hand_mask))
    
    return hand_segmented, (x, y, w, h)

def main():
    # Available models with their accuracies
    models = {
        '1': ('svm', 'SVM (97% accuracy)'),
        '2': ('random_forest', 'Random Forest (95% accuracy)'),
        '3': ('knn', 'KNN (94% accuracy)'),
        '4': ('decision_tree', 'Decision Tree (89% accuracy)')
    }
    
    # Print model options
    print("\nAvailable models:")
    for key, (model_type, description) in models.items():
        print(f"{key}. {description}")
    
    # Get user choice
    while True:
        choice = input("\nSelect a model (1-4): ")
        if choice in models:
            model_type, _ = models[choice]
            break
        else:
            print("Please enter a number between 1 and 4")
    
    # Load the selected model
    print(f"\nLoading {model_type} model...")
    classifier = GestureClassifier(model_type=model_type)
    classifier.load_model(f'models/saved/{model_type}')
    print("Model loaded successfully!")
    
    # Initialize components
    cap = cv2.VideoCapture(0)
    preprocessor = ImagePreprocessor()
    extractor = FeatureExtractor()
    
    # For smoothing predictions
    prediction_history = []
    history_size = 5
    
    print("\nStarting real-time gesture recognition...")
    print("Press 'q' to quit")
    print("Press 'm' to switch models")
    print("Press 'd' to toggle debug mode")
    
    debug_mode = True  # Start with debug mode on
    
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
        
        # Detect and segment hand
        hand_segmented, hand_bbox = detect_hand(frame)
        
        if hand_segmented is not None:
            # Draw rectangle around detected hand
            x, y, w, h = hand_bbox
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Preprocess the segmented hand
            processed = preprocessor.preprocess_image(hand_segmented)
            
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
                
                # Add to history
                prediction_history.append(gesture_name)
                if len(prediction_history) > history_size:
                    prediction_history.pop(0)
                
                # Get most common prediction
                if prediction_history:
                    gesture_name = max(set(prediction_history), key=prediction_history.count)
                
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
                
                # Debug information
                if debug_mode:
                    # Show confidence for each gesture
                    y_pos = 150
                    for i, (name, prob) in enumerate(zip(gesture_names, probabilities)):
                        color = (0, 255, 0) if name == gesture_name else (0, 0, 255)
                        cv2.putText(display_frame, f"{name}: {prob:.2f}", (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        y_pos += 30
                    
                    # Show prediction history
                    history_text = "History: " + " -> ".join(prediction_history)
                    cv2.putText(display_frame, history_text, (10, y_pos + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Show segmented hand
                    if hand_segmented is not None:
                        # Resize for display
                        display_size = (200, 200)
                        hand_display = cv2.resize(hand_segmented, display_size)
                        # Place in top-right corner
                        display_frame[10:10+display_size[1], 
                                    display_frame.shape[1]-10-display_size[0]:display_frame.shape[1]-10] = hand_display
        
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
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()