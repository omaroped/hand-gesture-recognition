import cv2
import os
import numpy as np
from datetime import datetime

class HandGestureCapture:
    def __init__(self, output_dir='data/raw'):
        self.output_dir = output_dir
        self.gestures = ['thumb', 'palm', 'fist']
        self.cap = cv2.VideoCapture(0)
        
        # Create output directories for each gesture
        for gesture in self.gestures:
            os.makedirs(os.path.join(output_dir, gesture), exist_ok=True)
    
    def capture_images(self, gesture_name, num_images=50):
        """Capture images for a specific gesture."""
        if gesture_name not in self.gestures:
            raise ValueError(f"Gesture must be one of {self.gestures}")
        
        print(f"Capturing {num_images} images for {gesture_name} gesture...")
        print("Press 'c' to capture an image, 'q' to quit")
        
        count = 0
        while count < num_images:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Add text overlay
            cv2.putText(frame, f"Capturing: {gesture_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Images captured: {count}/{num_images}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Hand Gesture Capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Save the image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{gesture_name}_{timestamp}_{count}.jpg"
                filepath = os.path.join(self.output_dir, gesture_name, filename)
                cv2.imwrite(filepath, frame)
                count += 1
                print(f"Captured image {count}/{num_images}")
            elif key == ord('q'):
                break
        
        print(f"Finished capturing {count} images for {gesture_name}")
    
    def run(self):
        """Run the capture process for all gestures."""
        try:
            for gesture in self.gestures:
                self.capture_images(gesture)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Initialize and run the capture process
    capture = HandGestureCapture()
    capture.run() 