import cv2
import numpy as np
import os
from skimage import transform
from skimage.util import random_noise
import random

class ImagePreprocessor:
    def __init__(self, input_dir='data/raw', output_dir='data/processed'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = (128, 128)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        for gesture in ['thumb', 'palm', 'fist']:
            os.makedirs(os.path.join(output_dir, gesture), exist_ok=True)
    
    def preprocess_image(self, image):
        """Apply basic preprocessing to an image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Find the largest contour (assumed to be the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Extract and resize the hand region
        hand_region = gray[y:y+h, x:x+w]
        resized = cv2.resize(hand_region, self.target_size)
        
        # Normalize pixel values
        normalized = resized / 255.0
        
        return normalized
    
    def augment_image(self, image):
        """Apply data augmentation to an image."""
        augmented_images = []
        
        # Original image
        augmented_images.append(image)
        
        # Rotation
        for angle in [-15, 15]:
            rotated = transform.rotate(image, angle, mode='edge')
            augmented_images.append(rotated)
        
        # Horizontal flip
        flipped = np.fliplr(image)
        augmented_images.append(flipped)
        
        # Brightness adjustment
        bright = np.clip(image * 1.2, 0, 1)
        dark = np.clip(image * 0.8, 0, 1)
        augmented_images.extend([bright, dark])
        
        # Add noise
        noisy = random_noise(image, mode='gaussian', var=0.01)
        augmented_images.append(noisy)
        
        return augmented_images
    
    def process_dataset(self):
        """Process all images in the dataset."""
        for gesture in ['thumb', 'palm', 'fist']:
            gesture_dir = os.path.join(self.input_dir, gesture)
            output_gesture_dir = os.path.join(self.output_dir, gesture)
            
            for filename in os.listdir(gesture_dir):
                if not filename.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                # Read image
                image_path = os.path.join(gesture_dir, filename)
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"Failed to read image: {image_path}")
                    continue
                
                # Preprocess image
                processed = self.preprocess_image(image)
                if processed is None:
                    print(f"Failed to preprocess image: {image_path}")
                    continue
                
                # Save processed image
                base_name = os.path.splitext(filename)[0]
                processed_path = os.path.join(output_gesture_dir, f"{base_name}_processed.jpg")
                cv2.imwrite(processed_path, (processed * 255).astype(np.uint8))
                
                # Generate and save augmented images
                augmented_images = self.augment_image(processed)
                for i, aug_img in enumerate(augmented_images):
                    aug_path = os.path.join(output_gesture_dir, f"{base_name}_aug_{i}.jpg")
                    cv2.imwrite(aug_path, (aug_img * 255).astype(np.uint8))

if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    preprocessor.process_dataset() 