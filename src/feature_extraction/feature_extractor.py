import cv2
import numpy as np
from skimage.feature import hog
from skimage import measure

class FeatureExtractor:
    def __init__(self):
        self.hog_orientations = 9
        self.hog_pixels_per_cell = (8, 8)
        self.hog_cells_per_block = (2, 2)
    
    def extract_hog_features(self, image):
        """Extract HOG features from the image."""
        features = hog(
            image,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            visualize=False
        )
        return features
    
    def extract_shape_features(self, image):
        """Extract shape-based features from the image."""
        # Convert to binary image
        binary = (image > 0.5).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return np.zeros(7)  # Return zeros if no contours found
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate shape features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Calculate convex hull and its area
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        
        # Calculate solidity
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Calculate aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Calculate Hu moments
        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Combine all shape features
        shape_features = np.array([
            area,
            perimeter,
            solidity,
            aspect_ratio,
            hu_moments[0],
            hu_moments[1],
            hu_moments[2]
        ])
        
        return shape_features
    
    def extract_statistical_features(self, image):
        """Extract statistical features from the image."""
        # Divide image into 4x4 grid
        h, w = image.shape
        cell_h, cell_w = h // 4, w // 4
        
        features = []
        for i in range(4):
            for j in range(4):
                cell = image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                features.extend([
                    np.mean(cell),
                    np.std(cell),
                    np.max(cell),
                    np.min(cell)
                ])
        
        return np.array(features)
    
    def extract_all_features(self, image):
        """Extract all features from the image."""
        hog_features = self.extract_hog_features(image)
        shape_features = self.extract_shape_features(image)
        statistical_features = self.extract_statistical_features(image)
        
        # Combine all features
        all_features = np.concatenate([
            hog_features,
            shape_features,
            statistical_features
        ])
        
        return all_features
    
    def normalize_features(self, features):
        """Normalize features to zero mean and unit variance."""
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        normalized = (features - mean) / std
        return normalized

if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor()
    
    # Load a sample image
    image = cv2.imread('data/processed/thumb/sample.jpg', cv2.IMREAD_GRAYSCALE)
    if image is not None:
        # Normalize image
        image = image / 255.0
        
        # Extract features
        features = extractor.extract_all_features(image)
        print(f"Total number of features: {len(features)}")
        print(f"Feature vector: {features}") 