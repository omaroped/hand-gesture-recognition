import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

class GestureClassifier:
    def __init__(self, model_type='svm'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.classes = ['thumb', 'palm', 'fist']
        
        # Initialize the selected model
        if model_type == 'svm':
            self.model = SVC(probability=True)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=5)
        elif model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(random_state=42)
        else:
            raise ValueError("Model type must be one of: 'svm', 'random_forest', 'knn', 'decision_tree'")
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """Train the model on the provided data."""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if self.model_type == 'svm':
            # Define parameter grid for SVM
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1],
                'kernel': ['rbf']
            }
        elif self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        elif self.model_type == 'knn':
            param_grid = {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        elif self.model_type == 'decision_tree':
            param_grid = {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Print results
        print(f"\nBest parameters for {self.model_type}:", grid_search.best_params_)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.classes))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    def predict(self, X):
        """Make predictions on new data."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save_model(self, model_path):
        """Save the trained model."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, f"{model_path}_model.joblib")
        joblib.dump(self.scaler, f"{model_path}_scaler.joblib")
    
    def load_model(self, model_path):
        """Load a trained model."""
        self.model = joblib.load(f"{model_path}_model.joblib")
        self.scaler = joblib.load(f"{model_path}_scaler.joblib")

if __name__ == "__main__":
    # Example usage
    # Load and prepare your data here
    # X = ...  # Your feature matrix
    # y = ...  # Your labels
    
    # Create and train different models
    models = ['svm', 'random_forest', 'knn', 'decision_tree']
    for model_type in models:
        print(f"\nTraining {model_type} model...")
        classifier = GestureClassifier(model_type=model_type)
        # classifier.train(X, y) 