import os
import joblib
import pandas as pd
from typing import Dict, Any

def predict(sample_data: pd.DataFrame, model_path: str = '../models/best_model.pkl', preprocessor_path: str = '../models/preprocessor.pkl'):
    """
    Loads the saved model and preprocessor, and predicts the outcome for the given sample data.
    """
    print("\n--- Running Prediction Module ---")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
        
    # Load model and preprocessor
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Preprocess the sample data
    # (assuming sample_data has the same columns as the original training data before dropping target)
    try:
        processed_data = preprocessor.transform(sample_data)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None
        
    # Predict
    prediction = model.predict(processed_data)
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(processed_data)
        for i, (pred, prob) in enumerate(zip(prediction, probabilities)):
            class_name = "Survived" if pred == 1 else "Did Not Survive"
            print(f"Sample {i+1}: Prediction = {class_name}")
            print(f"          Probabilities: [Did Not Survive: {prob[0]:.4f}, Survived: {prob[1]:.4f}]")
    else:
        for i, pred in enumerate(prediction):
            class_name = "Survived" if pred == 1 else "Did Not Survive"
            print(f"Sample {i+1}: Prediction = {class_name}")
            
    return prediction
    
if __name__ == "__main__":
    # Example sample input
    sample_dict = {
        'pclass': [3, 1],
        'sex': ['male', 'female'],
        'age': [22.0, 38.0],
        'sibsp': [1, 1],
        'parch': [0, 0],
        'fare': [7.25, 71.2833],
        'embarked': ['S', 'C']
    }
    sample_df = pd.DataFrame(sample_dict)
    
    # Needs to be run from project root to resolve relative paths correctly or adjust paths
    try:
        predict(sample_df, model_path='models/best_model.pkl', preprocessor_path='models/preprocessor.pkl')
    except Exception as e:
        print(e)
