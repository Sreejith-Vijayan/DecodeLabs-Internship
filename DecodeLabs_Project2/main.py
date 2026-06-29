import os
import joblib
import pandas as pd
from src.data_loader import load_data, perform_eda
from src.preprocessing import preprocess_data
from src.train import train_pipeline
from src.evaluate import evaluate_model
from src.predict import predict

def main():
    print("Starting AI Classification Project Pipeline\n" + "="*40)
    
    # 1. Load Data
    df = load_data(data_dir='data')
    
    # 2. EDA
    perform_eda(df, images_dir='images')
    
    # 3. Preprocessing
    # We pass the dataframe, get processed splits
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Save the preprocessor for future inference
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("Preprocessor saved to models/preprocessor.pkl")
    
    # 4. Training
    best_model = train_pipeline(X_train, y_train, models_dir='models')
    
    # 5. Evaluation
    evaluate_model(best_model, X_test, y_test, reports_dir='reports', images_dir='images')
    
    # 6. Prediction on new sample
    print("\nSimulating Prediction on New Unseen Data...")
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
    predict(sample_df, model_path='models/best_model.pkl', preprocessor_path='models/preprocessor.pkl')
    
    print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    main()
