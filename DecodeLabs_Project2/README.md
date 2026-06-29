# End-to-End AI Classification Project

A complete, beginner-friendly, and professional AI classification pipeline built in Python. This project demonstrates best practices in data loading, preprocessing, model training, hyperparameter tuning, evaluation, and prediction.

**Project Overview**: We use the **Titanic Dataset** to predict passenger survival based on various features (like age, sex, passenger class). It implements an end-to-end Machine Learning pipeline with multiple classifiers and automatically selects the best-performing model.

## Features

- **Automated Data Fetching**: Uses `sklearn.datasets.fetch_openml` to directly download the dataset.
- **Robust Preprocessing**: Handles missing values, performs one-hot encoding on categorical data, and scales numerical data using `scikit-learn` pipelines.
- **Comprehensive EDA**: Generates statistical summaries and various plots (correlation heatmap, target distribution).
- **Multiple Classifiers**: Trains and compares Logistic Regression, Decision Tree, Random Forest, KNN, SVM, and Naive Bayes.
- **Hyperparameter Tuning**: Utilizes `GridSearchCV` to optimize the Random Forest model with 5-fold cross-validation.
- **Detailed Evaluation**: Outputs Accuracy, Precision, Recall, F1 Score, Classification Report, Confusion Matrix, and ROC Curve.
- **Independent Prediction Module**: A standalone prediction script that loads the saved preprocessor and model to classify new unseen data.

## Folder Structure

```text
AI-Classification-Project/
│
├── data/                  # Directory for raw/processed data (if saved locally)
├── notebooks/             # Directory for Jupyter notebooks (EDA)
├── src/                   # Source code for the pipeline
│   ├── __init__.py
│   ├── data_loader.py     # Data fetching and EDA plotting
│   ├── preprocessing.py   # Data cleaning, encoding, scaling, splitting
│   ├── train.py           # Model training, evaluation, tuning, saving
│   ├── evaluate.py        # Model evaluation and metrics generation
│   ├── predict.py         # Prediction module for unseen data
│
├── models/                # Saved models (best_model.pkl, preprocessor.pkl)
├── reports/               # Text reports (e.g., classification report)
├── images/                # Visualizations (Heatmap, ROC, Confusion Matrix)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── main.py                # Main orchestration script
```

## Installation & Dependencies

Requires Python 3.12+.

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd DecodeLabs_Project2
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Dependencies included: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`.

## Dataset Information

- **Name**: Titanic Dataset (OpenML ID: 40945)
- **Problem**: Binary Classification (Survived: 1 or 0)
- **Features**: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked (Note: Name, Ticket, Cabin, etc. are dropped).

## Training Instructions

To run the entire pipeline (Data Loading -> Preprocessing -> Training -> Evaluation -> Prediction Simulation):

```bash
python main.py
```

This will output the progress to the console and save the respective artifacts in the `models/`, `reports/`, and `images/` directories.

## Prediction Instructions

To predict on new data using the saved model:

1. Ensure you have run `main.py` at least once so that `models/best_model.pkl` and `models/preprocessor.pkl` are generated.
2. Run the standalone prediction module:
   ```bash
   python -m src.predict
   ```

You can modify the `sample_dict` inside `src/predict.py` to test different inputs.

## Sample Outputs

* **Console**: CV Accuracy for all models, best hyperparameters for Random Forest, and final metrics on the Test set.
* **Reports**: `reports/classification_report.txt` contains Precision, Recall, and F1 per class.
* **Images**: `images/correlation_heatmap.png`, `images/confusion_matrix.png`, `images/roc_curve.png`, `images/feature_importance.png`.

## Future Improvements

- Integrate deep learning models using PyTorch or TensorFlow.
- Create an API using FastAPI or Flask to serve the prediction model.
- Add MLflow for robust experiment tracking.
- Create a Streamlit web app for an interactive UI.

---
*Resume-worthy project description*: Developed a complete modular AI Classification pipeline predicting Titanic survival with robust preprocessing, multiple classifier comparison, hyperparameter tuning via GridSearchCV, and automated model evaluation and saving.
