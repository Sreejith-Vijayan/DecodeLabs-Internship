# AI Classification Project Guide

This guide provides step-by-step instructions on how to set up, run, and understand the outputs of the End-to-End AI Classification Project.

## Prerequisites
- Python 3.12 or higher.
- `pip` (Python package installer).

## Step 1: Installation

First, clone this repository (if you haven't already), and navigate to the project folder:
```bash
git clone <repository_url>
cd DecodeLabs-Internship/DecodeLabs_Project2
```

Install the required Python packages using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Step 2: Running the Complete Pipeline

The easiest way to execute the entire project from start to finish is by running `main.py`. This script acts as an orchestrator that calls all the individual modules in the correct sequence.

```bash
python main.py
```

### What happens when you run `main.py`?
1. **Data Loading**: Fetches the Titanic dataset from OpenML and saves a local copy in the `data/` folder as `titanic.csv`.
2. **Exploratory Data Analysis (EDA)**: Analyzes the data (missing values, statistics) and generates plots (Target Distribution, Correlation Heatmap). These are saved in the `images/` directory.
3. **Preprocessing**: Cleans the data by imputing missing values, one-hot encoding categorical features, and scaling numerical features.
4. **Training & Tuning**: Trains 6 different machine learning models, applies Cross-Validation, and uses `GridSearchCV` to find the best hyperparameters for a Random Forest classifier.
5. **Evaluation**: Evaluates the best model on a test set, generating a Classification Report (saved to `reports/classification_report.txt`), a Confusion Matrix, Feature Importance, and an ROC curve (saved to `images/`).
6. **Prediction Simulation**: Creates sample unseen data and runs the prediction module to demonstrate inference.

## Step 3: Running Modules Independently

You can also run specific scripts independently if you want to focus on a particular stage.

### Exploring the Data
If you only want to load the data and generate the EDA plots:
```bash
python -m src.data_loader
```

### Predicting on New Data
Once the model has been trained and saved (which happens after running `main.py`), you can use the prediction script to classify new passengers.
```bash
python -m src.predict
```
*Note: You can open `src/predict.py` and modify the `sample_dict` at the bottom of the file to test the model with different passenger details!*

## Understanding the Generated Artifacts

- **`data/titanic.csv`**: The dataset used for training the models.
- **`models/best_model.pkl`**: The trained, hyperparameter-tuned model saved for later use (Note: this folder is not tracked by Git to save space/security).
- **`models/preprocessor.pkl`**: The pipeline responsible for cleaning and transforming raw input data before it's passed to the model.
- **`reports/classification_report.txt`**: A detailed breakdown of Precision, Recall, and F1-scores for both the "Survived" and "Did Not Survive" classes.
- **`images/*.png`**: Visualizations including the correlation between different features, the performance of the model (ROC curve), and which features were most important to the model's decisions.
