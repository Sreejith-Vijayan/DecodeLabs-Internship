import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from typing import Dict, Any

def train_and_evaluate_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """
    Trains multiple models, evaluates them using Cross Validation, 
    and returns a dictionary of fitted models and their CV scores.
    """
    print("Training models...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=42),
        "Naive Bayes": GaussianNB()
    }
    
    results = {}
    
    for name, model in models.items():
        # Perform 5-fold cross validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mean_cv_score = cv_scores.mean()
        print(f"{name} - Mean CV Accuracy: {mean_cv_score:.4f}")
        
        # Fit model on the full training set
        model.fit(X_train, y_train)
        results[name] = {'model': model, 'score': mean_cv_score}
        
    return results

def tune_best_model(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Performs Hyperparameter Tuning on RandomForest as an example of GridSearchCV.
    """
    print("Tuning Random Forest...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters for Random Forest: {grid_search.best_params_}")
    print(f"Best CV Accuracy for Random Forest: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_score_

def train_pipeline(X_train: pd.DataFrame, y_train: pd.Series, models_dir: str = '../models'):
    """
    Executes the training pipeline and saves the best model.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Train basic models
    results = train_and_evaluate_models(X_train, y_train)
    
    # Tune Random Forest
    tuned_rf, tuned_score = tune_best_model(X_train, y_train)
    
    # Add tuned model to results
    results["Tuned Random Forest"] = {'model': tuned_rf, 'score': tuned_score}
    
    # Select the best model overall
    best_model_name = max(results, key=lambda k: results[k]['score'])
    best_model = results[best_model_name]['model']
    best_score = results[best_model_name]['score']
    
    print(f"\nBest Model Selected: {best_model_name} with CV Accuracy: {best_score:.4f}")
    
    # Save the best model
    model_path = os.path.join(models_dir, 'best_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")
    
    return best_model
