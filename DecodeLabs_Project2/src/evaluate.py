import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, reports_dir: str = '../reports', images_dir: str = '../images'):
    """
    Evaluates the trained model on the test dataset.
    Generates metrics, confusion matrix, ROC curve, and classification report.
    """
    print("Evaluating Model...")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Classification Report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:\n", report)
    with open(os.path.join(reports_dir, 'classification_report.txt'), 'w') as f:
        f.write("Classification Report\n")
        f.write("=====================\n\n")
        f.write(report)
        
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(images_dir, 'confusion_matrix.png'))
    plt.close()
    
    # ROC Curve Plot (if probabilities are available)
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(images_dir, 'roc_curve.png'))
        plt.close()
        
    # Feature Importance (if applicable, e.g., for Random Forest)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = X_test.columns
        feature_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        feature_df = feature_df.sort_values(by='Importance', ascending=False).head(15) # Top 15
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_df)
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'feature_importance.png'))
        plt.close()
        print("Feature importance plot saved.")
