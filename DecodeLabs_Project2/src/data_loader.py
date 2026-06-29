import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from typing import Tuple

def load_data() -> pd.DataFrame:
    """
    Loads the Titanic dataset using fetch_openml.
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    print("Loading Titanic dataset...")
    data = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
    df = data.frame
    # For titanic dataset, the target is 'survived'
    return df

def perform_eda(df: pd.DataFrame, images_dir: str = '../images') -> None:
    """
    Performs Exploratory Data Analysis and saves plots to the images directory.
    
    Args:
        df (pd.DataFrame): The dataset.
        images_dir (str): Path to the directory where images will be saved.
    """
    print("Performing Data Exploration...")
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"Shape of the dataset: {df.shape}")
    print("\nColumns in the dataset:")
    print(df.columns.tolist())
    
    print("\nDatatype Analysis:")
    print(df.dtypes)
    
    print("\nMissing Value Analysis:")
    print(df.isnull().sum())
    
    print("\nDuplicate Analysis:")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    
    print("\nStatistical Summary:")
    print(df.describe(include='all'))
    
    print("\nTarget Distribution (Class Balance):")
    # Titanic target is 'survived'
    if 'survived' in df.columns:
        print(df['survived'].value_counts())
        
        # Plot Target Distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='survived')
        plt.title('Target Distribution (Survived)')
        plt.savefig(os.path.join(images_dir, 'target_distribution.png'))
        plt.close()
    
    # Correlation Heatmap for numeric features
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.savefig(os.path.join(images_dir, 'correlation_heatmap.png'))
        plt.close()

if __name__ == "__main__":
    df = load_data()
    perform_eda(df, images_dir='../images')
