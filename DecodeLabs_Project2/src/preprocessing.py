import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Preprocesses the dataset:
    - Handles missing values
    - Encodes categorical features
    - Scales numerical features
    - Splits into train and test sets
    
    Args:
        df (pd.DataFrame): The raw dataset.
        
    Returns:
        Tuple containing X_train, X_test, y_train, y_test, and the fitted preprocessor.
    """
    print("Preprocessing Data...")
    
    # Define target and features
    target = 'survived'
    # Drop columns that are mostly unique or not useful for basic modeling
    cols_to_drop = ['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Drop rows where target is missing
    df = df.dropna(subset=[target])
    
    y = df[target].astype(int) # Ensure target is integer
    X = df.drop(columns=[target])
    
    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Preprocessing for numerical data: Impute missing values with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical data: Impute missing with most frequent, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Train-Test Split (test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit and transform training data, transform testing data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Convert processed data back to DataFrame/Series for easier handling later
    # Get feature names after one-hot encoding
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_feature_names)
    
    X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_processed, columns=all_feature_names, index=X_test.index)
    
    print(f"Training data shape: {X_train_df.shape}")
    print(f"Testing data shape: {X_test_df.shape}")
    
    return X_train_df, X_test_df, y_train, y_test, preprocessor
