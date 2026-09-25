import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier


def train(X_train, y_train, X_val, y_val):
    """
    Design your own tabular binary classifier on real census data.

    X_* are pandas DataFrames with MIXED dtypes (numeric + categorical).
    Features are RAW — you own imputation, encoding, scaling, imbalance
    handling, model choice, and calibration.

    Target: predict whether income is >50K (class 1, minority).

    Args:
        X_train: pd.DataFrame (n_train, n_features)
        y_train: np.ndarray (n_train,) binary labels {0, 1}
        X_val:   pd.DataFrame (n_val, n_features)
        y_val:   np.ndarray (n_val,) binary labels {0, 1}

    Returns:
        predict_proba: callable predict_proba(X) -> np.ndarray (n,)
            Positive-class scores for a DataFrame X with the same
            columns as X_train. Finite scores; ranking is what matters
            for PR-AUC.
    """
    # TODO: build any classical pipeline you like
    # Ideas:
    #   - ColumnTransformer: impute + scale numerics, encode categoricals
    #   - class_weight, resampling / SMOTE on the train fold only
    #   - logistic / SVM / RF / HistGradientBoosting / stacking
    #   - tune on val, refit on train+val
    #   - calibrate on val

    # 1. Dynamically isolate columns by data type
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # 2. Build the Numeric Sub-Pipeline
    # Impute missing values with the median and scale to mean 0, variance 1
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 3. Build the Categorical Sub-Pipeline
    # Impute missing strings with the most frequent value, then convert to dummy metrics.
    # handle_unknown='ignore' protects against unseen categories during validation/testing.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 4. Bind Preprocessors together into a single master ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )

    # 5. Define the Top-Tier Tabular Classifier
    # Tuning max_iter, learning_rate, and max_depth balances extreme generalization against time limits.
    classifier = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_depth=6,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42
    )

    # 6. Assemble the monolithic end-to-end Pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # 7. Fit the model exclusively on the training sets
    full_pipeline.fit(X_train, y_train)

    def predict_proba(X):
        # Return only the probabilities mapping directly to the positive class (column index 1)
        return full_pipeline.predict_proba(X)[:, 1]

    return predict_proba
