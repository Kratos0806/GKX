"""
Model training module for Gu, Kelly & Xiu (2020) replication.

Implements Random Forest with hyperparameter tuning on validation set.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib


class GKXRandomForest:
    """
    Random Forest model with hyperparameter tuning for GKX (2020) replication.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        mtry_values: Optional[List[int]] = None,
        min_samples_split_values: Optional[List[int]] = None,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 1
    ):
        """
        Initialize Random Forest with hyperparameter grid.

        Parameters
        ----------
        n_estimators : int
            Number of trees (fixed at 300 in GKX paper)
        mtry_values : Optional[List[int]]
            Values for max_features to tune (GKX uses {3, 5, 10, 20, 30, 50})
        min_samples_split_values : Optional[List[int]]
            Values for min_samples_split to tune (GKX uses {5000, 10000})
        n_jobs : int
            Number of parallel jobs
        random_state : int
            Random seed
        verbose : int
            Verbosity level
        """
        self.n_estimators = n_estimators
        self.mtry_values = mtry_values or [3, 5, 10, 20, 30, 50]
        self.min_samples_split_values = min_samples_split_values or [5000, 10000]
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self.best_params = None
        self.best_model = None
        self.best_score = np.inf
        self.tuning_results = []

    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict:
        """
        Tune hyperparameters using validation set RMSE.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation target

        Returns
        -------
        Dict
            Best hyperparameters found
        """
        if self.verbose:
            print(f"\nTuning Random Forest with {len(self.mtry_values)} x "
                  f"{len(self.min_samples_split_values)} = "
                  f"{len(self.mtry_values) * len(self.min_samples_split_values)} configurations...")

        self.best_score = np.inf
        self.tuning_results = []

        for mtry in self.mtry_values:
            for min_samples_split in self.min_samples_split_values:
                # Ensure mtry doesn't exceed number of features
                max_features = min(mtry, X_train.shape[1])

                if self.verbose:
                    print(f"  Testing max_features={max_features}, "
                          f"min_samples_split={min_samples_split}...", end=' ')

                # Train model
                model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_features=max_features,
                    min_samples_split=min_samples_split,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    bootstrap=True,
                    oob_score=False
                )

                model.fit(X_train, y_train)

                # Evaluate on validation set
                y_val_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

                if self.verbose:
                    print(f"RMSE: {rmse:.6f}")

                # Store results
                self.tuning_results.append({
                    'max_features': max_features,
                    'min_samples_split': min_samples_split,
                    'rmse': rmse
                })

                # Update best model
                if rmse < self.best_score:
                    self.best_score = rmse
                    self.best_params = {
                        'max_features': max_features,
                        'min_samples_split': min_samples_split
                    }
                    self.best_model = model

        if self.verbose:
            print(f"\nBest hyperparameters: {self.best_params}")
            print(f"Best validation RMSE: {self.best_score:.6f}")

        return self.best_params

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        params: Optional[Dict] = None
    ):
        """
        Fit Random Forest model.

        If X_val and y_val provided, will tune hyperparameters.
        Otherwise, uses provided params or default params.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : Optional[pd.DataFrame]
            Validation features for hyperparameter tuning
        y_val : Optional[pd.Series]
            Validation target for hyperparameter tuning
        params : Optional[Dict]
            Hyperparameters to use (if not tuning)
        """
        if X_val is not None and y_val is not None:
            # Tune hyperparameters
            self.tune_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            # Use provided params or defaults
            if params is None:
                params = {
                    'max_features': min(10, X_train.shape[1]),
                    'min_samples_split': 5000
                }

            self.best_params = params

            if self.verbose:
                print(f"Training Random Forest with params: {params}")

            self.best_model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_features=params['max_features'],
                min_samples_split=params['min_samples_split'],
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                bootstrap=True
            )

            self.best_model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using best model.

        Parameters
        ----------
        X : pd.DataFrame
            Features

        Returns
        -------
        np.ndarray
            Predictions
        """
        if self.best_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.best_model.predict(X)

    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance from best model.

        Parameters
        ----------
        feature_names : List[str]
            Names of features

        Returns
        -------
        pd.DataFrame
            Feature importance sorted by importance
        """
        if self.best_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df

    def save_model(self, filepath: str):
        """
        Save trained model to disk.

        Parameters
        ----------
        filepath : str
            Path to save model
        """
        if self.best_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        joblib.dump({
            'model': self.best_model,
            'params': self.best_params,
            'score': self.best_score,
            'tuning_results': self.tuning_results
        }, filepath)

        if self.verbose:
            print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load trained model from disk.

        Parameters
        ----------
        filepath : str
            Path to load model from
        """
        data = joblib.load(filepath)
        self.best_model = data['model']
        self.best_params = data['params']
        self.best_score = data.get('score', None)
        self.tuning_results = data.get('tuning_results', [])

        if self.verbose:
            print(f"Model loaded from {filepath}")
            print(f"Parameters: {self.best_params}")


class OutOfSamplePredictor:
    """
    Manages out-of-sample predictions with expanding window training.
    """

    def __init__(
        self,
        model_class=GKXRandomForest,
        model_params: Optional[Dict] = None,
        verbose: int = 1
    ):
        """
        Initialize predictor.

        Parameters
        ----------
        model_class : class
            Model class to use
        model_params : Optional[Dict]
            Parameters for model initialization
        verbose : int
            Verbosity level
        """
        self.model_class = model_class
        self.model_params = model_params or {}
        self.verbose = verbose
        self.models = []
        self.predictions = []

    def run_expanding_window(
        self,
        splits: List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
        feature_cols: List[str],
        target_col: str = 'ret_excess',
        stock_id_col: str = 'permno',
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Run expanding window predictions.

        Parameters
        ----------
        splits : List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
            List of (train, validation, test) splits
        feature_cols : List[str]
            Feature column names
        target_col : str
            Target column name
        stock_id_col : str
            Stock identifier column
        date_col : str
            Date column

        Returns
        -------
        pd.DataFrame
            Out-of-sample predictions with stock IDs and dates
        """
        all_predictions = []

        for i, (train_df, val_df, test_df) in enumerate(splits):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Split {i+1}/{len(splits)}")
                print(f"Train: {train_df[date_col].min()} to {train_df[date_col].max()}")
                print(f"Val:   {val_df[date_col].min()} to {val_df[date_col].max()}")
                print(f"Test:  {test_df[date_col].unique()[0]}")
                print(f"Train size: {len(train_df):,}, Val size: {len(val_df):,}, Test size: {len(test_df):,}")
                print(f"{'='*60}")

            # Prepare data
            X_train = train_df[feature_cols].values
            y_train = train_df[target_col].values

            X_val = val_df[feature_cols].values
            y_val = val_df[target_col].values

            X_test = test_df[feature_cols].values

            # Initialize and train model
            model = self.model_class(**self.model_params)
            model.fit(
                pd.DataFrame(X_train, columns=feature_cols),
                pd.Series(y_train),
                pd.DataFrame(X_val, columns=feature_cols),
                pd.Series(y_val)
            )

            # Make predictions
            y_pred = model.predict(pd.DataFrame(X_test, columns=feature_cols))

            # Store predictions with metadata
            pred_df = pd.DataFrame({
                stock_id_col: test_df[stock_id_col].values,
                date_col: test_df[date_col].values,
                'predicted_return': y_pred,
                'actual_return': test_df[target_col].values
            })

            all_predictions.append(pred_df)
            self.models.append(model)

            if self.verbose:
                print(f"Predictions for {len(pred_df)} stocks completed.")

        # Combine all predictions
        predictions_df = pd.concat(all_predictions, ignore_index=True)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Total predictions: {len(predictions_df):,}")
            print(f"Date range: {predictions_df[date_col].min()} to {predictions_df[date_col].max()}")
            print(f"{'='*60}")

        return predictions_df


def train_and_predict(data_path: str, train_start: str, train_end: str,
                      n_estimators: int, results_dir: str,
                      validation_months: int = 12) -> str:
    """
    Simplified interface: Train RF model and generate predictions.

    Parameters
    ----------
    data_path : str
        Path to preprocessed data
    train_start : str
        Training start date (YYYY-MM)
    train_end : str
        Initial training end date (YYYY-MM)
    n_estimators : int
        Number of trees
    results_dir : str
        Directory to save results
    validation_months : int
        Number of months for validation period (default: 12)

    Returns
    -------
    str
        Path to predictions file
    """
    import os
    from preprocess import create_temporal_splits

    print("  Loading preprocessed data...")
    df = pd.read_csv(data_path, parse_dates=['month'])

    # Identify feature columns (all except month, permno, ret_excess, mktcap_lag)
    feature_cols = [col for col in df.columns
                    if col not in ['month', 'permno', 'ret_excess', 'mktcap_lag']]

    print(f"  Creating temporal splits...")
    splits = create_temporal_splits(
        df, date_col='month',
        train_start=train_start,
        train_end=train_end,
        validation_months=validation_months,
        refit_frequency='annual'
    )

    print(f"  Training on {len(splits)} splits...")
    predictor = OutOfSamplePredictor(
        model_class=GKXRandomForest,
        model_params={'n_estimators': n_estimators},
        verbose=1
    )

    predictions_df = predictor.run_expanding_window(splits, feature_cols, date_col='month')

    # Save predictions
    os.makedirs(results_dir, exist_ok=True)
    predictions_path = os.path.join(results_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)

    print(f"  Saved predictions to {predictions_path}")
    return predictions_path


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)

    # Create sample data
    n_samples = 10000
    n_features = 20

    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_train = pd.Series(
        X_train['feature_0'] * 0.5 + X_train['feature_1'] * 0.3 + np.random.randn(n_samples) * 0.1
    )

    X_val = pd.DataFrame(
        np.random.randn(2000, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_val = pd.Series(
        X_val['feature_0'] * 0.5 + X_val['feature_1'] * 0.3 + np.random.randn(2000) * 0.1
    )

    # Initialize and train model
    model = GKXRandomForest(
        n_estimators=100,  # Reduced for demo
        mtry_values=[3, 5, 10],
        min_samples_split_values=[100, 500],  # Reduced for demo
        verbose=1
    )

    model.fit(X_train, y_train, X_val, y_val)

    # Get feature importance
    importance = model.get_feature_importance(X_train.columns.tolist())
    print("\nTop 10 most important features:")
    print(importance.head(10))
