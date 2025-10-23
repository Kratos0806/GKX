"""
Data preprocessing module for Gu, Kelly & Xiu (2020) replication.

This module implements the core data transformations:
1. Cross-sectional ranking to [-1, 1] interval
2. Missing value imputation with cross-sectional medians
3. Feature engineering with macro-characteristic interactions
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class GKXPreprocessor:
    """
    Preprocessor for implementing GKX (2020) data transformations.
    """

    def __init__(
        self,
        stock_characteristics: List[str],
        macro_predictors: List[str],
        industry_dummies: Optional[List[str]] = None,
        industry_column: Optional[str] = None,
        date_col: str = 'date',
        return_col: str = 'ret_excess',
        stock_id_col: str = 'permno'
    ):
        """
        Initialize the preprocessor.

        Parameters
        ----------
        stock_characteristics : List[str]
            Names of stock characteristic columns
        macro_predictors : List[str]
            Names of macroeconomic predictor columns
        industry_dummies : Optional[List[str]]
            Names of industry dummy columns (if already one-hot encoded)
        industry_column : Optional[str]
            Name of categorical industry column (e.g., 'sic2') to be one-hot encoded
        date_col : str
            Name of date column
        return_col : str
            Name of return column (target variable)
        stock_id_col : str
            Name of stock identifier column
        """
        self.stock_characteristics = stock_characteristics
        self.macro_predictors = macro_predictors
        self.industry_dummies = industry_dummies or []
        self.industry_column = industry_column
        self.date_col = date_col
        self.return_col = return_col
        self.stock_id_col = stock_id_col

        # Will be populated during fit
        self.interaction_features = []
        self.created_industry_dummies = []

    def cross_sectional_rank(
        self,
        df: pd.DataFrame,
        features: List[str]
    ) -> pd.DataFrame:
        """
        Apply cross-sectional ranking transformation.

        Ranks each feature within each time period and maps to [-1, 1] interval.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with date column
        features : List[str]
            Features to rank

        Returns
        -------
        pd.DataFrame
            DataFrame with ranked features
        """
        df_ranked = df.copy()

        for feature in features:
            if feature in df.columns:
                # Group by date and rank within each period
                df_ranked[feature] = df.groupby(self.date_col)[feature].transform(
                    lambda x: self._rank_to_interval(x)
                )

        return df_ranked

    @staticmethod
    def _rank_to_interval(series: pd.Series) -> pd.Series:
        """
        Rank values and map to [-1, 1] interval.

        Parameters
        ----------
        series : pd.Series
            Input series

        Returns
        -------
        pd.Series
            Ranked and scaled series in [-1, 1]
        """
        # Handle all NaN case
        if series.isna().all():
            return series

        # Rank with average method for ties, ignoring NaN
        ranks = series.rank(method='average', na_option='keep')

        # Get count of non-NaN values
        n_valid = series.notna().sum()

        if n_valid <= 1:
            # If only one valid value, map to 0
            return ranks.apply(lambda x: 0 if pd.notna(x) else np.nan)

        # Map to [-1, 1]: 2 * (rank - 1) / (n - 1) - 1
        scaled = 2 * (ranks - 1) / (n_valid - 1) - 1

        return scaled

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        features: List[str]
    ) -> pd.DataFrame:
        """
        Handle missing values using cross-sectional median imputation.

        First replaces NaN with cross-sectional median, then remaining NaN with 0.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        features : List[str]
            Features to impute

        Returns
        -------
        pd.DataFrame
            DataFrame with imputed values
        """
        df_imputed = df.copy()

        for feature in features:
            if feature in df.columns:
                # Fill with cross-sectional median
                df_imputed[feature] = df.groupby(self.date_col)[feature].transform(
                    lambda x: x.fillna(x.median())
                )

                # Fill remaining NaN with 0
                df_imputed[feature] = df_imputed[feature].fillna(0)

        return df_imputed

    def create_industry_dummies(
        self,
        df: pd.DataFrame,
        industry_column: str
    ) -> pd.DataFrame:
        """
        Create one-hot encoded industry dummies from categorical industry column.

        Following the tutorial: step_dummy(sic2, one_hot = TRUE)

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        industry_column : str
            Name of categorical industry column (e.g., 'sic2')

        Returns
        -------
        pd.DataFrame
            DataFrame with one-hot encoded industry dummies
        """
        df_dummies = df.copy()

        if industry_column in df.columns:
            # Create one-hot encoding
            dummies = pd.get_dummies(df[industry_column], prefix=industry_column, dtype=int)

            # Store dummy names
            self.created_industry_dummies = dummies.columns.tolist()

            # Concatenate with original dataframe
            df_dummies = pd.concat([df_dummies, dummies], axis=1)

            # Drop original categorical column
            df_dummies = df_dummies.drop(columns=[industry_column])

        return df_dummies

    def add_macro_intercept(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add macro intercept column (all ones).

        Following the tutorial: mutate(macro_intercept = 1)

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        pd.DataFrame
            DataFrame with macro_intercept column
        """
        df_intercept = df.copy()
        df_intercept['macro_intercept'] = 1
        return df_intercept

    def create_interaction_features(
        self,
        df: pd.DataFrame,
        macro_predictors: Optional[List[str]] = None,
        stock_chars: Optional[List[str]] = None,
        keep_original_cols: bool = False
    ) -> pd.DataFrame:
        """
        Create interaction terms between macro predictors and stock characteristics.

        Following the tutorial: step_interact(terms = ~ contains("characteristic"):contains("macro"),
                                              keep_original_cols = FALSE)

        IMPORTANT: The tutorial sets keep_original_cols = FALSE, meaning only interactions are kept.
        This is the default behavior to match the GKX (2020) replication exactly.

        The tutorial creates: 94 characteristics × 9 macros (8 + intercept) = 846 interactions
        Plus 74 industry dummies = 920 total features

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        macro_predictors : Optional[List[str]]
            Macro predictors to use (defaults to self.macro_predictors + 'macro_intercept')
        stock_chars : Optional[List[str]]
            Stock characteristics to use (defaults to self.stock_characteristics)
        keep_original_cols : bool
            Whether to keep original columns (True) or only interactions (False)
            Tutorial uses False (default) to match exact replication with 920 features

        Returns
        -------
        pd.DataFrame
            DataFrame with interaction features (and optionally original features)
        """
        if macro_predictors is None:
            # Include macro_intercept in the predictors
            macro_predictors = self.macro_predictors + ['macro_intercept']
        if stock_chars is None:
            stock_chars = self.stock_characteristics

        df_interact = df.copy()
        self.interaction_features = []

        # Create all interaction terms
        for macro in macro_predictors:
            for char in stock_chars:
                if macro in df.columns and char in df.columns:
                    interaction_name = f"{char}_x_{macro}"
                    df_interact[interaction_name] = df[char] * df[macro]
                    self.interaction_features.append(interaction_name)

        # If not keeping original columns, drop the characteristics and macros
        # (but keep industry dummies and other essential columns)
        if not keep_original_cols:
            cols_to_drop = [col for col in (stock_chars + macro_predictors)
                          if col in df_interact.columns]
            df_interact = df_interact.drop(columns=cols_to_drop)

        return df_interact

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline.

        Steps (following GKX 2020 tutorial):
        1. Cross-sectional ranking of stock characteristics
        2. Missing value imputation (median → 0)
        3. Create industry dummies (if sic2 column provided)
        4. Add macro intercept
        5. Create interaction features (characteristics × macros)
        6. Drop original characteristics and macros (keep only interactions + industry dummies)

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with all required columns

        Returns
        -------
        pd.DataFrame
            Fully preprocessed dataframe with exactly 920 features (for GKX dataset)
        """
        df_processed = df.copy()

        print("Step 1: Cross-sectional ranking...")
        df_processed = self.cross_sectional_rank(df_processed, self.stock_characteristics)

        print("Step 2: Handling missing values...")
        features_to_impute = (
            self.stock_characteristics +
            self.macro_predictors +
            self.industry_dummies
        )
        df_processed = self.handle_missing_values(df_processed, features_to_impute)

        print("Step 3: Creating industry dummies...")
        if self.industry_column is not None:
            df_processed = self.create_industry_dummies(df_processed, self.industry_column)
            print(f"  Created {len(self.created_industry_dummies)} industry dummies")

        print("Step 4: Adding macro intercept...")
        df_processed = self.add_macro_intercept(df_processed)

        print("Step 5: Creating interaction features...")
        df_processed = self.create_interaction_features(df_processed)
        print(f"  Created {len(self.interaction_features)} interaction features")

        all_features = self.get_all_features()
        print(f"\nTotal features: {len(all_features)}")
        print(f"  - Interaction features: {len(self.interaction_features)}")
        print(f"  - Industry dummies: {len(self.industry_dummies) + len(self.created_industry_dummies)}")

        return df_processed

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessing pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        pd.DataFrame
            Transformed dataframe
        """
        return self.fit_transform(df)

    def get_all_features(self) -> List[str]:
        """
        Get list of all feature names after preprocessing.

        Following GKX tutorial with keep_original_cols=FALSE:
        - Only interaction features (characteristics × macros)
        - Plus industry dummies

        Returns
        -------
        List[str]
            All feature names (interactions + industry dummies)
        """
        # With keep_original_cols=FALSE (default), we only have:
        # - Interaction features
        # - Industry dummies (either pre-existing or created)
        all_industries = self.industry_dummies + self.created_industry_dummies

        return self.interaction_features + all_industries


def create_temporal_splits(
    df: pd.DataFrame,
    date_col: str = 'date',
    train_start: str = '1957-03',
    train_end: str = '1974-12',
    validation_months: int = 12,
    refit_frequency: str = 'annual'
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Create expanding window train-validation-test splits.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with date column
    date_col : str
        Name of date column
    train_start : str
        Start date for initial training period (YYYY-MM format)
    train_end : str
        End date for initial training period (YYYY-MM format)
    validation_months : int
        Number of months for validation window
    refit_frequency : str
        How often to refit model ('annual' or 'monthly')

    Returns
    -------
    List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
        List of (train, validation, test) splits
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    splits = []

    # Get unique dates
    unique_dates = sorted(df[date_col].unique())

    # Convert train_start and train_end to datetime
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)

    # Find initial training end index
    train_end_idx = None
    for idx, date in enumerate(unique_dates):
        if date >= train_end_dt:
            train_end_idx = idx
            break

    if train_end_idx is None:
        raise ValueError("train_end date not found in dataset")

    # Determine step size based on refit frequency
    step = 12 if refit_frequency == 'annual' else 1

    # Create expanding window splits
    current_train_end_idx = train_end_idx

    while current_train_end_idx + validation_months < len(unique_dates):
        # Training set: from start to current_train_end_idx
        train_end_date = unique_dates[current_train_end_idx]
        train_df = df[df[date_col] <= train_end_date].copy()

        # Validation set: next validation_months
        val_start_idx = current_train_end_idx + 1
        val_end_idx = min(val_start_idx + validation_months - 1, len(unique_dates) - 1)
        val_start_date = unique_dates[val_start_idx]
        val_end_date = unique_dates[val_end_idx]
        val_df = df[
            (df[date_col] >= val_start_date) &
            (df[date_col] <= val_end_date)
        ].copy()

        # Test set: month after validation
        test_idx = val_end_idx + 1
        if test_idx >= len(unique_dates):
            break
        test_date = unique_dates[test_idx]
        test_df = df[df[date_col] == test_date].copy()

        if len(test_df) > 0:
            splits.append((train_df, val_df, test_df))

        # Move forward by step
        current_train_end_idx += step

    return splits


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)

    # Create sample data
    dates = pd.date_range('1957-03-01', '1960-12-01', freq='MS')
    n_stocks = 100

    data = []
    for date in dates:
        for stock_id in range(n_stocks):
            row = {
                'date': date,
                'permno': stock_id,
                'ret_excess': np.random.normal(0.01, 0.05),
                'char1': np.random.normal(0, 1),
                'char2': np.random.normal(0, 1),
                'macro1': np.random.normal(0, 0.5),
                'industry_1': np.random.choice([0, 1]),
            }
            data.append(row)

    df = pd.DataFrame(data)

    # Add some missing values
    df.loc[df.sample(frac=0.1).index, 'char1'] = np.nan
    df.loc[df.sample(frac=0.1).index, 'char2'] = np.nan

    # Initialize preprocessor
    preprocessor = GKXPreprocessor(
        stock_characteristics=['char1', 'char2'],
        macro_predictors=['macro1'],
        industry_dummies=['industry_1']
    )

    # Apply preprocessing
    df_processed = preprocessor.fit_transform(df)

    print("\nOriginal shape:", df.shape)
    print("Processed shape:", df_processed.shape)
    print("\nFeatures:", preprocessor.get_all_features())
    print("\nSample of processed data:")
    print(df_processed.head())
