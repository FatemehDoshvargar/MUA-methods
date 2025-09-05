"""
This module implements a unified framework for connectivity-based predictive modeling
Author: Fatemeh Doshvargar
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, rankdata
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import time
import HCP

# DATA PREPROCESSING

def remove_subjects_with_missing_data(connectivity_matrix, behavioral_data,
                                    missing_strategy='any', verbose=True):
    """
    Remove subjects with missing data from connectivity and behavioral data.

    This function handles various input formats and automatically detects the
    correct orientation of the data, converting to standard format
    (subjects × features for 2D, subjects × regions × regions for 3D).

    Parameters
    ----------
    connectivity_matrix : array-like
        Either:
        - 2D: subjects × features (true format) or features × subjects
        - 3D: subjects × regions × regions (true format) or other orientations

    behavioral_data : array-like
        Either:
        - 1D: Behavioral scores (subjects,)
        - 2D: Behavioral scores (subjects, features) or (features, subjects)

    missing_strategy : str, default='any'
        Strategy for identifying missing data:
        - 'zero': behavioral_data == 0
        - 'nan': NaN values in behavioral_data
        - 'inf': inf/-inf values in behavioral_data
        - 'any': zero, NaN, or inf values in behavioral_data

    verbose : bool, default=True
        Whether to print information about removed subjects

    Returns
    -------
    clean_connectivity : array-like
        Connectivity data with missing subjects removed (in standard format)
    clean_behavioral : array-like
        Behavioral data with missing subjects removed (in standard format)
    removed_indices : array-like
        Indices of subjects that were removed
    """

    behavioral_data = np.array(behavioral_data)
    connectivity_matrix = np.array(connectivity_matrix)
    original_connectivity_shape = connectivity_matrix.shape
    original_behavioral_shape = behavioral_data.shape

    # Convert behavioral_data to standard format (subjects, features)
    if behavioral_data.ndim == 1:
        behavioral_true_format = behavioral_data.reshape(-1, 1)
        n_subjects_behavioral = len(behavioral_data)
    elif behavioral_data.ndim == 2:
        # Heuristic: assume the larger dimension is subjects
        if behavioral_data.shape[0] >= behavioral_data.shape[1]:
            behavioral_true_format = behavioral_data
            n_subjects_behavioral = behavioral_data.shape[0]
        else:
            behavioral_true_format = behavioral_data.T
            n_subjects_behavioral = behavioral_data.shape[1]
            if verbose:
                print(f"Behavioral data transposed from {behavioral_data.shape} to {behavioral_true_format.shape}")
    else:
        raise ValueError(f"Behavioral data must be 1D or 2D, got {behavioral_data.ndim}D")

    # Convert connectivity_matrix to standard format
    if connectivity_matrix.ndim == 2:
        # Auto-detect format based on behavioral data dimension
        if connectivity_matrix.shape[0] == n_subjects_behavioral:
            connectivity_true_format = connectivity_matrix
            n_subjects_connectivity = connectivity_matrix.shape[0]
        elif connectivity_matrix.shape[1] == n_subjects_behavioral:
            connectivity_true_format = connectivity_matrix.T
            n_subjects_connectivity = connectivity_matrix.shape[1]
            if verbose:
                print(f"Connectivity matrix transposed from {connectivity_matrix.shape} to {connectivity_true_format.shape}")
        else:
            # Fallback: assume larger dimension is subjects
            if connectivity_matrix.shape[0] >= connectivity_matrix.shape[1]:
                connectivity_true_format = connectivity_matrix
                n_subjects_connectivity = connectivity_matrix.shape[0]
            else:
                connectivity_true_format = connectivity_matrix.T
                n_subjects_connectivity = connectivity_matrix.shape[1]

    elif connectivity_matrix.ndim == 3:
        # Auto-detect 3D format and convert to (subjects, regions, regions)
        shape = connectivity_matrix.shape

        if shape[0] == n_subjects_behavioral:
            connectivity_true_format = connectivity_matrix
            n_subjects_connectivity = shape[0]
        elif shape[1] == n_subjects_behavioral:
            connectivity_true_format = np.transpose(connectivity_matrix, (1, 0, 2))
            n_subjects_connectivity = shape[1]
        elif shape[2] == n_subjects_behavioral:
            connectivity_true_format = np.transpose(connectivity_matrix, (2, 0, 1))
            n_subjects_connectivity = shape[2]
        else:
            # Fallback heuristics for ambiguous cases
            if shape[1] == shape[2] and shape[0] != shape[1]:
                connectivity_true_format = connectivity_matrix
                n_subjects_connectivity = shape[0]
            elif shape[0] == shape[2] and shape[0] != shape[1]:
                connectivity_true_format = np.transpose(connectivity_matrix, (1, 0, 2))
                n_subjects_connectivity = shape[1]
            elif shape[0] == shape[1] and shape[0] != shape[2]:
                connectivity_true_format = np.transpose(connectivity_matrix, (2, 0, 1))
                n_subjects_connectivity = shape[2]
            else:
                connectivity_true_format = connectivity_matrix
                n_subjects_connectivity = shape[0]

    else:
        raise ValueError(f"Connectivity matrix must be 2D or 3D, got {connectivity_matrix.ndim}D")

    # Verify subject counts match
    if n_subjects_connectivity != n_subjects_behavioral:
        raise ValueError(f"Subject count mismatch: connectivity has {n_subjects_connectivity} subjects, "
                        f"behavioral has {n_subjects_behavioral} subjects")

    # Find subjects to remove based on strategy
    if missing_strategy == 'zero':
        missing_mask = behavioral_true_format == 0
    elif missing_strategy == 'nan':
        missing_mask = np.isnan(behavioral_true_format)
    elif missing_strategy == 'inf':
        missing_mask = np.isinf(behavioral_true_format)
    elif missing_strategy == 'any':
        missing_mask = (behavioral_true_format == 0) | np.isnan(behavioral_true_format) | np.isinf(behavioral_true_format)
    else:
        raise ValueError(f"Unknown missing_strategy: {missing_strategy}")

    # Get indices of subjects to remove
    subjects_with_missing = np.any(missing_mask, axis=1) if behavioral_true_format.ndim == 2 else missing_mask.flatten()
    removed_indices = np.where(subjects_with_missing)[0]

    if len(removed_indices) == 0:
        if verbose:
            print("No subjects with missing data found.")
        return connectivity_true_format, behavioral_true_format.squeeze(), removed_indices

    # Remove subjects from both datasets
    clean_connectivity = np.delete(connectivity_true_format, removed_indices, axis=0)
    clean_behavioral = np.delete(behavioral_true_format, removed_indices, axis=0)

    # Squeeze behavioral data if it was originally 1D
    if original_behavioral_shape == (n_subjects_behavioral,):
        clean_behavioral = clean_behavioral.squeeze()

    if verbose:
        print(f"Missing data removal ({missing_strategy} strategy):")
        print(f"  Original subjects: {n_subjects_behavioral}")
        print(f"  Removed subjects: {len(removed_indices)}")
        print(f"  Final subjects: {len(clean_behavioral) if clean_behavioral.ndim == 1 else clean_behavioral.shape[0]}")
        print(f"  Connectivity shape: {original_connectivity_shape} → {clean_connectivity.shape}")
        print(f"  Behavioral shape: {original_behavioral_shape} → {clean_behavioral.shape}")

    return clean_connectivity, clean_behavioral, removed_indices


def vectorize_3d(connectome_3d, verbose=True):
    """
    Convert 3D connectome to 2D feature matrix by extracting upper triangle.

    Parameters
    ----------
    connectome_3d : array-like of shape (n_subjects, n_regions, n_regions)
        3D connectivity matrices
    verbose : bool, default=True
        Print conversion information

    Returns
    -------
    X : array-like of shape (n_subjects, n_features)
        2D feature matrix where features are upper triangular elements
    upper_tri_indices : tuple
        Indices of upper triangular elements
    """
    connectome_3d = np.array(connectome_3d)

    if connectome_3d.ndim != 3:
        raise ValueError(f"Input must be 3D array, got {connectome_3d.ndim}D")

    n_subjects, n_regions_1, n_regions_2 = connectome_3d.shape

    if n_regions_1 != n_regions_2:
        raise ValueError(f"Connectivity matrices must be square")

    n_regions = n_regions_1
    upper_tri_indices = np.triu_indices(n_regions, k=1)
    n_features = len(upper_tri_indices[0])

    X = np.zeros((n_subjects, n_features))
    for subject in range(n_subjects):
        X[subject, :] = connectome_3d[subject][upper_tri_indices]

    if verbose:
        print(f"3D to 2D vectorization:")
        print(f"  Input shape: {connectome_3d.shape}")
        print(f"  Output shape: {X.shape}")
        print(f"  Features per subject: {n_features}")

    return X, upper_tri_indices

# CORE ALGORITHMS

def vectorized_pearson_correlation(X, y):
    """
    Compute Pearson correlations for all features using vectorized operations.

    This function efficiently computes correlations between each feature and
    the target variable using matrix operations instead of loops.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target values

    Returns
    -------
    correlations : array-like of shape (n_features,)
        Correlation coefficients
    p_values : array-like of shape (n_features,)
        P-values for each correlation
    """
    n_samples, n_features = X.shape

    # Standardize y
    y_mean = np.mean(y)
    y_std = np.std(y, ddof=1)
    if y_std == 0:
        return np.zeros(n_features), np.ones(n_features)
    y_z = (y - y_mean) / y_std

    # Standardize X columns
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0, ddof=1)

    # Initialize arrays
    correlations = np.zeros(n_features)
    p_values = np.ones(n_features)

    # Find valid features (non-zero variance)
    valid_features = X_std > 1e-10

    if np.any(valid_features):
        # Standardize valid features
        X_z = np.zeros_like(X)
        X_z[:, valid_features] = (X[:, valid_features] - X_mean[valid_features]) / X_std[valid_features]

        # Compute correlations using matrix multiplication
        correlations[valid_features] = np.dot(X_z[:, valid_features].T, y_z) / (n_samples - 1)

        # Compute p-values
        t_stats = correlations[valid_features] * np.sqrt(
            (n_samples - 2) / (1 - correlations[valid_features]**2 + 1e-10))
        p_values[valid_features] = 2 * (1 - stats.t.cdf(np.abs(t_stats), n_samples - 2))

    return correlations, p_values


def vectorized_spearman_correlation(X, y):
    """
    Compute Spearman correlations for all features using vectorized operations.

    Spearman correlation is Pearson correlation of the ranks.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target values

    Returns
    -------
    correlations : array-like of shape (n_features,)
        Spearman correlation coefficients
    p_values : array-like of shape (n_features,)
        P-values for each correlation
    """

    # Rank the target variable
    y_ranked = rankdata(y)

    # Rank each feature column
    X_ranked = np.apply_along_axis(rankdata, axis=0, arr=X)

    correlations, p_values = vectorized_pearson_correlation(X_ranked, y_ranked)

    return correlations, p_values

# MAIN ESTIMATOR CLASS

class ConnectivityEstimator(BaseEstimator, RegressorMixin):
    """
    Unified Connectivity-based Predictive Modeling Estimator.

    This estimator implements both CPM (Connectome-based Predictive Modeling) and
    PCS (Polyconnectomic Scoring) approaches within a
    flexible framework that allows various aggregation, selection, and weighting
    strategies.

    Parameters
    ----------
    aggregation_method : str, default='networks'
        How to aggregate features:
        - 'networks': CPM-style - aggregate into positive/negative networks (2 features)
        - 'edges': PCS-style - each edge is a separate feature
        - 'single_score': Sum all weighted features into one predictor (1 feature)

    selection_method : str, default='pvalue'
        How to select edges:
        - 'pvalue': Select edges with p < threshold
        - 'top_k': Select top k edges by absolute correlation
        - 'all': Use all edges

    selection_threshold : float, default=0.05
        Threshold for edge selection:
        - If selection_method='pvalue': p-value threshold
        - If selection_method='top_k': number of edges to select
        - If selection_method='all': ignored

    weighting_method : str, default='binary'
        How to weight edges:
        - 'binary': +1/-1 for positive/negative correlations
        - 'correlation': Use correlation coefficients
        - 'squared_correlation': Use squared correlations
        - 'regression': Use linear regression coefficients

    correlation_type : str, default='pearson'
        Type of correlation: 'pearson' or 'spearman'

    regression_type : str, default='linear regression'
        Type of regression: 'linear regression', 'robust regression', 'ridge regression', 'lasso regression'

    Attributes
    ----------
    correlations_ : array-like of shape (n_features,)
        Correlation coefficients between each edge and the target
    p_values_ : array-like of shape (n_features,)
        P-values for each correlation
    selected_edges_ : array-like of shape (n_features,)
        Boolean mask indicating selected edges
    weights_ : array-like of shape (n_features,)
        Weights assigned to each edge
    n_selected_edges_ : int
        Number of selected edges
    n_positive_ : int
        Number of positive edges (if aggregation_method='networks')
    n_negative_ : int
        Number of negative edges (if aggregation_method='networks')
    model_ : object
        Fitted regression model
    """

    def __init__(self, aggregation_method='networks',
                 selection_method='pvalue', selection_threshold=0.05,
                 weighting_method='binary', correlation_type='pearson',
                 regression_type='linear regression'):

        self.aggregation_method = aggregation_method
        self.selection_method = selection_method
        self.selection_threshold = selection_threshold
        self.weighting_method = weighting_method
        self.correlation_type = correlation_type
        self.regression_type = regression_type

    def fit(self, X, y):
        """
        Fit the connectivity model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (vectorized connectivity features)
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : object
            Fitted estimator
        """
        n_samples, n_features = X.shape

        if self.correlation_type == 'pearson':
            self.correlations_, self.p_values_ = vectorized_pearson_correlation(X, y)
        elif self.correlation_type == 'spearman':
            self.correlations_, self.p_values_ = vectorized_spearman_correlation(X, y)
        else:
            raise ValueError(f"Unknown correlation type: {self.correlation_type}")

        # Step 2: Select edges based on method
        if self.selection_method == 'pvalue':
            self.selected_edges_ = self.p_values_ < self.selection_threshold
        elif self.selection_method == 'top_k':
            k = int(min(self.selection_threshold, n_features))
            top_k_indices = np.argpartition(np.abs(self.correlations_), -k)[-k:]
            self.selected_edges_ = np.zeros(n_features, dtype=bool)
            self.selected_edges_[top_k_indices] = True
        else:  # 'all'
            self.selected_edges_ = np.ones(n_features, dtype=bool)

        self.n_selected_edges_ = np.sum(self.selected_edges_)

        # Step 3: Calculate weights
        self.weights_ = self._calculate_weights(X, y)

        # Step 4: Create features based on aggregation method
        if self.aggregation_method == 'networks':
            # CPM-style: aggregate into positive and negative networks
            features = self._aggregate_networks(X)
            # Store network info
            self.n_positive_ = np.sum((self.weights_ > 0) & self.selected_edges_)
            self.n_negative_ = np.sum((self.weights_ < 0) & self.selected_edges_)
        elif self.aggregation_method == 'edges':
            # PCS-style: each edge is a feature
            features = self._create_edge_features(X)
        elif self.aggregation_method == 'single_score':
            # Single score: sum all weighted features
            features = self._aggregate_single_score(X)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        # Step 5: Fit regression model
        self._fit_regression(features, y)

        return self

    def predict(self, X):
        """
        Make predictions using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values
        """
        if not hasattr(self, 'weights_'):
            raise ValueError("Model not fitted yet.")

        # Create features
        if self.aggregation_method == 'networks':
            features = self._aggregate_networks(X)
        elif self.aggregation_method == 'edges':
            features = self._create_edge_features(X)
        elif self.aggregation_method == 'single_score':
            features = self._aggregate_single_score(X)

        # Make predictions
        return self._predict_regression(features)

    def _calculate_weights(self, X, y):
        """Calculate edge weights based on weighting method."""
        n_features = X.shape[1]
        weights = np.zeros(n_features)

        if self.weighting_method == 'binary':
            # Binary weights for selected edges
            weights[self.selected_edges_ & (self.correlations_ > 0)] = 1.0
            weights[self.selected_edges_ & (self.correlations_ < 0)] = -1.0

        elif self.weighting_method == 'correlation':
            # Use correlation coefficients
            weights[self.selected_edges_] = self.correlations_[self.selected_edges_]

        elif self.weighting_method == 'squared_correlation':
            # Squared correlations preserving sign
            weights[self.selected_edges_] = (np.sign(self.correlations_[self.selected_edges_]) *
                                           self.correlations_[self.selected_edges_]**2)

        elif self.weighting_method == 'regression':
            # Linear regression coefficients
            for i in np.where(self.selected_edges_)[0]:
                if np.std(X[:, i]) > 1e-10:
                    edge_values = X[:, i].reshape(-1, 1)
                    reg = LinearRegression().fit(edge_values, y)
                    weights[i] = reg.coef_[0]

        return weights

    def _aggregate_networks(self, X):
        """Aggregate edges into positive and negative networks (CPM-style)."""
        n_samples = X.shape[0]
        pos_mask = (self.weights_ > 0) & self.selected_edges_
        neg_mask = (self.weights_ < 0) & self.selected_edges_

        features = np.zeros((n_samples, 2))

        if np.any(pos_mask):
            if self.weighting_method == 'binary':
                # Sum for binary weights
                features[:, 0] = np.sum(X[:, pos_mask], axis=1)
            else:
                # Weighted sum for other methods
                pos_weights = np.abs(self.weights_[pos_mask])
                features[:, 0] = np.sum(X[:, pos_mask] * pos_weights, axis=1)

        if np.any(neg_mask):
            if self.weighting_method == 'binary':
                # Sum for binary weights
                features[:, 1] = np.sum(X[:, neg_mask], axis=1)
            else:
                # Weighted sum for other methods
                neg_weights = np.abs(self.weights_[neg_mask])
                features[:, 1] = np.sum(X[:, neg_mask] * neg_weights, axis=1)

        return features

    def _create_edge_features(self, X):
        """Create weighted edge features (PCS-style)."""
        selected_indices = np.where(self.selected_edges_)[0]
        n_samples = X.shape[0]
        n_selected = len(selected_indices)

        features = np.zeros((n_samples, n_selected))

        for i, edge_idx in enumerate(selected_indices):
            features[:, i] = X[:, edge_idx] * self.weights_[edge_idx]

        # Normalize features if PCS-style
        if not hasattr(self, 'scaler_'):
            self.scaler_ = StandardScaler()
            features = self.scaler_.fit_transform(features)
        else:
            features = self.scaler_.transform(features)

        return features

    def _aggregate_single_score(self, X):
        """
        Aggregate all weighted features into a single score.

        This implements: y = β₀ + β₁ × Σ(wᵢ × fᵢ) + ε
        where the sum of all weighted features becomes a single predictor.
        """
        n_samples = X.shape[0]
        features = np.zeros((n_samples, 1))  # Only 1 feature

        # Get selected edges
        selected_indices = np.where(self.selected_edges_)[0]

        if len(selected_indices) > 0:
            # Vectorized computation of weighted sum
            selected_weights = self.weights_[selected_indices]
            selected_features = X[:, selected_indices]
            features[:, 0] = np.dot(selected_features, selected_weights)

        # Standardize the single score
        if not hasattr(self, 'score_scaler_'):
            self.score_scaler_ = StandardScaler()
            features = self.score_scaler_.fit_transform(features)
        else:
            features = self.score_scaler_.transform(features)

        return features

    def _fit_regression(self, features, y):
        """Fit the regression model."""
        if self.regression_type in ['linear regression']:
            self.model_ = LinearRegression().fit(features, y)
        elif self.regression_type == 'robust regression':
            features_with_const = sm.add_constant(features)
            self.model_ = sm.RLM(y, features_with_const, M=sm.robust.norms.HuberT()).fit()
        elif self.regression_type == 'ridge regression':
            self.model_ = Ridge(alpha=1.0).fit(features, y)
        elif self.regression_type == 'lasso regression':
            self.model_ = Lasso(alpha=0.1).fit(features, y)

    def _predict_regression(self, features):
        """Make predictions with the regression model."""
        if self.regression_type == 'robust regression':
            features_with_const = sm.add_constant(features)
            return self.model_.predict(features_with_const)
        else:
            return self.model_.predict(features)

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True values for X

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

# ANALYSIS FUNCTIONS

def run_connectivity_analysis(connectome_data, behavioral_data,
                            n_splits=10, random_state=42,
                            verbose=True, **kwargs):
    """
    Run connectivity analysis with flexible parameter configuration.

    This function provides a high-level interface for running connectivity-based
    predictive modeling with cross-validation.

    Parameters
    ----------
    connectome_data : array-like of shape (n_subjects, n_regions, n_regions)
        3D connectome data
    behavioral_data : array-like of shape (n_subjects,)
        Behavioral measures
    n_splits : int, default=10
        Number of CV folds
    random_state : int, default=42
        Random seed
    verbose : bool, default=True
        Print progress
    **kwargs : dict
        Analysis parameters:
        - aggregation_method: 'networks' or 'edges' or 'single_score'
        - selection_method: 'pvalue', 'top_k', or 'all'
        - selection_threshold: float (p-value or number of edges)
        - weighting_method: 'binary', 'correlation', etc.
        - correlation_type: 'pearson' or 'spearman'
        - regression_type: 'linear regression', 'robust regression', 'ridge regression', 'lasso regression'

    Returns
    -------
    results : dict
        Analysis results including:
        - predictions: Cross-validated predictions
        - actual: True values
        - correlation: Pearson correlation between predictions and actual
        - mae: Mean absolute error
        - r2: R-squared score
        - model: Fitted model on all data
        - configuration: Used parameters

    Examples
    --------
    # Traditional CPM style
    results = run_connectivity_analysis(
        data, behavior,
        aggregation_method='networks',
        selection_method='pvalue',
        selection_threshold=0.05,
        weighting_method='binary',
        regression_type='linear regression'
    )

    # Traditional PCS style
    results = run_connectivity_analysis(
        data, behavior,
        aggregation_method='edges',
        selection_method='all',
        weighting_method='correlation',
        regression_type='linear regression'
    )

    # Single score aggregation
    results = run_connectivity_analysis(
        data, behavior,
        aggregation_method='single_score',
        selection_method='pvalue',
        selection_threshold=0.05,
        weighting_method='correlation',
        regression_type='linear regression'
    )
    """

    if verbose:
        print(f"Connectome data shape: {connectome_data.shape}")
        print("\n" + "="*60)
        print("CONNECTIVITY ANALYSIS")
        print("="*60)

    # Extract n_regions
    n_regions = connectome_data.shape[1]

    # Vectorize connectomes
    X, upper_tri_indices = vectorize_3d(connectome_data, verbose=verbose)

    # Create estimator
    estimator = ConnectivityEstimator(**kwargs)

    if verbose:
        print(f"\nConfiguration:")
        print(f"  Aggregation: {estimator.aggregation_method}")
        print(f"  Selection: {estimator.selection_method}")
        if estimator.selection_method != 'all':
            print(f"  Threshold: {estimator.selection_threshold}")
        print(f"  Weighting: {estimator.weighting_method}")
        print(f"  Correlation: {estimator.correlation_type}")
        print(f"  Regression: {estimator.regression_type}")

    # Cross-validation
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if verbose:
        print(f"\nRunning {n_splits}-fold cross-validation...")

    y_pred = cross_val_predict(estimator, X, behavioral_data, cv=cv)

    # Calculate metrics
    r_value = np.corrcoef(behavioral_data, y_pred)[0, 1]
    mae = mean_absolute_error(behavioral_data, y_pred)
    r2 = r2_score(behavioral_data, y_pred)

    # Train final model on all data
    estimator.fit(X, behavioral_data)

    # Compile results
    results = {
        'predictions': y_pred,
        'actual': behavioral_data,
        'correlation': r_value,
        'mae': mae,
        'r2': r2,
        'model': estimator,
        'upper_tri_indices': upper_tri_indices,
        'n_regions': n_regions,
        'configuration': {
            'aggregation_method': estimator.aggregation_method,
            'selection_method': estimator.selection_method,
            'selection_threshold': estimator.selection_threshold,
            'weighting_method': estimator.weighting_method,
            'correlation_type': estimator.correlation_type,
            'regression_type': estimator.regression_type
        }
    }

    # Add method-specific info
    if estimator.aggregation_method == 'networks':
        results['n_positive_features'] = estimator.n_positive_
        results['n_negative_features'] = estimator.n_negative_
    elif estimator.aggregation_method == 'single_score':
        results['n_selected_edges'] = estimator.n_selected_edges_
        results['n_features'] = 1  # Single score has only 1 feature
    else:
        results['n_selected_edges'] = estimator.n_selected_edges_

    if verbose:
        print(f"\nResults:")
        print(f"  Correlation (r): {r_value:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAE: {mae:.3f}")

        if estimator.aggregation_method == 'networks':
            print(f"  Positive features: {estimator.n_positive_}")
            print(f"  Negative features: {estimator.n_negative_}")
        elif estimator.aggregation_method == 'single_score':
            print(f"  Selected edges: {estimator.n_selected_edges_}")
            print(f"  Aggregated into: 1 feature")
        else:
            print(f"  Selected edges: {estimator.n_selected_edges_}")

    return results


def compare_configurations(connectome_data, behavioral_data,
                         configurations, n_splits=10, random_state=42):
    """
    Compare multiple analysis configurations.

    Parameters
    ----------
    connectome_data : array-like
        3D connectome data
    behavioral_data : array-like
        Behavioral data
    configurations : dict
        Dictionary of configuration names and parameters
    n_splits : int, default=10
        Number of CV folds
    random_state : int, default=42
        Random seed

    Returns
    -------
    comparison : dict
        Results for each configuration
    """

    print("="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)

    comparison = {}

    for name, config in configurations.items():
        print(f"\nRunning: {name}")
        results = run_connectivity_analysis(
            connectome_data, behavioral_data,
            n_splits=n_splits, random_state=random_state,
            verbose=False,
            **config
        )
        comparison[name] = results

        # Print summary
        print(f"  r = {results['correlation']:.3f}, "
              f"R² = {results['r2']:.3f}, "
              f"MAE = {results['mae']:.2f}")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Configuration':<25} {'r':<8} {'R²':<8} {'MAE':<8} {'Features'}")
    print("-"*80)

    for name, results in comparison.items():
        if results['model'].aggregation_method == 'networks':
            features = f"Pos:{results['n_positive_features']} Neg:{results['n_negative_features']}"
        elif results['model'].aggregation_method == 'single_score':
            features = f"1 (from {results['n_selected_edges']} edges)"
        else:
            features = f"Edges:{results['n_selected_edges']}"

        print(f"{name:<25} {results['correlation']:<8.3f} "
              f"{results['r2']:<8.3f} {results['mae']:<8.2f} {features}")

    return comparison

# VISUALIZATION

def plot_results(predictions, actual, title=None):
    """
    Plot prediction results with scatter plot and error distribution.

    Parameters
    ----------
    predictions : array-like
        Predicted values
    actual : array-like
        Actual values
    title : str, optional
        Plot title
    """
    if predictions is None:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    ax1.scatter(actual, predictions, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)

    min_val = min(actual.min(), predictions.min())
    max_val = max(actual.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.6)

    r_val = np.corrcoef(actual, predictions)[0, 1]
    r2 = r2_score(actual, predictions)

    ax1.set_xlabel('Actual Values', fontsize=12)
    ax1.set_ylabel('Predicted Values', fontsize=12)
    ax1.set_title(f'Prediction Accuracy\n(r = {r_val:.3f}, R² = {r2:.3f})', fontsize=12, pad=10)
    ax1.grid(True, alpha=0.3)

    # Error distribution
    errors = predictions - actual
    n, bins, patches = ax2.hist(errors, bins=25, edgecolor='black',
                               alpha=0.7, color='skyblue')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8)

    mu, std = stats.norm.fit(errors)
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax2.plot(x, p * n.max() / p.max(), 'k-', linewidth=2, alpha=0.8)

    mae = np.mean(np.abs(errors))
    ax2.set_xlabel('Prediction Error', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Error Distribution\n(MAE = {mae:.3f})', fontsize=12, pad=10)
    ax2.grid(True, alpha=0.3, axis='y')

    if title is None:
        title = 'Connectivity Analysis Results'

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("MASS UNIVARIATE AGGREGATION METHODS FOR MACHINE LEARNING")
    print("A Unified Scikit-learn Framework")
    print("="*80)

    mat_file_path = "H:/untitled/s_hcp_fc_noble_corr.mat"
    connectome_train1 = HCP.reconstruct_fc_matrix(mat_file_path)
    behavioral_train1 = HCP.extract_behavioral_data_1d(mat_file_path,'test1')
    connectome_train, behavioral_train = HCP.remove_subjects_with_missing_data(connectome_train1, behavioral_train1)

    configurations = {
        # Pearson-based methods (original)
        'CPM Binary (Pearson)': {
            'aggregation_method': 'networks',
            'selection_method': 'pvalue',
            'selection_threshold': 0.05,
            'weighting_method': 'binary',
            'correlation_type': 'pearson',
            'regression_type': 'linear regression'
        },
        'CPM Correlation (Pearson)': {
            'aggregation_method': 'networks',
            'selection_method': 'pvalue',
            'selection_threshold': 0.05,
            'weighting_method': 'correlation',
            'correlation_type': 'pearson',
            'regression_type': 'linear regression'
        },


        # Spearman-based methods (new tests)
        'CPM Binary (Pearson)': {
            'aggregation_method': 'networks',
            'selection_method': 'pvalue',
            'selection_threshold': 0.05,
            'weighting_method': 'binary',
            'correlation_type': 'pearson',
            'regression_type': 'linear regression'
        },
        'CPM Correlation (pearson)': {
            'aggregation_method': 'networks',
            'selection_method': 'pvalue',
            'selection_threshold': 0.05,
            'weighting_method': 'correlation',
            'correlation_type': 'pearson',
            'regression_type': 'linear regression'
        },

        # Single score methods (NEW)
        'Single Score (Pearson)': {
            'aggregation_method': 'single_score',
            'selection_method': 'pvalue',
            'selection_threshold': 0.05,
            'weighting_method': 'correlation',
            'correlation_type': 'pearson',
            'regression_type': 'linear regression'
        },
        'Single Score Top-500 (Spearman)': {
            'aggregation_method': 'single_score',
            'selection_method': 'top_k',
            'selection_threshold': 500,
            'weighting_method': 'correlation',
            'correlation_type': 'spearman',
            'regression_type': 'linear regression'
        }
    }

    # Run comparison
    comparison_results = compare_configurations(
        connectome_train,
        behavioral_train,
        configurations,
        n_splits=10
    )

    # Plot best performing method
    best_method = max(comparison_results.items(),
                     key=lambda x: x[1]['correlation'])
    print(f"\nBest performing method: {best_method[0]}")
    plot_results(
        best_method[1]['predictions'],
        best_method[1]['actual'],
        f'Best Method: {best_method[0]}'
    )