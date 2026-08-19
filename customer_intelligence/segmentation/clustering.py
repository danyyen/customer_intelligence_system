"""Feature scaling and the hybrid (DBSCAN outlier detection + K-Means)
customer segmentation used on top of the RFM table.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ["Recency", "Frequency", "Monetary"]


def scale_rfm(rfm_df: pd.DataFrame, feature_cols: Sequence[str] = FEATURE_COLS):
    """log1p-transform (handles Frequency>=1 and Monetary>=0 safely -- no
    flooring or exclusion needed) then standard-scale the RFM columns.

    Returns (scaled_df, fitted_scaler). Customer ID is carried through
    untouched, never part of the scaling.
    """
    log_df = rfm_df.copy()
    log_df[feature_cols] = np.log1p(rfm_df[feature_cols])

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(log_df[feature_cols])

    scaled_df = log_df[["Customer ID"]].copy()
    scaled_df[feature_cols] = scaled_values
    return scaled_df, scaler


def kmeans_elbow_silhouette(X: np.ndarray, k_range: Iterable[int] = range(2, 11), random_state: int = 42):
    """Fit K-Means across a range of k. Returns (list_of_k, inertias,
    silhouette_scores) for plotting/inspection -- picking k is a judgment
    call informed by both curves plus whether the resulting clusters make
    business sense, not an automatic argmax.
    """
    k_values = list(k_range)
    inertias, silhouette_scores = [], []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
    return k_values, inertias, silhouette_scores


def kmeans_stability(X: np.ndarray, k: int, seeds: Sequence[int] = (0, 1, 2, 3, 4, 42, 99, 123)):
    """Refit K-Means across several random seeds and return the pairwise
    Adjusted Rand Index between every pair of runs. ARI (not raw label
    matching) is used because K-Means cluster labels are arbitrary between
    runs -- ARI correctly measures whether the same customers keep ending up
    grouped together, regardless of which number a cluster gets.

    Values near 1.0 mean k reflects real, reliably-recovered structure;
    values near 0 mean it doesn't.
    """
    all_labels = []
    for seed in seeds:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        all_labels.append(km.fit_predict(X))

    ari_scores = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            ari_scores.append(adjusted_rand_score(all_labels[i], all_labels[j]))
    return ari_scores


def dbscan_k_distance(X: np.ndarray, min_samples: int = 10) -> np.ndarray:
    """Distance from every point to its min_samples-th nearest neighbor,
    sorted ascending. Plot this and look for the knee to choose DBSCAN's
    eps -- below the knee, points sit in dense neighborhoods; above it,
    they're increasingly isolated (candidate outliers).
    """
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)
    return np.sort(distances[:, -1])


def dbscan_core_outliers(
    rfm_scaled: pd.DataFrame,
    eps_values: Sequence[float],
    min_samples: int = 10,
    feature_cols: Sequence[str] = FEATURE_COLS,
) -> set:
    """Run DBSCAN at every eps in eps_values and return the customers flagged
    as noise at ALL of them -- robust to the exact eps choice, rather than
    committing to one arbitrary value.
    """
    X = rfm_scaled[feature_cols].values
    noise_sets = []
    for eps in eps_values:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        noise_sets.append(set(rfm_scaled.loc[labels == -1, "Customer ID"]))
    return set.intersection(*noise_sets) if noise_sets else set()


def hybrid_segments(
    rfm_df: pd.DataFrame,
    rfm_scaled: pd.DataFrame,
    eps_values: Sequence[float] = (0.40, 0.45, 0.50),
    min_samples: int = 10,
    k: int = 4,
    random_state: int = 42,
    feature_cols: Sequence[str] = FEATURE_COLS,
):
    """The full hybrid segmentation: DBSCAN identifies robust outliers
    (customers flagged as noise at every tested eps), then K-Means (k) is
    fit fresh on everyone else.

    The remainder gets its OWN freshly-fit scaler, not the full-population
    one -- removing the outliers shifts the remaining population's mean and
    std, so reusing the original scaling would leave the remainder
    calibrated against extremes that are no longer in the data.

    Returns (hybrid_df, kmeans_model, remainder_scaler). hybrid_df has one
    row per customer with a 'Segment' column: 'Outlier', or a K-Means
    cluster label (as a string, e.g. '0'..'3').
    """
    core_outliers = dbscan_core_outliers(rfm_scaled, eps_values, min_samples, feature_cols)

    is_outlier = rfm_df["Customer ID"].isin(core_outliers)
    rfm_outliers = rfm_df[is_outlier].reset_index(drop=True)
    rfm_remainder = rfm_df[~is_outlier].reset_index(drop=True)

    remainder_scaled, remainder_scaler = scale_rfm(rfm_remainder, feature_cols)
    X_remainder = remainder_scaled[feature_cols].values

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    remainder_labels = km.fit_predict(X_remainder)

    rfm_remainder = rfm_remainder.copy()
    rfm_remainder["Segment"] = remainder_labels.astype(str)
    rfm_outliers = rfm_outliers.copy()
    rfm_outliers["Segment"] = "Outlier"

    hybrid_df = pd.concat([rfm_outliers, rfm_remainder], ignore_index=True)
    return hybrid_df, km, remainder_scaler
