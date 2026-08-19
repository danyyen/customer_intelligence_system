"""retail_segmentation: cancellation-aware RFM feature engineering and hybrid
(K-Means + DBSCAN) customer segmentation, built for the UCI Online Retail II
dataset.

See the project README and retail_customer_segmentation.ipynb for the full
analysis narrative -- this package holds the reusable logic behind it.
"""


from .rfm import (
    build_purchases_df,
    build_rfm_table,
    net_group_fifo,
    net_group_price_compatible,
)
from .clustering import (
    dbscan_core_outliers,
    dbscan_k_distance,
    hybrid_segments,
    kmeans_elbow_silhouette,
    kmeans_stability,
    scale_rfm,
)
from .viz import SEGMENT_PALETTE, apply_style, style_ax

__all__ = [
    "load_transactions",
    "tag_invoice_type",
    "categorize_stock_code",
    "build_model_df",
    "net_group_fifo",
    "net_group_price_compatible",
    "build_purchases_df",
    "build_rfm_table",
    "scale_rfm",
    "kmeans_elbow_silhouette",
    "kmeans_stability",
    "dbscan_k_distance",
    "dbscan_core_outliers",
    "hybrid_segments",
    "apply_style",
    "style_ax",
    "SEGMENT_PALETTE",
]
