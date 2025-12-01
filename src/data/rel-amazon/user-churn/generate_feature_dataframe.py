import argparse
import polars as pl
import os


def create_customer_features(
    train_df: pl.DataFrame,        # Training data with: timestamp, customer_id, churn
    review_lazy: pl.LazyFrame,     # Review table
    product_lazy: pl.LazyFrame,    # Product table  
    time_window: str = "-6mo"      # Lookback window (e.g., "-6mo", "-1y")
) -> pl.DataFrame:
    """
    Create a simple feature vector for each customer based on their review history
    within the specified time window before the prediction timestamp.
    """
    
    # Convert train_df to lazy if needed
    if isinstance(train_df, pl.DataFrame):
        train_lazy = train_df.lazy()
    else:
        train_lazy = train_df
    
    train_lazy = train_lazy.with_row_index('_original_idx')
    
    # Join reviews with products to get product info
    reviews_with_products = review_lazy.join(
        product_lazy.select(['product_id', 'price', 'category', 'brand']),
        on='product_id',
        how='left'
    )
    
    # Join training data with reviews (filtering by time window)
    train_with_reviews = train_lazy.join(
        reviews_with_products,
        on='customer_id',
        how='left'
    ).filter(
        # Only reviews BEFORE the prediction timestamp and within the window
        (pl.col('review_time') < pl.col('timestamp')) &
        (pl.col('review_time') >= pl.col('timestamp').dt.offset_by(time_window))
    )
    
    # Aggregate features per customer (per training row)
    customer_features = train_with_reviews.group_by(['customer_id', 'timestamp', 'churn'], maintain_order=True).agg([
        # Review count features
        pl.len().alias('num_reviews'),
        pl.col('product_id').n_unique().alias('num_unique_products'),
        
        # Rating features
        pl.col('rating').mean().alias('avg_rating'),
        pl.col('rating').std().alias('std_rating'),
        pl.col('rating').min().alias('min_rating'),
        pl.col('rating').max().alias('max_rating'),
        
        # Verified purchase features
        pl.col('verified').sum().alias('num_verified_purchases'),
        
        # Product diversity features
        pl.col('category').n_unique().alias('num_unique_categories'),
        pl.col('brand').n_unique().alias('num_unique_brands'),
        
        # Price features
        pl.col('price').mean().alias('avg_product_price'),
        pl.col('price').std().alias('std_product_price'),
        pl.col('price').min().alias('min_product_price'),
        pl.col('price').max().alias('max_product_price'),
        
        # Temporal features
        pl.col('review_time').max().alias('last_review_time'),
        pl.col('review_time').min().alias('first_review_time'),
    ]).with_columns([
        # Days since last review (recency)
        ((pl.col('timestamp') - pl.col('last_review_time')).dt.total_days()).alias('days_since_last_review'),
        # Days since first review (tenure)
        ((pl.col('timestamp') - pl.col('first_review_time')).dt.total_days()).alias('days_since_first_review'),
    ]).with_columns([
        # Review frequency (reviews per 30 days)
        (pl.col('num_reviews') / (pl.col('days_since_first_review') / 30.0 + 1e-6)).alias('review_frequency'),
    ]).drop(['last_review_time', 'first_review_time'])

    # LEFT JOIN back to original training data to keep ALL customers
    features = train_lazy.join(
        customer_features,
        on=['customer_id', 'timestamp', 'churn'],
        how='left',
        maintain_order='left'
    ).sort('_original_idx')	
    
    # Collect and handle missing values
    features_df = features.drop('_original_idx').collect()
    
    # Fill nulls with sensible defaults
    features_df = features_df.fill_null(0).fill_nan(0)
    
    return features_df

def get_feature_columns() -> list:
    """Returns the list of feature column names for the MLP input."""
    return [
        'num_reviews',
        'num_unique_products', 
        'avg_rating',
        'std_rating',
        'min_rating',
        'max_rating',
        'num_verified_purchases',
        'num_unique_categories',
        'num_unique_brands',
        'avg_product_price',
        'std_product_price',
        'min_product_price',
        'max_product_price',
        'days_since_last_review',
        'days_since_first_review',
        'review_frequency',
    ]

def save_features_parquet(df: pl.DataFrame, output_path: str, verbose: bool = False):
    """Save the features dataframe to a parquet file with a preview txt log."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.write_parquet(output_path)
    
    # Save preview of first 50 rows as txt log
    preview_path = output_path.replace('.parquet', '_preview.txt')
    with open(preview_path, 'w', encoding='utf-8') as f:
        f.write(f"Preview of features dataframe (first 50 rows)\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(df.head(50)))
        f.write(f"\n\n{'=' * 80}\n")
        f.write(f"Total rows: {len(df)}\n")
        f.write(f"Total columns: {len(df.columns)}\n")
        f.write(f"\nFeature columns:\n")
        for col in get_feature_columns():
            f.write(f"  - {col}\n")
    
    if verbose:
        print(f"Features saved to: {output_path}")
        print(f"Preview saved to: {preview_path}")
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")


def main(args):
    if args.verbose:
        print("Loading data files...")
        print(f"  Training file: {args.training_data_path}")
        print(f"  Review file: {args.review_path}")
        print(f"  Product file: {args.product_path}")
        print(f"  Time window: {args.time_window}")
    
    # Load data
    review_lazy = pl.scan_parquet(args.review_path)
    product_lazy = pl.scan_parquet(args.product_path)
    train_df = pl.scan_parquet(args.training_data_path)
    
    if args.verbose:
        print("\nGenerating customer features...")
    
    # Generate features
    features_df = create_customer_features(
        train_df, 
        review_lazy, 
        product_lazy, 
        args.time_window
    )
    
    if args.verbose:
        print(f"\nFeatures generated successfully!")
        print(f"Shape: {features_df.shape}")
        print(f"\nSample of features:\n{features_df.head()}")
    
    # Save features
    save_features_parquet(features_df, args.output_path, args.verbose)
    
    if args.verbose:
        print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate feature dataframe from training data for user churn prediction."
    )
    parser.add_argument(
        'training_data_path', 
        type=str, 
        help="Path to the training parquet file (must contain: customer_id, timestamp, churn)"
    )
    parser.add_argument(
        'output_path', 
        type=str, 
        help="Path for the output features parquet file"
    )
    parser.add_argument(
        '--time_window', 
        type=str, 
        default='-6mo', 
        help="Time window for feature lookback (e.g., '-6mo', '-1y', '-3y'). Default: -6mo"
    )
    parser.add_argument(
        '--base_data_path', 
        type=str, 
        default=r'data\1_raw', 
        help="Base directory for rel-amazon database files"
    )
    parser.add_argument(
        '--review_path', 
        type=str, 
        default=None, 
        help="Path to review.parquet (overrides base_data_path)"
    )
    parser.add_argument(
        '--product_path', 
        type=str, 
        default=None, 
        help="Path to product.parquet (overrides base_data_path)"
    )
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Set default paths based on base_data_path if not explicitly provided
    if args.review_path is None:
        args.review_path = os.path.join(args.base_data_path, 'rel-amazon', 'db', 'review.parquet')
    if args.product_path is None:
        args.product_path = os.path.join(args.base_data_path, 'rel-amazon', 'db', 'product.parquet')
    
    main(args)

