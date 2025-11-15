import argparse
import sys
from typing import Dict, Any, List
import polars as pl

## 1. Helper Functions

def get_yearly_top_k_products(
    review_lazy: pl.LazyFrame, 
    k: int, 
    time_column: str = "review_time",
    group_by_col: str = "product_id"
) -> Dict[int, pl.DataFrame]:
    """
    Calculates the top K most reviewed products for each year in the review data.
    """
    print(f"Calculating Top {k} products per year from review data...")
    
    # Find the year range in the review data
    try:
        range_df = review_lazy.select(
            pl.col(time_column).min().alias("min_timestamp"),
            pl.col(time_column).max().alias("max_timestamp"),
        ).collect()
    except Exception as e:
        print(f"Error collecting time range from review data: {e}", file=sys.stderr)
        return {}

    start_year, end_year = range_df['min_timestamp'][0].year, range_df['max_timestamp'][0].year
    
    yearly_top_products: Dict[int, pl.DataFrame] = {}

    for year in range(start_year, end_year + 1):
        # Apply the filter: timestamp.dt.year() == current year
        yearly_lf = review_lazy.filter(
            pl.col(time_column).dt.year() == year
        )
        
        top_k_plan = (
            yearly_lf
            .group_by(group_by_col)
            .agg(
                pl.len().alias("count") # Count reviews per product_id
            )
            .sort(
                "count", 
                descending=True # Sort to put highest counts first
            )
            .limit(k) # Keep only the top K
        )
        
        # Execute the plan and store the result
        yearly_top_products[year] = top_k_plan.collect()
        print(f"  ... calculated top {k} for {year}")

    return yearly_top_products


def create_train_feature(
    train_lazy: pl.LazyFrame, 
    review_lazy: pl.LazyFrame
) -> pl.LazyFrame:
    """
    Performs an asof join to find the most recent product a customer
    reviewed at or before each timestamp in the train data.
    """
    print("Performing asof join to get most_recent_product_id...")
    
    reviews_prepped_lf = (
        review_lazy
        .select("customer_id", "review_time", "product_id")
        .sort("review_time")
    )

    train_sorted_lf = train_lazy.sort("timestamp")

    # Perform the ASOF Join
    train_with_feature_lf = train_sorted_lf.join_asof(
        reviews_prepped_lf,
        left_on="timestamp",   # The time column from this (left) frame
        right_on="review_time", # The time column from the 'other' (right) frame
        by="customer_id",       # The exact-match key(s)
        strategy="backward"     # 'backward' = most recent at or before
    )

    train_with_feature_lf = train_with_feature_lf.rename(
        {"product_id": "most_recent_product_id"}
    )
    
    print("Asof join complete.")
    return train_with_feature_lf

## 2. Main Execution Logic

def main(args):
    """
    Main function to load data, process it based on flags,
    sample, and save the results.
    """
    
    # --- 1. Load Data (Lazy) ---
    print("Loading data lazily...")
    try:
        review_lazy = pl.scan_parquet(args.review_parquet)
        train_lazy = pl.scan_parquet(args.train_parquet)
        # product_lazy = pl.scan_parquet(args.product_parquet) # Not strictly needed for this logic
        # customer_lazy = pl.scan_parquet(args.customer_parquet) # Not strictly needed for this logic
    except Exception as e:
        print(f"Error loading Parquet files: {e}", file=sys.stderr)
        print("Please check file paths.", file=sys.stderr)
        return 1 # Exit with an error code

    # --- 2. Feature Engineering (Asof Join) ---
    # This step is common to both paths
    train_with_feature_lf = create_train_feature(train_lazy, review_lazy)

    # --- 3. Get Training Data Year Range ---
    try:
        train_range_df = train_lazy.select(
            pl.col("timestamp").min().alias("min_timestamp"),
            pl.col("timestamp").max().alias("max_timestamp"),
        ).collect()
        start_year, end_year = train_range_df['min_timestamp'][0].year, train_range_df['max_timestamp'][0].year
        train_years = list(range(start_year, end_year + 1))
        print(f"Training data covers years: {train_years}")
    except Exception as e:
        print(f"Error collecting time range from train data: {e}", file=sys.stderr)
        return 1

    # --- 4. Conditional Filtering Logic ---
    # This dictionary will hold the LazyFrames to be sampled from
    lfs_to_sample_from: Dict[int, pl.LazyFrame] = {}

    if args.filter_by_top_k_products:
        print(f"--- Filtering Yearly Train LFs based on Top {args.top_k} Products ---")
        
        # This step is expensive, only run if filtering
        yearly_top_products = get_yearly_top_k_products(review_lazy, args.top_k)

        for year in train_years:
            print(f"\nProcessing Year {year} (Filter Path):")
            
            top_products_df = yearly_top_products.get(year)
            if top_products_df is None or top_products_df.is_empty():
                print(f"  Warning: No 'yearly_top_products' data found for {year}. Skipping year.")
                continue
                
            top_product_ids = top_products_df.get_column("product_id")

            train_this_year_lf = train_with_feature_lf.filter(
                pl.col("timestamp").dt.year() == year
            )

            # Apply the filter
            filtered_lf = train_this_year_lf.filter(
                pl.col("most_recent_product_id").is_in(top_product_ids)
            )
            
            lfs_to_sample_from[year] = filtered_lf

    else:
        print("--- Skipping Top K Product Filter ---")
        print("Preparing to sample directly from asof-joined data...")
        
        for year in train_years:
            print(f"\nProcessing Year {year} (No Filter Path):")
            
            train_this_year_lf = train_with_feature_lf.filter(
                pl.col("timestamp").dt.year() == year
            )
            
            # Store the unfiltered (but year-split) LazyFrame
            lfs_to_sample_from[year] = train_this_year_lf

    # --- 5. Sampling ---
    sampled_train_yearly_dfs: Dict[int, pl.DataFrame] = {}
    print(f"\n--- Randomly Sampling {args.sample_size:,} Rows Per Year ---")

    for year, yearly_lf in lfs_to_sample_from.items():
        print(f"Year {year}: Collecting filtered data...")
        
        # We must collect before we can get an accurate length or sample
        try:
            yearly_df = yearly_lf.collect()
        except Exception as e:
            print(f"  Error collecting data for {year}: {e}", file=sys.stderr)
            continue
            
        available_rows = len(yearly_df)

        if available_rows == 0:
            print(f"Year {year}: No data available, skipping.")
            continue

        actual_sample_size = min(args.sample_size, available_rows)

        sampled_df = yearly_df.sample(
            n=actual_sample_size, 
            shuffle=True, 
            seed=args.seed
        )

        sampled_train_yearly_dfs[year] = sampled_df
        print(f"  Sampled {actual_sample_size:,} rows (from {available_rows:,} available)")

    # --- 6. Saving Results ---
    print(f"\n--- Saving Sampled DataFrames for Years: {args.years_to_save} ---")
    
    saved_count = 0
    for year in args.years_to_save:
        if year in sampled_train_yearly_dfs:
            df_to_save = sampled_train_yearly_dfs[year]
            output_path = f"{args.output_prefix}_{year}.parquet"
            try:
                df_to_save.write_parquet(output_path)
                print(f"  Successfully saved: {output_path} ({len(df_to_save):,} rows)")
                saved_count += 1
            except Exception as e:
                print(f"  Error saving file {output_path}: {e}", file=sys.stderr)
        else:
            print(f"  Warning: No sampled data found for year {year}. Nothing to save.")

    print(f"\nProcessing complete. Saved {saved_count} file(s).")
    return 0


## 3. Argument Parsing and Script Entrypoint

def setup_argparser():
    parser = argparse.ArgumentParser(description="Process and sample relational data.")
    
    # --- Input Files ---
    parser.add_argument("--review_parquet", type=str, required=True, help="Path to review.parquet")
    parser.add_argument("--train_parquet", type=str, required=True, help="Path to train.parquet")
    # Add paths for product and customer if needed by future logic
    # parser.add_argument("--product_parquet", type=str, required=True, help="Path to product.parquet")
    # parser.add_argument("--customer_parquet", type=str, required=True, help="Path to customer.parquet")

    # --- Processing Parameters ---
    parser.add_argument(
        "--filter_by_top_k_products",
        action="store_true",
        help="If set, filters train data by top K products"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1000,
        help="The 'K' value for finding top products. (Default: 1000)"
    )
    
    # --- Sampling Parameters ---
    parser.add_argument(
        "--sample_size",
        type=int,
        default=5000,
        help="The target number of rows to sample per year. (Default: 5000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for sampling. (Default: 123)"
    )
    
    # --- Output Parameters ---
    parser.add_argument(
        "--years_to_save",
        type=int,
        nargs='+',
        default=[2012, 2013, 2014, 2015],
        help="List of years to process and save. (Default: 2012 2013 2014 2015)"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="train_sample_",
        help="Prefix for the output parquet files. (Default: 'train_sample_')"
    )
    
    return parser

if __name__ == "__main__":
    # Suppress Polars warnings
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=".*Sortedness of columns cannot be checked.*",
        category=UserWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=".*`is_in` with a collection of the same datatype is ambiguous.*",
        category=pl.exceptions.PolarsDeprecatedWarning,
    )

    arg_parser = setup_argparser()
    args = arg_parser.parse_args()
    
    # Run main logic
    sys.exit(main(args))