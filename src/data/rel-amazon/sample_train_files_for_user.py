import argparse
import sys
from typing import Dict, List
from pathlib import Path
import polars as pl

## 1. Helper Functions

def parse_time_window(window_str: str) -> str:
    # Return the negated offset string for Polars
    return f"-{window_str}"

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
    range_df = review_lazy.select(
        pl.col(time_column).min().alias("min_timestamp"),
        pl.col(time_column).max().alias("max_timestamp"),
    ).collect()


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


## 2. Main Execution Logic

def main(args):
    """
    Main function to load data, process it based on flags,
    sample, and save the results.
    """
    
    # --- 1. Load Data ---
    review_lazy = pl.scan_parquet(args.review_parquet)
    train_lazy = pl.scan_parquet(args.train_parquet)

    # --- 2. Get Training Data Year Range ---
    train_range_df = train_lazy.select(
        pl.col("timestamp").min().alias("min_timestamp"),
        pl.col("timestamp").max().alias("max_timestamp"),
    ).collect()
    start_year, end_year = train_range_df['min_timestamp'][0].year, train_range_df['max_timestamp'][0].year
    train_years = list(range(start_year, end_year + 1))
    print(f"Training data covers years: {train_years}")


    # --- 3. Conditional Filtering Logic ---
    # This dictionary will hold the LazyFrames to be sampled from
    lfs_to_sample_from: Dict[int, pl.LazyFrame] = {}

    if args.filter_by_top_k_products:
        print(f"--- Filtering Yearly Train LFs based on Top {args.top_k} Products ---")
        
        # This step is expensive, only run if filtering
        yearly_top_products = get_yearly_top_k_products(review_lazy, args.top_k)

        #print(yearly_top_products.get(2013))

        for year in train_years:
            print(f"\nProcessing Year {year} (Filter Path):")
            
            top_products_df = yearly_top_products.get(year)
                
            top_product_ids = top_products_df.get_column("product_id")

            # Filter reviews to only top K products for this year
            top_k_reviews_lf = review_lazy.filter(
                pl.col("product_id").is_in(top_product_ids)
            ).select("customer_id", "review_time").sort("review_time")

            # Get train data for this year
            train_this_year_lf = train_lazy.filter(
                pl.col("timestamp").dt.year() == year
            ).sort("timestamp")

            # Join to find if customer has ANY top K review at or before their timestamp
            joined_lf = train_this_year_lf.join_asof(
                top_k_reviews_lf,
                left_on="timestamp",
                right_on="review_time",
                by="customer_id",
                strategy="backward"
            )
            
            # Apply filtering based on whether time_window is specified
            if args.time_window:
                window_offset = parse_time_window(args.time_window)
                print(f"  Applying time window filter: {args.time_window} (offset: {window_offset})")
                # Keep rows where a top K review exists AND is within the time window
                filtered_lf = joined_lf.filter(
                    pl.col("review_time").is_not_null() &
                    (pl.col("review_time") >= pl.col("timestamp").dt.offset_by(window_offset))
                ).drop("review_time")
            else:
                # Original behavior: any review at or before timestamp
                filtered_lf = joined_lf.filter(
                    pl.col("review_time").is_not_null()
                ).drop("review_time")
            
            lfs_to_sample_from[year] = filtered_lf
    else:
        print("--- Skipping Top K Product Filter ---")
        print("Preparing to sample directly from train data...")
        
        for year in train_years:
            print(f"\nProcessing Year {year} (No Filter Path):")
            
            train_this_year_lf = train_lazy.filter(
                pl.col("timestamp").dt.year() == year
            )
            
            lfs_to_sample_from[year] = train_this_year_lf

    # --- 4. Collect and Combine All Yearly Data ---
    print("\n--- Collecting and combining data from all years ---")
    yearly_dfs: List[pl.DataFrame] = []
    
    for year, yearly_lf in lfs_to_sample_from.items():
        print(f"Year {year}: Collecting filtered data...")
        try:
            yearly_df = yearly_lf.collect()
            if len(yearly_df) > 0:
                yearly_dfs.append(yearly_df)
                print(f"  Collected {len(yearly_df):,} rows from {year}")
            else:
                print(f"  Year {year}: No data available, skipping.")
        except Exception as e:
            print(f"  Error collecting data for {year}: {e}", file=sys.stderr)
            continue
    
    if not yearly_dfs:
        print("Error: No data collected from any year.", file=sys.stderr)
        return 1
    
    # Concatenate all yearly DataFrames into one combined pool
    combined_df = pl.concat(yearly_dfs)
    available_rows = len(combined_df)
    print(f"\nTotal combined pool: {available_rows:,} rows")

    # --- 5. Sampling n_files times with sequential seeds ---
    sampled_dfs: List[pl.DataFrame] = []
    
    if args.churn_ratio is not None:
        print(f"\n--- Stratified Sampling {args.sample_size:,} Rows x {args.n_files} Files (Target: {args.churn_ratio*100:.0f}% churned / {(1-args.churn_ratio)*100:.0f}% not-churned) ---")
        
        # Pre-split by churn label (only need to do this once)
        churn_col = args.churn_column
        churned_df = combined_df.filter(pl.col(churn_col) == 1)
        not_churned_df = combined_df.filter(pl.col(churn_col) == 0)
        
        available_churned = len(churned_df)
        available_not_churned = len(not_churned_df)
        print(f"  Available: {available_churned:,} churned, {available_not_churned:,} not-churned")
        
        # Calculate target samples for each class
        target_churned = int(args.sample_size * args.churn_ratio)
        target_not_churned = args.sample_size - target_churned
        
        # Adjust if not enough samples in either class
        actual_churned = min(target_churned, available_churned)
        actual_not_churned = min(target_not_churned, available_not_churned)
        
        # If one class is short, try to compensate with the other (up to available)
        if actual_churned < target_churned:
            shortfall = target_churned - actual_churned
            extra_not_churned = min(shortfall, available_not_churned - actual_not_churned)
            actual_not_churned += extra_not_churned
        elif actual_not_churned < target_not_churned:
            shortfall = target_not_churned - actual_not_churned
            extra_churned = min(shortfall, available_churned - actual_churned)
            actual_churned += extra_churned
        
        for i in range(args.n_files):
            current_seed = args.seed + i
            
            # Sample from each class
            sampled_churned = churned_df.sample(
                n=actual_churned, shuffle=True, seed=current_seed
            ) if actual_churned > 0 else churned_df.head(0)
            
            sampled_not_churned = not_churned_df.sample(
                n=actual_not_churned, shuffle=True, seed=current_seed
            ) if actual_not_churned > 0 else not_churned_df.head(0)
            
            # Combine and shuffle
            sampled_df = pl.concat([sampled_churned, sampled_not_churned]).sample(
                fraction=1.0, shuffle=True, seed=current_seed
            )
            
            sampled_dfs.append(sampled_df)
            
            actual_sample_size = len(sampled_df)
            actual_ratio = actual_churned / actual_sample_size if actual_sample_size > 0 else 0
            print(f"  File {i}: Sampled {actual_sample_size:,} rows (seed={current_seed}): {actual_churned:,} churned ({actual_ratio*100:.1f}%), {actual_not_churned:,} not-churned ({(1-actual_ratio)*100:.1f}%)")
    else:
        print(f"\n--- Randomly Sampling {args.sample_size:,} Rows x {args.n_files} Files ---")
        
        actual_sample_size = min(args.sample_size, available_rows)
        
        for i in range(args.n_files):
            current_seed = args.seed + i
            
            sampled_df = combined_df.sample(
                n=actual_sample_size, 
                shuffle=True, 
                seed=current_seed
            )
            sampled_dfs.append(sampled_df)
            print(f"  File {i}: Sampled {actual_sample_size:,} rows (seed={current_seed})")

    # --- 6. Saving Results ---
    print(f"\n--- Saving {args.n_files} Sampled DataFrames ---")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving files to: {output_dir.resolve()}")
    
    saved_count = 0
    for i, df_to_save in enumerate(sampled_dfs):
        output_path = output_dir / f"{args.output_prefix}{i}.parquet"
        try:
            df_to_save.write_parquet(output_path)
            print(f"  Successfully saved: {output_path} ({len(df_to_save):,} rows)")
            saved_count += 1
        except Exception as e:
            print(f"  Error saving file {output_path}: {e}", file=sys.stderr)

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
    parser.add_argument(
        "--time_window",
        type=str,
        default=None,
        help="Time window for filtering (e.g., '6mo', '1y', '1y6mo'). If not set, considers all reviews before timestamp."
    )
    
    # --- Sampling Parameters ---
    parser.add_argument(
        "--sample_size",
        type=int,
        default=5000,
        help="The target number of rows to sample from the combined pool. (Default: 5000)"
    )
    parser.add_argument(
        "--n_files",
        type=int,
        default=1,
        help="Number of output files to generate with different random seeds. (Default: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for sampling. (Default: 123)"
    )
    parser.add_argument(
        "--churn_ratio",
        type=float,
        default=None,
        help="Target ratio of churned (class 1) samples (e.g., 0.6 for 60%% churned). If not set, random sampling without class balance is used."
    )
    parser.add_argument(
        "--churn_column",
        type=str,
        default="churn",
        help="Name of the binary churn label column. (Default: 'churn')"
    )
    
    # --- Output Parameters ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the output parquet files. (Default: current directory)"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="train_sample_",
        help="Prefix for the output parquet files. (Default: 'train_sample_')"
    )
    
    return parser

if __name__ == "__main__":
    arg_parser = setup_argparser()
    args = arg_parser.parse_args()
    
    sys.exit(main(args))