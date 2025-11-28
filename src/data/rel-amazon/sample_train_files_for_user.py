import argparse
import re
import sys
from typing import Dict
from pathlib import Path
import polars as pl

## 1. Helper Functions

def parse_time_window(window_str: str) -> str:
    """
    Parses a time window string (e.g., '6mo', '1y', '1y6mo') 
    and returns a negative offset string for Polars dt.offset_by().
    
    Supported formats:
        - '6mo' -> '-6mo'
        - '1y' -> '-1y'
        - '1y6mo' -> '-1y6mo'
        - '18mo' -> '-18mo'
    """
    # Pattern matches: optional years (e.g., '1y'), optional months (e.g., '6mo')
    pattern = r'^(?:(\d+)y)?(?:(\d+)mo)?$'
    match = re.match(pattern, window_str)
    
    if not match or (match.group(1) is None and match.group(2) is None):
        raise ValueError(
            f"Invalid time window format: '{window_str}'. "
            "Expected formats like '6mo', '1y', '1y6mo', '18mo'."
        )
    
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

        print(yearly_top_products.get(2013))

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

    print(lfs_to_sample_from[2013].head(15).collect())
    print("Number of rows in lfs_to_sample_from[2013]:", lfs_to_sample_from[2013].collect().height)
    #else:
    #    print("--- Skipping Top K Product Filter ---")
    #    print("Preparing to sample directly from train data...")
    #    
    #    for year in train_years:
    #        print(f"\nProcessing Year {year} (No Filter Path):")
    #        
    #        train_this_year_lf = train_lazy.filter(
    #            pl.col("timestamp").dt.year() == year
    #        )
    #        
    #        lfs_to_sample_from[year] = train_this_year_lf

    ## --- 4. Sampling ---
    #sampled_train_yearly_dfs: Dict[int, pl.DataFrame] = {}
    #print(f"\n--- Randomly Sampling {args.sample_size:,} Rows Per Year ---")

    #for year, yearly_lf in lfs_to_sample_from.items():
    #    print(f"Year {year}: Collecting filtered data...")
    #    
    #    # We must collect before we can get an accurate length or sample
    #    try:
    #        yearly_df = yearly_lf.collect()
    #    except Exception as e:
    #        print(f"  Error collecting data for {year}: {e}", file=sys.stderr)
    #        continue
    #        
    #    available_rows = len(yearly_df)

    #    if available_rows == 0:
    #        print(f"Year {year}: No data available, skipping.")
    #        continue

    #    actual_sample_size = min(args.sample_size, available_rows)

    #    sampled_df = yearly_df.sample(
    #        n=actual_sample_size, 
    #        shuffle=True, 
    #        seed=args.seed
    #    )

    #    sampled_train_yearly_dfs[year] = sampled_df
    #    print(f"  Sampled {actual_sample_size:,} rows (from {available_rows:,} available)")

    ## --- 5. Saving Results ---
    #print(f"\n--- Saving Sampled DataFrames for Years: {args.years_to_save} ---")
    #
    #output_dir = Path(args.output_dir)
    #output_dir.mkdir(parents=True, exist_ok=True)
    #print(f"Saving files to: {output_dir.resolve()}")
    #
    #saved_count = 0
    #for year in args.years_to_save:
    #    if year in sampled_train_yearly_dfs:
    #        df_to_save = sampled_train_yearly_dfs[year]
    #        output_path = output_dir / f"{args.output_prefix}_{year}.parquet"
    #        try:
    #            df_to_save.write_parquet(output_path)
    #            print(f"  Successfully saved: {output_path} ({len(df_to_save):,} rows)")
    #            saved_count += 1
    #        except Exception as e:
    #            print(f"  Error saving file {output_path}: {e}", file=sys.stderr)
    #    else:
    #        print(f"  Warning: No sampled data found for year {year}. Nothing to save.")

    #print(f"\nProcessing complete. Saved {saved_count} file(s).")
    #return 0


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
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the output parquet files. (Default: current directory)"
    )
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
    arg_parser = setup_argparser()
    args = arg_parser.parse_args()
    
    sys.exit(main(args))