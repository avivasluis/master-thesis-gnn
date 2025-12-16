import argparse
import polars as pl
import os

def expand_train_data_with_db_field(expanded_train_foreign_keys, foreign_key, dimention_table, data_column, name_dimention_table, verbose=False):
    temp_df = (
    expanded_train_foreign_keys.select(['node_id', foreign_key])
    .explode(foreign_key)
    .join(
        dimention_table.select([foreign_key, data_column]),
        on = foreign_key,
        how ='left'
    )
    .group_by('node_id', maintain_order=True)
    .agg(
        pl.implode(data_column).alias(f'{name_dimention_table}_{data_column}')
        )
    )

    #print('*'*50)
    #print(f'\n TEMP_DF -> \n {temp_df.collect()}')

    # Join the 'product_titles' column back to the original dataframe
    result = expanded_train_foreign_keys.join(
        temp_df,
        on='node_id',
        how='left'
    ).sort('node_id')

    if verbose:
        print('*'*50)
        print(f'\n RESULT_DF -> \n {result.head().collect()}')

    return result


def save_data_csv(df ,data_name, output_base_path):
    output_dir = os.path.join(output_base_path, data_name)
    os.makedirs(output_dir, exist_ok = True)
    output_path = os.path.join(output_dir, 'expanded_train.csv')

    df = df.collect().to_pandas()
    df.to_csv(output_path, index = False)


def save_data_parquet(df ,data_name, output_base_path):
    output_dir = os.path.join(output_base_path, data_name)
    os.makedirs(output_dir, exist_ok = True)
    output_path = os.path.join(output_dir, 'expanded_train.parquet')
    preview_path = os.path.join(output_dir, 'expanded_train.txt')

    df = df.collect().to_pandas()
    df.to_parquet(output_path, index = False)
    
    # Save preview of first 50 rows as txt log (formatted like console output)
    with open(preview_path, 'w', encoding='utf-8') as f:
        f.write(f"Preview of {data_name} (first 50 rows)\n")
        f.write("="*80 + "\n\n")
        f.write(str(df.head(50)))
        f.write(f"\n\n{'='*80}\n")
        f.write(f"Total rows: {len(df)}\n")
        f.write(f"Total columns: {len(df.columns)}\n")


def return_train_section(train_lazy, sample_size):
    train_lazy_sorted = train_lazy.sort('timestamp').tail(sample_size).with_row_index('node_id')
    return train_lazy_sorted


def collect_foreign_keys(train_data, fact_table, train_primary_key, foreign_key, fact_timestamp, training_data_timestamp, time_window):
    expanded_train_df_foreign_keys = train_data.join(
        fact_table, on = train_primary_key, how = 'left', maintain_order = 'left'
    ).group_by(
        train_data.collect_schema().names()  # Group by all original columns from training data
    ).agg(
        pl.col(foreign_key).filter(
            (pl.col(fact_timestamp) < pl.col(training_data_timestamp)) &
            (pl.col(fact_timestamp) >= pl.col(training_data_timestamp).dt.offset_by(time_window))
        ).alias(foreign_key)
    ).sort('node_id')

    return expanded_train_df_foreign_keys


def main(args):   
    if args.generate_train_section:
        if args.verbose:
            print("Generating a subsection of the training file...")
        train_lazy = return_train_section(args.train_lazy, args.sample_size)

        train_collected = train_lazy.collect()
        train_collected.write_parquet(os.path.join(args.output_path, 'train_section.parquet'))
        train_collected.write_csv(os.path.join(args.output_path, 'train_section.csv'))

        if args.verbose:
            print(f'\n TRAINING DATA: \n {train_collected} \n')
            print('... done')
    else:
        train_lazy = args.train_lazy

    # Base join for product table where I collect the list of foreign keys in fact table inside the time_window!
    product_product_id = collect_foreign_keys(train_lazy, args.review_lazy, 'customer_id', 'product_id', 'review_time', 'timestamp', args.time_window)

    if args.verbose:
        print('*'*50)
        print(f'\n PRODUCT_PRODUCT_ID -> \n {product_product_id.head().collect()}')
    save_data_parquet(product_product_id, 'product_id', args.output_path)

    data_columns = ['title', 'brand', 'description', 'price', 'category']
    data_columns = ['brand', 'price', 'category']
    data_columns = ['price']
    expanded_train_foreign_keys = product_product_id
    foreign_key = 'product_id'
    dimention_table = args.product_lazy
    name_dimention_table = 'product'

    for data_column in data_columns:
        expanded_df = expand_train_data_with_db_field(expanded_train_foreign_keys, foreign_key, dimention_table, data_column, name_dimention_table, args.verbose)
        save_data_parquet(expanded_df, f'{name_dimention_table}_{data_column}', args.output_path)
        

    # Base join for review table where I collect the list of foreign keys in the fact table inside the time_window!
    review_review_id = collect_foreign_keys(train_lazy, args.review_lazy, 'customer_id', 'review_id', 'review_time', 'timestamp', args.time_window)
    if args.verbose:
        print('*'*50)
        print(f'\n review_review_id -> \n {review_review_id.head().collect()}')
    save_data_parquet(review_review_id, 'review_id', args.output_path)

    #data_columns = ['review_text', 'summary', 'rating', 'verified']
    data_columns = ['rating']
        
    expanded_train_foreign_keys = review_review_id
    foreign_key = 'review_id'
    dimention_table = args.review_lazy
    name_dimention_table = 'review'

    for data_column in data_columns:
        expanded_df = expand_train_data_with_db_field(expanded_train_foreign_keys, foreign_key, dimention_table, data_column, name_dimention_table, args.verbose)
        save_data_parquet(expanded_df, f'{name_dimention_table}_{data_column}', args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process raw relational data into an intermediate format for graph construction.")
    parser.add_argument('output_path', type=str, help="Directory for output data")
    parser.add_argument('time_window', type=str, help="Time window for the processed data")    
    parser.add_argument('--base_data_path', type=str, default=r'data\1_raw', help="Directory base for /rel-amazon/")
    parser.add_argument('--training_data_path', type=str, default=r' ', help="Directory base for training files")
    parser.add_argument('--generate_train_section', action='store_true', help="Flag to indicate whether to generate new section of the training data")
    parser.add_argument('--sample_size', type=int, default=5000, help="Number of samples to process from the train set")
    parser.add_argument('--verbose', action='store_true', help="Flag to enable verbose output with print statements")
    args = parser.parse_args()

    review_file = os.path.join(args.base_data_path, 'rel-amazon', 'db', 'review.parquet')
    product_file = os.path.join(args.base_data_path, 'rel-amazon', 'db', 'product.parquet')
    customer_file = os.path.join(args.base_data_path, 'rel-amazon', 'db', 'customer.parquet')

    args.review_lazy = pl.scan_parquet(review_file).with_row_index('review_id')
    args.product_lazy = pl.scan_parquet(product_file)
    args.customer_lazy = pl.scan_parquet(customer_file)
    args.train_lazy = pl.scan_parquet(args.training_data_path).with_row_index('node_id')

    main(args)