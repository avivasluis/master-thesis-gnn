# This script will take raw data paths as input and write the intermediate CSVs to data/2_intermediate/.

import os
import argparse
import polars as pl   
import pandas as pd
import numpy as np

def generate_intermediate_train_table(db_path, tasks_path, time_window, n_examples ):
    """
    Processes the raw Amazon dataset for the user churn task.

    This function loads the raw parquet files, joins the training data with
    review history, aggregates past purchases within a specific time window.

    Args:
        db_path (str): Path to the directory containing the database parquet files
                       (customer.parquet, product.parquet, review.parquet).
        tasks_path (str): Path to the directory for the 'user-churn' task,
                          containing train.parquet, test.parquet, etc.
        output_path (str): Path to save the output intermediate CSV file.
        time_window (str): A string representing the look-back period for aggregating
                           past purchases (e.g., '-6mo', '-1y', '-90d').
    """
    # Define file paths
    review_file = os.path.join(db_path, 'review.parquet')
    train_file = os.path.join(tasks_path, 'user-churn', 'train.parquet')

    # Lazily scan the parquet files to save memory
    review_lzydf = pl.scan_parquet(review_file)
    train_lzydf = pl.scan_parquet(train_file)

    #print("Joining training data with review history...")
    # Join train data with all reviews for each customer
    expanded_train_lazy = train_lzydf.join(
        review_lzydf, on='customer_id', how='left'
    ).group_by(
        train_lzydf.collect_schema().names()  # Group by all original columns from train_df
    ).agg(
        # For each customer, collect a list of product_ids from reviews
        # that occurred *before* the churn prediction timestamp
        # and within the specified time_window.
        pl.col('product_id').filter(
            (pl.col('review_time') < pl.col('timestamp')) &
            (pl.col('review_time') >= pl.col('timestamp').dt.offset_by(time_window))
        ).alias('product_id')
    )

    expanded_train_df = expanded_train_lazy.collect().head(n_examples).to_pandas()
    
    return expanded_train_df


def process_categories(category_list, words_to_remove = {'Books', '&', '&amp', '&amp;', 'Genre', 'Up', 'of','by'}):
    """
    Processes a list of category strings by splitting them into single words,
    removing specified words and ensuring all words are unique
    """

    # Return an empty list for missing or non-list data
    if not isinstance(category_list, (list, np.ndarray)):
        return []
    
    # Step 1: Split all items into single words
    all_words = []
    for item in category_list:
        if isinstance(item,str):
            all_words.extend(item.replace("'", "").replace('"', '').replace(',', '').replace('Thrillers', 'Thriller').split(' '))
    
    # Step 2: Remove unwanted words and duplicates
    unique_words = []
    seen = set()

    for word in all_words:
        # Process only if the word is not empty, not in the removal list,
        # and not already added to our unique list for this row
        if word and word not in words_to_remove and word not in seen:
            unique_words.append(word)
            seen.add(word)
    
    return unique_words


def generate_product_category_df(db_path, output_path, expanded_train_df):
    product_file = os.path.join(db_path, 'product.parquet')
    product_lzydf = pl.scan_parquet(product_file)

    product_id_series = expanded_train_df['product_id']
    # print('product_id_series', product_id_series)
    # print()
    # print("Data type of product_id_series:", product_id_series.dtype)
    # print()
    # print(f'product_id_series[0]: {product_id_series[0]}')
    # print()
    # print(f'type(product_id_series[0]): {type(product_id_series[0])}')
    # print()

    unique_ids_series = pd.Series(
        product_id_series.explode().dropna().unique()
    ).astype('int64').sort_values().reset_index(drop=True)

    # print(f'unique_ids_series: {unique_ids_series}')

    # print()
    # print('*'*50)
    product_category = product_lzydf.filter(pl.col('product_id').is_in(unique_ids_series)).collect().to_pandas()[['product_id', 'category']]
    # print(f'product_category: {product_category}')


    # print()
    # print('*'*50)
    product_category['category'] = product_category['category'].apply(process_categories)
    # print(f'product_category: {product_category}')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    product_category.to_csv(output_path, index=False)
    
    return product_category


def join_categories_with_train_df(product_ids, category_df):
    if product_ids is None:
        return []

    single_list = []

    for id in product_ids:
        try:
            categories_list = category_df[category_df['product_id'] == id]['category'].iloc[0]
            single_list.extend(categories_list)
        except IndexError:
            # This can happen if a product_id from the history is not in our product_category table.
            # We will just skip it.
            pass

    return list(set(single_list))


def generate_final_train_table(expanded_train_df, product_category, output_path):
    expanded_train_df['categories_expanded'] = expanded_train_df['product_id'].apply(lambda x: join_categories_with_train_df(x, product_category))

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    expanded_train_df.to_csv(output_path)
    
    print('*'*50)
    print(f'expanded_train_df: {expanded_train_df}')


def main():
    parser = argparse.ArgumentParser(description="Process raw relational data into an intermediate format for graph construction.")
    parser.add_argument('--base_data_path', type=str, default=r'data\1_raw', help="Name of the dataset to process (e.g., 'rel-amazon').")
    parser.add_argument('--output_base_path', type=str, default=r'data\2_intermediate', help="Name of the predictive task")
    parser.add_argument('--time_window', type=str, default='-6mo', help="Time window for aggregating historical data (e.g., '-3mo', '-1y').")
    parser.add_argument('--n_examples', type=int, default=5, help="Number of examples to generate")
    args = parser.parse_args()

    args.dataset = 'rel-amazon'
    args.task = 'user-churn'
    
    db_path = os.path.join(args.base_data_path, args.dataset, 'db')
    tasks_path = os.path.join(args.base_data_path, args.dataset, 'tasks')
    output_path_product = os.path.join(args.output_base_path, args.dataset, 'db', 'productId_processed_category.csv')
    output_path_training_table = os.path.join(args.output_base_path, args.dataset, 'tasks', args.task,'single_categories_string.csv')

    expanded_train_df = generate_intermediate_train_table(db_path, tasks_path, time_window = args.time_window, n_examples = args.n_examples)
    # print(f'Expanded train df: {expanded_train_df}')
    # print()

    product_category_df = generate_product_category_df(db_path, output_path_product, expanded_train_df)

    generate_final_train_table(expanded_train_df, product_category_df, output_path_training_table)
    
    
if __name__ == '__main__':
    main()