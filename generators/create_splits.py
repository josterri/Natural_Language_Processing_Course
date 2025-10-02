"""
Create train/validation/test splits for extended headlines and articles
Uses stratified sampling to maintain balanced category distributions
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

# Random seed for reproducibility
RANDOM_SEED = 42

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def create_stratified_splits(df, output_prefix, dataset_name):
    """
    Create stratified train/val/test splits for a dataset

    Args:
        df: DataFrame to split
        output_prefix: Directory prefix for output files
        dataset_name: Name of dataset for logging
    """
    print(f"\nCreating splits for {dataset_name}...")
    print(f"  Total samples: {len(df)}")

    # First split: separate out test set (15%)
    train_val, test = train_test_split(
        df,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=df['category']
    )

    # Second split: separate train and validation from remaining 85%
    # validation is 15% of total, so it's 15/85 of the train_val set
    val_size = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=RANDOM_SEED,
        stratify=train_val['category']
    )

    # Print statistics
    print(f"  Train: {len(train)} ({len(train)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val)} ({len(val)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test)} ({len(test)/len(df)*100:.1f}%)")

    # Verify category distributions
    print(f"\n  Category distribution in splits:")
    print(f"    Train: {dict(train['category'].value_counts().sort_index())}")
    print(f"    Val:   {dict(val['category'].value_counts().sort_index())}")
    print(f"    Test:  {dict(test['category'].value_counts().sort_index())}")

    # Save splits
    train_file = f"{output_prefix}/train.csv"
    val_file = f"{output_prefix}/val.csv"
    test_file = f"{output_prefix}/test.csv"

    print(f"\n  Saving splits...")
    train.to_csv(train_file, index=False)
    val.to_csv(val_file, index=False)
    test.to_csv(test_file, index=False)

    print(f"    {train_file}")
    print(f"    {val_file}")
    print(f"    {test_file}")

    return train, val, test


def verify_no_leakage(train, val, test, id_column):
    """Verify no data leakage between splits"""
    train_ids = set(train[id_column])
    val_ids = set(val[id_column])
    test_ids = set(test[id_column])

    # Check for overlaps
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("  WARNING: Data leakage detected!")
        if train_val_overlap:
            print(f"    Train-Val overlap: {len(train_val_overlap)} samples")
        if train_test_overlap:
            print(f"    Train-Test overlap: {len(train_test_overlap)} samples")
        if val_test_overlap:
            print(f"    Val-Test overlap: {len(val_test_overlap)} samples")
        return False
    else:
        print("  No data leakage detected - splits are clean")
        return True


def main():
    print("=" * 70)
    print("TRAIN/VALIDATION/TEST SPLITS CREATOR")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Split ratios: Train={TRAIN_RATIO:.0%}, Val={VAL_RATIO:.0%}, Test={TEST_RATIO:.0%}")

    # 1. Create splits for extended headlines
    print("\n" + "=" * 70)
    print("EXTENDED HEADLINES")
    print("=" * 70)

    headlines_file = '../extended/news_headlines_extended.csv'
    print(f"Loading {headlines_file}...")
    df_headlines = pd.read_csv(headlines_file)

    train_h, val_h, test_h = create_stratified_splits(
        df_headlines,
        '../extended',
        'Extended Headlines'
    )

    print("\nVerifying no data leakage...")
    verify_no_leakage(train_h, val_h, test_h, 'headline_id')

    # 2. Create splits for articles
    print("\n" + "=" * 70)
    print("ARTICLES")
    print("=" * 70)

    articles_file = '../articles/news_articles_dataset.csv'
    print(f"Loading {articles_file}...")
    df_articles = pd.read_csv(articles_file)

    train_a, val_a, test_a = create_stratified_splits(
        df_articles,
        '../articles',
        'Articles'
    )

    print("\nVerifying no data leakage...")
    verify_no_leakage(train_a, val_a, test_a, 'article_id')

    # Final summary
    print("\n" + "=" * 70)
    print("SPLIT CREATION COMPLETE")
    print("=" * 70)

    print("\nExtended Headlines:")
    print(f"  Total: {len(df_headlines):,} headlines")
    print(f"  Train: {len(train_h):,} headlines")
    print(f"  Val:   {len(val_h):,} headlines")
    print(f"  Test:  {len(test_h):,} headlines")

    print("\nArticles:")
    print(f"  Total: {len(df_articles):,} articles")
    print(f"  Train: {len(train_a):,} articles")
    print(f"  Val:   {len(val_a):,} articles")
    print(f"  Test:  {len(test_a):,} articles")

    print("\nAll splits have been created with balanced category distributions.")
    print("Use train sets for model training, validation sets for hyperparameter")
    print("tuning, and test sets for final evaluation.")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
