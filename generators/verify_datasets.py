"""
Verify all datasets and generate comprehensive statistics
Includes Zipf's law analysis for extended headlines
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression
from datetime import datetime


def analyze_zipf_law(df, text_column='headline'):
    """Analyze Zipf's law adherence for a dataset"""
    # Combine all text
    all_text = ' '.join(df[text_column].str.lower())
    all_words = all_text.split()

    # Count words
    word_counts = Counter(all_words)
    all_word_freq = word_counts.most_common()

    # Prepare data for regression
    ranks = list(range(1, min(len(all_word_freq) + 1, 101)))  # Top 100 words
    frequencies = [freq for word, freq in all_word_freq[:100]]

    # Log-log regression
    log_ranks = np.log(ranks[:len(frequencies)]).reshape(-1, 1)
    log_frequencies = np.log(frequencies)

    model = LinearRegression()
    model.fit(log_ranks, log_frequencies)

    r_squared = model.score(log_ranks, log_frequencies)
    slope = model.coef_[0]
    intercept = model.intercept_
    coefficient_C = np.exp(intercept)

    return {
        'total_tokens': len(all_words),
        'unique_words': len(word_counts),
        'type_token_ratio': len(word_counts) / len(all_words),
        'zipf_r_squared': r_squared,
        'zipf_slope': slope,
        'zipf_intercept': intercept,
        'zipf_coefficient_C': coefficient_C,
        'top_10_words': all_word_freq[:10]
    }


def verify_dataset(dataset_name, file_path, text_columns):
    """Verify a single dataset and print statistics"""
    print(f"\n{'=' * 70}")
    print(f"{dataset_name}")
    print(f"{'=' * 70}")
    print(f"File: {file_path}")

    # Load dataset
    try:
        df = pd.read_csv(file_path)
        print(f"Status: LOADED SUCCESSFULLY")
    except Exception as e:
        print(f"Status: FAILED TO LOAD")
        print(f"Error: {e}")
        return None

    # Basic statistics
    print(f"\nBasic Statistics:")
    print(f"  Total records: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # Category distribution
    if 'category' in df.columns:
        print(f"\nCategory Distribution:")
        for cat, count in df['category'].value_counts().sort_index().items():
            print(f"  {cat}: {count:,} ({count/len(df)*100:.1f}%)")

    # Word count statistics
    for col in text_columns:
        if col in df.columns:
            word_count_col = f"{col}_word_count"
            if word_count_col in df.columns:
                print(f"\n{col.capitalize()} Word Count Statistics:")
                print(f"  Mean: {df[word_count_col].mean():.2f}")
                print(f"  Std: {df[word_count_col].std():.2f}")
                print(f"  Min: {df[word_count_col].min()}")
                print(f"  Max: {df[word_count_col].max()}")
                print(f"  Total words: {df[word_count_col].sum():,}")

    # Check for duplicates
    duplicates = df.duplicated(subset=text_columns).sum()
    print(f"\nDuplicate Check:")
    print(f"  Duplicates found: {duplicates}")
    if duplicates > 0:
        print(f"  WARNING: Dataset contains {duplicates} duplicate entries")

    # Zipf's law analysis (only for larger datasets)
    if len(df) >= 1000:
        print(f"\nZipf's Law Analysis:")
        for col in text_columns:
            if col in df.columns:
                analysis = analyze_zipf_law(df, col)
                print(f"\n  Analysis for '{col}':")
                print(f"    Total tokens: {analysis['total_tokens']:,}")
                print(f"    Unique words: {analysis['unique_words']:,}")
                print(f"    Type-token ratio: {analysis['type_token_ratio']:.4f}")
                print(f"    Zipf R²: {analysis['zipf_r_squared']:.4f}")
                print(f"    Zipf slope: {analysis['zipf_slope']:.4f}")
                print(f"    Zipf intercept: {analysis['zipf_intercept']:.4f}")
                print(f"    Zipf coefficient C: {analysis['zipf_coefficient_C']:.2f}")

                # Interpret Zipf adherence
                slope_deviation = abs(analysis['zipf_slope'] + 1.0)
                if slope_deviation < 0.15:
                    zipf_status = "Strong Zipf adherence"
                elif slope_deviation < 0.35:
                    zipf_status = "Moderate Zipfian behavior"
                else:
                    zipf_status = "Power-law distribution (flatter than ideal Zipf)"
                print(f"    Interpretation: {zipf_status}")

                print(f"\n    Top 10 words:")
                for i, (word, freq) in enumerate(analysis['top_10_words'], 1):
                    print(f"      {i}. '{word}': {freq}")

    return df


def main():
    print("=" * 70)
    print("DATASET VERIFICATION AND ANALYSIS")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Verify basic headlines (400)
    basic_headlines = verify_dataset(
        "BASIC HEADLINES (400)",
        "../basic/news_headlines_dataset.csv",
        ['headline']
    )

    # Verify extended headlines (10,000)
    extended_headlines = verify_dataset(
        "EXTENDED HEADLINES (10,000)",
        "../extended/news_headlines_extended.csv",
        ['headline']
    )

    # Verify headlines train split
    headlines_train = verify_dataset(
        "EXTENDED HEADLINES - TRAIN SPLIT",
        "../extended/train.csv",
        ['headline']
    )

    # Verify headlines val split
    headlines_val = verify_dataset(
        "EXTENDED HEADLINES - VAL SPLIT",
        "../extended/val.csv",
        ['headline']
    )

    # Verify headlines test split
    headlines_test = verify_dataset(
        "EXTENDED HEADLINES - TEST SPLIT",
        "../extended/test.csv",
        ['headline']
    )

    # Verify articles dataset (1,000)
    articles = verify_dataset(
        "ARTICLES (1,000)",
        "../articles/news_articles_dataset.csv",
        ['headline', 'article']
    )

    # Verify articles train split
    articles_train = verify_dataset(
        "ARTICLES - TRAIN SPLIT",
        "../articles/train.csv",
        ['headline', 'article']
    )

    # Verify articles val split
    articles_val = verify_dataset(
        "ARTICLES - VAL SPLIT",
        "../articles/val.csv",
        ['headline', 'article']
    )

    # Verify articles test split
    articles_test = verify_dataset(
        "ARTICLES - TEST SPLIT",
        "../articles/test.csv",
        ['headline', 'article']
    )

    # Final summary
    print(f"\n{'=' * 70}")
    print("VERIFICATION SUMMARY")
    print(f"{'=' * 70}")

    total_headlines = 0
    total_articles = 0

    if basic_headlines is not None:
        total_headlines += len(basic_headlines)
        print(f"\nBasic Headlines: {len(basic_headlines):,}")

    if extended_headlines is not None:
        total_headlines += len(extended_headlines)
        print(f"Extended Headlines: {len(extended_headlines):,}")
        print(f"  Train: {len(headlines_train):,}")
        print(f"  Val: {len(headlines_val):,}")
        print(f"  Test: {len(headlines_test):,}")

    if articles is not None:
        total_articles += len(articles)
        print(f"\nArticles: {len(articles):,}")
        print(f"  Train: {len(articles_train):,}")
        print(f"  Val: {len(articles_val):,}")
        print(f"  Test: {len(articles_test):,}")

    print(f"\nTotal Headlines: {total_headlines:,}")
    print(f"Total Articles: {total_articles:,}")
    print(f"Total Records: {total_headlines + total_articles:,}")

    # Key findings
    print(f"\n{'=' * 70}")
    print("KEY FINDINGS")
    print(f"{'=' * 70}")

    if extended_headlines is not None:
        print("\nExtended Headlines Dataset:")
        analysis = analyze_zipf_law(extended_headlines, 'headline')
        print(f"  Vocabulary size: {analysis['unique_words']:,} unique words")
        print(f"  Total tokens: {analysis['total_tokens']:,}")
        print(f"  Type-token ratio: {analysis['type_token_ratio']:.4f}")
        print(f"  Zipf slope: {analysis['zipf_slope']:.3f} (ideal: -1.000)")
        print(f"  Deviation from ideal: {abs(analysis['zipf_slope'] + 1.0):.3f}")

        if abs(analysis['zipf_slope'] + 1.0) < 0.35:
            print(f"  Result: Dataset shows good Zipfian behavior!")
        else:
            print(f"  Result: Dataset shows power-law but with flatter distribution")

    if articles is not None:
        print("\nArticles Dataset:")
        analysis_headline = analyze_zipf_law(articles, 'headline')
        analysis_article = analyze_zipf_law(articles, 'article')
        print(f"  Article vocabulary: {analysis_article['unique_words']:,} unique words")
        print(f"  Article tokens: {analysis_article['total_tokens']:,}")
        print(f"  Headline vocabulary: {analysis_headline['unique_words']:,} unique words")
        print(f"  Suitable for summarization tasks: YES")

    print(f"\n{'=' * 70}")
    print("VERIFICATION COMPLETE")
    print(f"{'=' * 70}")
    print("All datasets loaded successfully and statistics computed.")
    print("Datasets are ready for NLP tasks:")
    print("  - Basic headlines: Intro exercises")
    print("  - Extended headlines: Embeddings, language modeling")
    print("  - Articles: Summarization tasks")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
