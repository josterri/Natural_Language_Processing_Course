import pandas as pd
import os
from ngram_model import NGramModel

def main():
    print("=" * 70)
    print("5-Gram Language Model Training")
    print("=" * 70)

    dataset_path = '../extended/news_headlines_extended.csv'
    model_save_path = 'models/5gram_extended.pkl'

    print(f"\nLoading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} headlines")
    print(f"Categories: {df['category'].unique()}")

    headlines = df['headline'].tolist()

    print("\n" + "=" * 70)
    print("Training Model")
    print("=" * 70)

    model = NGramModel(n=5, smoothing_k=0.01)
    model.train(headlines, verbose=True)

    print("\n" + "=" * 70)
    print("Model Statistics")
    print("=" * 70)
    stats = model.get_ngram_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 70)
    print("Top 20 Most Frequent 5-Grams")
    print("=" * 70)
    top_ngrams = model.get_top_ngrams(k=20)
    for i, (context, word, count) in enumerate(top_ngrams, 1):
        context_str = ' '.join(context)
        print(f"{i:2d}. [{context_str}] -> '{word}': {count}")

    print("\n" + "=" * 70)
    print("Sample Probabilities")
    print("=" * 70)

    sample_contexts = [
        ('president', 'announces', 'new', 'reform'),
        ('team', 'wins', 'championship', 'after'),
        ('company', 'launches', 'new', 'device'),
    ]

    for context in sample_contexts:
        if context in model.ngrams:
            print(f"\nContext: {' '.join(context)}")
            word_probs = [(word, model._get_probability(context, word))
                         for word in model.ngrams[context].keys()]
            word_probs.sort(key=lambda x: x[1], reverse=True)
            for word, prob in word_probs[:5]:
                print(f"  P({word} | context) = {prob:.4f}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)

    print("\n" + "=" * 70)
    print("Quick Generation Test")
    print("=" * 70)
    sample_text = model.generate(max_words=50)
    print(f"\nGenerated text (50 words):")
    print(sample_text)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    main()
