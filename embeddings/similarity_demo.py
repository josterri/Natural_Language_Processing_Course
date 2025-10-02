import numpy as np
import json
import pandas as pd

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    print("=" * 70)
    print("Headline Similarity Analysis")
    print("=" * 70)

    print("\nLoading embeddings and metadata...")
    embeddings = np.load('data/headlines_embeddings.npy')

    with open('data/embeddings_metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    headlines = metadata['headlines']
    categories = metadata['categories']

    print(f"  Loaded {len(embeddings):,} embeddings")

    # 1. Show pairwise similarities for sample headlines
    print("\n" + "=" * 70)
    print("1. Sample Pairwise Similarities")
    print("=" * 70)

    sample_indices = [0, 1, 10, 100, 500]

    print("\nComparing 5 sample headlines:")
    for i in sample_indices:
        print(f"  {i}: [{categories[i]}] {headlines[i]}")

    print("\nSimilarity Matrix:")
    print("     ", end="")
    for i in sample_indices:
        print(f"{i:6d} ", end="")
    print()

    for i, idx_i in enumerate(sample_indices):
        print(f"{idx_i:4d}: ", end="")
        for j, idx_j in enumerate(sample_indices):
            sim = cosine_similarity(embeddings[idx_i], embeddings[idx_j])
            print(f"{sim:6.3f} ", end="")
        print()

    # 2. Within-category vs Between-category similarities
    print("\n" + "=" * 70)
    print("2. Within-Category vs Between-Category Similarities")
    print("=" * 70)

    # Sample 100 random headlines from each category
    df = pd.DataFrame({'headline': headlines, 'category': categories})

    category_similarities = {}
    between_category_sims = []

    for cat in df['category'].unique():
        cat_indices = df[df['category'] == cat].index[:100].tolist()

        if len(cat_indices) < 2:
            continue

        within_sims = []
        for i in range(len(cat_indices)):
            for j in range(i+1, len(cat_indices)):
                sim = cosine_similarity(embeddings[cat_indices[i]],
                                      embeddings[cat_indices[j]])
                within_sims.append(sim)

        category_similarities[cat] = within_sims

        # Between-category similarities
        other_cats = df[df['category'] != cat].index[:50].tolist()
        for i in cat_indices[:50]:
            for j in other_cats[:10]:
                sim = cosine_similarity(embeddings[i], embeddings[j])
                between_category_sims.append(sim)

    print("\nWithin-category average similarities:")
    for cat, sims in category_similarities.items():
        print(f"  {cat:15s}: {np.mean(sims):.4f} (std: {np.std(sims):.4f})")

    print(f"\nBetween-category average similarity: {np.mean(between_category_sims):.4f}")
    print(f"  (std: {np.std(between_category_sims):.4f})")

    # 3. Find most similar and most dissimilar pairs
    print("\n" + "=" * 70)
    print("3. Most Similar and Most Dissimilar Headlines")
    print("=" * 70)

    # Sample 500 random pairs for efficiency
    np.random.seed(42)
    n_samples = 500
    pairs = []

    for _ in range(n_samples):
        i, j = np.random.choice(len(embeddings), size=2, replace=False)
        sim = cosine_similarity(embeddings[i], embeddings[j])
        pairs.append((i, j, sim))

    pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 5 Most Similar Pairs:")
    for rank, (i, j, sim) in enumerate(pairs[:5], 1):
        print(f"\n{rank}. Similarity: {sim:.4f}")
        print(f"   A [{categories[i]}]: {headlines[i]}")
        print(f"   B [{categories[j]}]: {headlines[j]}")

    print("\n\nTop 5 Most Dissimilar Pairs:")
    for rank, (i, j, sim) in enumerate(pairs[-5:], 1):
        print(f"\n{rank}. Similarity: {sim:.4f}")
        print(f"   A [{categories[i]}]: {headlines[i]}")
        print(f"   B [{categories[j]}]: {headlines[j]}")

    # 4. Find nearest neighbors for specific headlines
    print("\n" + "=" * 70)
    print("4. Nearest Neighbors Example")
    print("=" * 70)

    query_indices = [0, 100, 500]

    for query_idx in query_indices:
        print(f"\nQuery headline: [{categories[query_idx]}]")
        print(f"  {headlines[query_idx]}")

        neighbors = []
        for i in range(len(embeddings)):
            if i == query_idx:
                continue
            sim = cosine_similarity(embeddings[query_idx], embeddings[i])
            neighbors.append((i, sim))

        neighbors.sort(key=lambda x: x[1], reverse=True)

        print("\n  Top 3 Nearest Neighbors:")
        for rank, (i, sim) in enumerate(neighbors[:3], 1):
            print(f"    {rank}. [{categories[i]}] {headlines[i]} (sim: {sim:.4f})")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
