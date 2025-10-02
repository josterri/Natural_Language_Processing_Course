import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
import time
import os

def main():
    print("=" * 70)
    print("Generating Sentence Embeddings")
    print("Dataset: Extended Headlines (10,000 headlines)")
    print("Model: all-MiniLM-L6-v2 (sentence-transformers)")
    print("=" * 70)

    # Load dataset
    print("\n[1/5] Loading dataset...")
    dataset_path = '../extended/news_headlines_extended.csv'
    df = pd.read_csv(dataset_path)
    headlines = df['headline'].tolist()
    categories = df['category'].tolist()

    print(f"  Loaded {len(headlines):,} headlines")
    print(f"  Categories: {df['category'].unique().tolist()}")

    # Initialize model
    print("\n[2/5] Loading sentence-transformers model...")
    print("  Model: all-MiniLM-L6-v2")
    print("  Note: First run will download the model (~80 MB)")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_dim = model.get_sentence_embedding_dimension()

    print(f"  Model loaded successfully!")
    print(f"  Embedding dimension: {embedding_dim}")

    # Generate embeddings
    print("\n[3/5] Generating embeddings...")
    print("  This may take 1-2 minutes...")

    start_time = time.time()

    # Batch processing for efficiency
    batch_size = 32
    embeddings = model.encode(
        headlines,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    elapsed_time = time.time() - start_time

    print(f"\n  Generated {len(embeddings):,} embeddings")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Time elapsed: {elapsed_time:.2f} seconds")
    print(f"  Speed: {len(embeddings) / elapsed_time:.1f} headlines/second")

    # Save embeddings
    print("\n[4/5] Saving embeddings...")

    os.makedirs('data', exist_ok=True)

    embeddings_path = 'data/headlines_embeddings.npy'
    np.save(embeddings_path, embeddings)

    file_size_mb = os.path.getsize(embeddings_path) / (1024 * 1024)
    print(f"  Saved embeddings to: {embeddings_path}")
    print(f"  File size: {file_size_mb:.2f} MB")

    # Save metadata
    metadata = {
        'num_headlines': len(headlines),
        'embedding_dimension': int(embedding_dim),
        'model_name': 'all-MiniLM-L6-v2',
        'headlines': headlines,
        'categories': categories,
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_time_seconds': elapsed_time
    }

    metadata_path = 'data/embeddings_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Saved metadata to: {metadata_path}")

    # Calculate and display statistics
    print("\n[5/5] Embedding Statistics:")
    print("=" * 70)

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  Mean norm: {norms.mean():.4f}")
    print(f"  Std norm:  {norms.std():.4f}")
    print(f"  Min norm:  {norms.min():.4f}")
    print(f"  Max norm:  {norms.max():.4f}")

    # Sample similarity
    print("\n  Sample pairwise similarities:")
    for i in range(5):
        for j in range(i+1, min(i+3, 5)):
            sim = np.dot(embeddings[i], embeddings[j]) / (norms[i] * norms[j])
            print(f"    Headlines {i}-{j}: {sim:.4f}")
            print(f"      {headlines[i][:50]}...")
            print(f"      {headlines[j][:50]}...")

    print("\n" + "=" * 70)
    print("Embedding Generation Complete!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  1. {embeddings_path} ({file_size_mb:.2f} MB)")
    print(f"  2. {metadata_path}")
    print(f"\nYou can now:")
    print(f"  - Run semantic_search.py for semantic search")
    print(f"  - Run similarity_demo.py for similarity analysis")
    print(f"  - Open embedding_analysis.ipynb for visualization")

if __name__ == "__main__":
    main()
