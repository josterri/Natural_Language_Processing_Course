import numpy as np
import json
from sentence_transformers import SentenceTransformer
import argparse

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query, model, embeddings, headlines, categories, top_k=5):
    query_embedding = model.encode(query, convert_to_numpy=True)

    similarities = []
    for i, emb in enumerate(embeddings):
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((i, sim, headlines[i], categories[i]))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]

def main():
    parser = argparse.ArgumentParser(description='Semantic search in headlines')
    parser.add_argument('--query', type=str, help='Search query')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results (default: 5)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')

    args = parser.parse_args()

    print("=" * 70)
    print("Semantic Search - Extended Headlines Dataset")
    print("=" * 70)

    print("\nLoading embeddings and metadata...")
    embeddings = np.load('data/headlines_embeddings.npy')

    with open('data/embeddings_metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    headlines = metadata['headlines']
    categories = metadata['categories']

    print(f"  Loaded {len(embeddings):,} embeddings")
    print(f"  Embedding dimension: {metadata['embedding_dimension']}")

    print("\nLoading sentence-transformers model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("  Model loaded!")

    if args.interactive or not args.query:
        print("\n" + "=" * 70)
        print("Interactive Semantic Search")
        print("=" * 70)
        print("Enter your search query, or 'quit' to exit.\n")

        example_queries = [
            "president announces new policy",
            "team wins championship",
            "technology innovation breakthrough",
            "celebrity movie premiere",
            "economic crisis finance"
        ]

        print("Example queries:")
        for i, example in enumerate(example_queries, 1):
            print(f"  {i}. {example}")
        print()

        while True:
            query = input("Search query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if not query:
                continue

            print(f"\nSearching for: '{query}'")
            print("-" * 70)

            results = search(query, model, embeddings, headlines, categories, args.top_k)

            for rank, (idx, similarity, headline, category) in enumerate(results, 1):
                print(f"\n{rank}. [Score: {similarity:.4f}] [{category}]")
                print(f"   {headline}")

            print()

    else:
        print(f"\nSearching for: '{args.query}'")
        print("-" * 70)

        results = search(args.query, model, embeddings, headlines, categories, args.top_k)

        for rank, (idx, similarity, headline, category) in enumerate(results, 1):
            print(f"\n{rank}. [Score: {similarity:.4f}] [{category}]")
            print(f"   {headline}")

        print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
