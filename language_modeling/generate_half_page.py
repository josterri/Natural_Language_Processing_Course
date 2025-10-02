from ngram_model import NGramModel
import os
from datetime import datetime

print("=" * 70)
print("5-Gram Model: Half Page Generation (~200 words)")
print("=" * 70)

print("\nLoading model...")
model = NGramModel.load('models/5gram_extended.pkl')

stats = model.get_ngram_stats()
print(f"Model vocabulary: {stats['vocab_size']} words")
print(f"Unique contexts: {stats['unique_contexts']:,}")

print("\n" + "=" * 70)
print("Generating text (target: 200 words)")
print("=" * 70)

for i in range(3):
    print(f"\n--- Sample {i+1} ---\n")

    generated_text = model.generate(max_words=200)
    word_count = len(generated_text.split())

    print(f"[{word_count} words generated]\n")
    print(generated_text)
    print()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"samples/sample_{timestamp}_{i+1}.txt"
    os.makedirs('samples', exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"5-Gram Generated Text Sample\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Word count: {word_count}\n")
        f.write(f"Model: 5gram_extended.pkl\n")
        f.write(f"\n{'=' * 70}\n\n")
        f.write(generated_text)

    print(f"Saved to: {filename}")
    print()

print("=" * 70)
print("Generation Complete!")
print("=" * 70)
