from ngram_model import NGramModel
import sys

print("Loading model...")
try:
    model = NGramModel.load('models/5gram_extended.pkl')
    print("Model loaded successfully!")

    stats = model.get_ngram_stats()
    print(f"\nModel stats: {stats}")

    print("\nAttempting to generate 20 words...")
    sys.stdout.flush()

    generated = model.generate(max_words=20)
    print(f"Generated: {generated}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
