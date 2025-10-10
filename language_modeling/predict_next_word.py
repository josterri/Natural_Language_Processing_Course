"""
Interactive Next-Word Prediction Tool

Compare next-word predictions from both 5-gram and LSTM models in real-time.

Usage:
    python predict_next_word.py
"""

import torch
from pathlib import Path
import sys

from ngram_model import NGramModel
from neural_lm import LSTMLanguageModel, Vocabulary


def get_ngram_predictions(model, context_str, top_k=5):
    """Get top k predictions from n-gram model."""
    tokens = context_str.lower().split()

    # Get context
    if len(tokens) < model.n - 1:
        context = tuple(['<START>'] * (model.n - 1 - len(tokens)) + tokens)
    else:
        context = tuple(tokens[-(model.n - 1):])

    # Get probabilities for all possible next words
    if context in model.ngrams:
        word_counts = model.ngrams[context]
        word_probs = [(word, model._get_probability(context, word))
                      for word in word_counts.keys()]
        word_probs.sort(key=lambda x: x[1], reverse=True)
        return word_probs[:top_k]
    else:
        return [("<no predictions>", 0.0)]


def get_lstm_predictions(model, vocab, context_str, top_k=5, device='cpu'):
    """Get top k predictions from LSTM model."""
    # Create vocabulary object
    vocab_obj = Vocabulary()
    vocab_obj.word2idx = vocab
    vocab_obj.idx2word = {v: k for k, v in vocab.items()}

    # Encode context
    tokens = ['<START>'] + context_str.lower().split()
    indices = vocab_obj.encode(tokens)

    # Convert to tensor
    context_tensor = torch.tensor([indices], dtype=torch.long).to(device)

    # Get predictions
    try:
        top_indices, top_probs = model.predict_next_word(context_tensor, top_k=top_k)

        # Decode
        predictions = [(vocab_obj.idx2word[idx], prob)
                       for idx, prob in zip(top_indices, top_probs)]
        return predictions
    except Exception as e:
        return [("<error>", 0.0)]


def print_predictions(context, ngram_preds, lstm_preds):
    """Print predictions in a nice format."""
    print("\n" + "=" * 80)
    print(f"Context: \"{context}\"")
    print("=" * 80)

    # Side-by-side comparison
    print(f"\n{'5-GRAM (Statistical)':<40} {'LSTM (Neural)':<40}")
    print("-" * 80)

    for i in range(max(len(ngram_preds), len(lstm_preds))):
        # 5-gram column
        if i < len(ngram_preds):
            word, prob = ngram_preds[i]
            bar = '█' * int(prob * 30)
            ngram_str = f"{i+1}. {word:12s} {prob:.4f} {bar}"
        else:
            ngram_str = ""

        # LSTM column
        if i < len(lstm_preds):
            word, prob = lstm_preds[i]
            bar = '█' * int(prob * 30)
            lstm_str = f"{i+1}. {word:12s} {prob:.4f} {bar}"
        else:
            lstm_str = ""

        print(f"{ngram_str:<40} {lstm_str:<40}")

    print("=" * 80)


def main():
    print("\n" + "=" * 80)
    print("INTERACTIVE NEXT-WORD PREDICTION")
    print("=" * 80)
    print("\nComparing 5-gram (Statistical) vs LSTM (Neural) language models")
    print("Course Focus: Predicting the Next Word")
    print("\n" + "-" * 80)

    # Load models
    print("\nLoading models...")

    # Load 5-gram
    try:
        ngram_model = NGramModel.load('models/5gram_extended.pkl')
        print("✓ 5-gram model loaded")
    except Exception as e:
        print(f"✗ Failed to load 5-gram model: {e}")
        print("  Make sure 'models/5gram_extended.pkl' exists")
        sys.exit(1)

    # Load LSTM
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lstm_model, lstm_vocab = LSTMLanguageModel.load('models/lstm_lm', device=device)
        print(f"✓ LSTM model loaded (device: {device})")
    except Exception as e:
        print(f"✗ Failed to load LSTM model: {e}")
        print("  Please train the model first:")
        print("  python train_neural_lm.py")
        sys.exit(1)

    print("\n" + "-" * 80)
    print("\nReady! Type a context to see next-word predictions.")
    print("Examples:")
    print("  - The president will")
    print("  - New technology company")
    print("  - Scientists discover")
    print("\nType 'quit' or 'exit' to stop.")
    print("-" * 80)

    # Interactive loop
    while True:
        try:
            # Get user input
            context = input("\nYour context: ").strip()

            # Check for exit
            if context.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            # Skip empty input
            if not context:
                continue

            # Get predictions
            ngram_preds = get_ngram_predictions(ngram_model, context, top_k=5)
            lstm_preds = get_lstm_predictions(lstm_model, lstm_vocab, context,
                                             top_k=5, device=device)

            # Display
            print_predictions(context, ngram_preds, lstm_preds)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again with a different context.")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
