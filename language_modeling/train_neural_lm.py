"""
Train Neural Language Model

Trains an LSTM-based language model on news headlines for next-word prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from neural_lm import LSTMLanguageModel, Vocabulary


class HeadlineDataset(Dataset):
    """Dataset for language modeling with news headlines."""

    def __init__(self, headlines: list, vocab: Vocabulary, seq_length: int = 20):
        self.vocab = vocab
        self.seq_length = seq_length
        self.samples = []

        # Process each headline
        for headline in headlines:
            tokens = ['<START>'] + headline.lower().split() + ['<END>']
            indices = vocab.encode(tokens)

            # Create input-target pairs with sliding window
            for i in range(len(indices) - 1):
                # Input: tokens up to position i
                # Target: token at position i+1
                if i < seq_length:
                    # Pad with <START> if needed
                    input_seq = [vocab.word2idx['<START>']] * (seq_length - i) + indices[:i+1]
                else:
                    input_seq = indices[i - seq_length + 1:i + 1]

                target = indices[i + 1]

                self.samples.append((input_seq, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target = self.samples[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def build_vocabulary(train_headlines: list) -> Vocabulary:
    """Build vocabulary from training headlines."""
    print("Building vocabulary...")
    vocab = Vocabulary()

    for headline in train_headlines:
        vocab.add_sentence(headline)

    print(f"Vocabulary size: {len(vocab)}")
    return vocab


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs, _ = model(inputs)

        # Get predictions for last position
        outputs = outputs[:, -1, :]  # (batch_size, vocab_size)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / total_samples
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs, _ = model(inputs)
            outputs = outputs[:, -1, :]

            # Calculate loss
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity


def plot_training_curves(train_losses, val_losses, save_path):
    """Plot and save training curves."""
    plt.figure(figsize=(12, 5))

    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Perplexity curves
    plt.subplot(1, 2, 2)
    train_perplexities = [np.exp(loss) for loss in train_losses]
    val_perplexities = [np.exp(loss) for loss in val_losses]
    plt.plot(train_perplexities, label='Train Perplexity', linewidth=2)
    plt.plot(val_perplexities, label='Val Perplexity', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Training and Validation Perplexity')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def main():
    # Configuration
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BATCH_SIZE = 64
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001
    SEQ_LENGTH = 20
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {DEVICE}")
    print("=" * 60)

    # Load data
    print("Loading datasets...")
    data_dir = Path(__file__).parent.parent / 'extended'
    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'val.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')

    train_headlines = train_df['headline'].tolist()
    val_headlines = val_df['headline'].tolist()
    test_headlines = test_df['headline'].tolist()

    print(f"Train: {len(train_headlines)} headlines")
    print(f"Val:   {len(val_headlines)} headlines")
    print(f"Test:  {len(test_headlines)} headlines")
    print()

    # Build vocabulary from training data only
    vocab = build_vocabulary(train_headlines)

    # Create datasets
    print("Creating datasets...")
    train_dataset = HeadlineDataset(train_headlines, vocab, seq_length=SEQ_LENGTH)
    val_dataset = HeadlineDataset(val_headlines, vocab, seq_length=SEQ_LENGTH)
    test_dataset = HeadlineDataset(test_headlines, vocab, seq_length=SEQ_LENGTH)

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples:   {len(val_dataset):,}")
    print(f"Test samples:  {len(test_dataset):,}")
    print()

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    print("Initializing model...")
    model = LSTMLanguageModel(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=2, verbose=True)

    # Training loop
    print("Starting training...")
    print("=" * 60)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)

        # Validate
        val_loss, val_perplexity = evaluate(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Train PPL: {np.exp(train_loss):.2f}")
        print(f"Val Loss:   {val_loss:.4f} | Val PPL:   {val_perplexity:.2f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = Path(__file__).parent / 'models' / 'lstm_lm'
            model_path.parent.mkdir(exist_ok=True)
            model.save(str(model_path), vocab.word2idx)
            print(f"✓ New best model saved (Val PPL: {val_perplexity:.2f})")

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_perplexity = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Perplexity: {test_perplexity:.2f}")

    # Plot training curves
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    plot_training_curves(train_losses, val_losses, results_dir / 'training_curves.png')

    # Save final results
    results = {
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'test_loss': test_loss,
        'test_perplexity': test_perplexity,
        'best_val_loss': best_val_loss,
        'num_epochs': NUM_EPOCHS,
        'vocab_size': len(vocab),
        'total_params': total_params
    }

    import json
    with open(results_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_dir}/")
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Best Val Perplexity:  {np.exp(best_val_loss):.2f}")
    print(f"Final Test Perplexity: {test_perplexity:.2f}")
    print(f"Model saved to: models/lstm_lm.pt")
    print("=" * 60)


if __name__ == '__main__':
    main()
