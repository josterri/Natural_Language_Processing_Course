"""
Neural Language Model using LSTM

A simple yet effective LSTM-based language model for next-word prediction.
Designed for educational purposes with clear, readable code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict


class LSTMLanguageModel(nn.Module):
    """
    LSTM-based language model for predicting the next word.

    Architecture:
        - Embedding layer: vocab_size → embedding_dim
        - LSTM layers: embedding_dim → hidden_dim (num_layers)
        - Dropout: for regularization
        - Output layer: hidden_dim → vocab_size
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super(LSTMLanguageModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with appropriate distributions."""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Forward pass through the model.

        Args:
            x: Input tensor of token indices (batch_size, seq_len)
            hidden: Optional hidden state tuple (h_0, c_0)

        Returns:
            output: Logits for next word prediction (batch_size, seq_len, vocab_size)
            hidden: Updated hidden state tuple (h_n, c_n)
        """
        # Embed input tokens
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)

        # Pass through LSTM
        if hidden is None:
            lstm_out, hidden = self.lstm(embedded)
        else:
            lstm_out, hidden = self.lstm(embedded, hidden)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Project to vocabulary
        output = self.fc(lstm_out)  # (batch_size, seq_len, vocab_size)

        return output, hidden

    def init_hidden(self, batch_size: int, device: str = 'cpu'):
        """Initialize hidden state with zeros."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

    def predict_next_word(self, context: torch.Tensor, hidden: Optional[Tuple] = None,
                         top_k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Predict the next word given a context.

        Args:
            context: Input tensor of token indices (batch_size=1, seq_len)
            hidden: Optional hidden state
            top_k: Number of top predictions to return

        Returns:
            top_indices: List of top k token indices
            top_probs: List of top k probabilities
        """
        self.eval()
        with torch.no_grad():
            output, hidden = self.forward(context, hidden)

            # Get logits for last position
            logits = output[0, -1, :]  # (vocab_size,)
            probs = F.softmax(logits, dim=0)

            # Get top k predictions
            top_probs, top_indices = torch.topk(probs, top_k)

            return top_indices.cpu().tolist(), top_probs.cpu().tolist()

    def generate(self, start_tokens: List[int], max_length: int = 100,
                temperature: float = 1.0, device: str = 'cpu') -> List[int]:
        """
        Generate text autoregressively.

        Args:
            start_tokens: Initial context token indices
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            device: Device to run on

        Returns:
            generated: List of generated token indices
        """
        self.eval()
        generated = start_tokens.copy()
        hidden = None

        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input
                x = torch.tensor([generated], dtype=torch.long).to(device)

                # Forward pass
                output, hidden = self.forward(x, hidden)

                # Get logits for last position
                logits = output[0, -1, :] / temperature
                probs = F.softmax(logits, dim=0)

                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1).item()

                # Stop if end token (assuming 2 is <END>)
                if next_token == 2:
                    break

                generated.append(next_token)

        return generated

    def get_perplexity(self, data_loader, device: str = 'cpu') -> float:
        """
        Calculate perplexity on a dataset.

        Args:
            data_loader: DataLoader with (input, target) batches
            device: Device to run on

        Returns:
            perplexity: Model perplexity on the dataset
        """
        self.eval()
        total_loss = 0.0
        total_tokens = 0
        criterion = nn.CrossEntropyLoss(reduction='sum')

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs, _ = self.forward(inputs)

                # Reshape for loss computation
                outputs = outputs.view(-1, self.vocab_size)
                targets = targets.view(-1)

                # Calculate loss
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                total_tokens += targets.size(0)

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity

    def save(self, filepath: str, vocab: Dict[str, int]):
        """
        Save model weights and configuration.

        Args:
            filepath: Path to save model (will create .pt and .json files)
            vocab: Word to index mapping
        """
        filepath = Path(filepath)

        # Save model weights
        torch.save(self.state_dict(), filepath.with_suffix('.pt'))

        # Save configuration
        config = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout_rate,
            'vocab': vocab
        }
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {filepath.with_suffix('.pt')}")
        print(f"Config saved to {filepath.with_suffix('.json')}")

    @staticmethod
    def load(filepath: str, device: str = 'cpu') -> Tuple['LSTMLanguageModel', Dict[str, int]]:
        """
        Load model from saved files.

        Args:
            filepath: Path to model file (without extension)
            device: Device to load model to

        Returns:
            model: Loaded model
            vocab: Word to index mapping
        """
        filepath = Path(filepath)

        # Load configuration
        with open(filepath.with_suffix('.json'), 'r') as f:
            config = json.load(f)

        vocab = config.pop('vocab')

        # Create model
        model = LSTMLanguageModel(**config)

        # Load weights
        model.load_state_dict(torch.load(filepath.with_suffix('.pt'),
                                        map_location=device))
        model.to(device)
        model.eval()

        print(f"Model loaded from {filepath.with_suffix('.pt')}")
        return model, vocab


class Vocabulary:
    """Helper class for managing vocabulary and token-word conversions."""

    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<START>', 2: '<END>', 3: '<UNK>'}
        self.word_count = {}

    def add_word(self, word: str):
        """Add word to vocabulary."""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.word_count[word] = self.word_count.get(word, 0) + 1

    def add_sentence(self, sentence: str):
        """Add all words in a sentence to vocabulary."""
        for word in sentence.lower().split():
            self.add_word(word)

    def __len__(self):
        return len(self.word2idx)

    def encode(self, words: List[str]) -> List[int]:
        """Convert words to indices."""
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]

    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices to words."""
        return [self.idx2word[idx] for idx in indices if idx in self.idx2word]
