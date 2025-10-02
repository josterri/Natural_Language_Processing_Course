import pickle
import random
from collections import defaultdict, Counter
from typing import List, Tuple, Optional

class NGramModel:
    def __init__(self, n: int = 5, smoothing_k: float = 0.01):
        self.n = n
        self.smoothing_k = smoothing_k
        self.ngrams = defaultdict(Counter)
        self.vocab = set()
        self.vocab_size = 0

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def _add_sentence_boundaries(self, tokens: List[str]) -> List[str]:
        start_tokens = ['<START>'] * (self.n - 1)
        end_token = ['<END>']
        return start_tokens + tokens + end_token

    def train(self, texts: List[str], verbose: bool = True):
        if verbose:
            print(f"Training {self.n}-gram model...")
            print(f"Corpus size: {len(texts)} texts")

        all_tokens = []
        for text in texts:
            tokens = self._tokenize(text)
            tokens_with_boundaries = self._add_sentence_boundaries(tokens)
            all_tokens.extend(tokens)

            for i in range(len(tokens_with_boundaries) - self.n + 1):
                ngram = tuple(tokens_with_boundaries[i:i + self.n])
                context = ngram[:-1]
                word = ngram[-1]

                self.ngrams[context][word] += 1

        self.vocab = set(all_tokens)
        self.vocab.add('<START>')
        self.vocab.add('<END>')
        self.vocab_size = len(self.vocab)

        if verbose:
            print(f"Vocabulary size: {self.vocab_size}")
            print(f"Unique {self.n}-gram contexts: {len(self.ngrams)}")
            total_ngrams = sum(sum(counts.values()) for counts in self.ngrams.values())
            print(f"Total {self.n}-grams: {total_ngrams}")

    def _get_probability(self, context: Tuple[str, ...], word: str) -> float:
        context_count = sum(self.ngrams[context].values())
        word_count = self.ngrams[context][word]

        numerator = word_count + self.smoothing_k
        denominator = context_count + self.smoothing_k * self.vocab_size

        return numerator / denominator

    def _sample_next_word(self, context: Tuple[str, ...]) -> str:
        if context not in self.ngrams or len(self.ngrams[context]) == 0:
            for i in range(1, len(context)):
                shorter_context = context[i:]
                if shorter_context in self.ngrams and len(self.ngrams[shorter_context]) > 0:
                    context = shorter_context
                    break
            else:
                return random.choice(list(self.vocab - {'<START>', '<END>'}))

        possible_words = list(self.ngrams[context].keys())
        probabilities = [self._get_probability(context, word) for word in possible_words]

        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]

        return random.choices(possible_words, weights=probabilities, k=1)[0]

    def generate(self, max_words: int = 200, seed: Optional[List[str]] = None,
                 temperature: float = 1.0, multi_sentence: bool = True) -> str:
        if seed is None:
            context = ['<START>'] * (self.n - 1)
        else:
            seed_tokens = self._tokenize(' '.join(seed)) if isinstance(seed, list) else self._tokenize(seed)
            if len(seed_tokens) < self.n - 1:
                context = ['<START>'] * (self.n - 1 - len(seed_tokens)) + seed_tokens
            else:
                context = seed_tokens[-(self.n - 1):]

        generated = list(context)
        words_generated = 0
        attempts = 0
        max_attempts = max_words * 20

        while words_generated < max_words and attempts < max_attempts:
            attempts += 1
            context_tuple = tuple(context[-(self.n - 1):])
            next_word = self._sample_next_word(context_tuple)

            if next_word == '<END>':
                if multi_sentence and words_generated < max_words:
                    generated.append('.')
                    context = ['<START>'] * (self.n - 1)
                    words_generated += 1
                    continue
                else:
                    break

            generated.append(next_word)
            context.append(next_word)
            words_generated += 1

        result = ' '.join([w for w in generated if w not in ['<START>', '<END>']])
        return result

    def get_ngram_stats(self) -> dict:
        stats = {
            'n': self.n,
            'vocab_size': self.vocab_size,
            'unique_contexts': len(self.ngrams),
            'total_ngrams': sum(sum(counts.values()) for counts in self.ngrams.values()),
            'smoothing_k': self.smoothing_k
        }
        return stats

    def get_top_ngrams(self, k: int = 20) -> List[Tuple[Tuple[str, ...], str, int]]:
        all_ngrams = []
        for context, word_counts in self.ngrams.items():
            for word, count in word_counts.items():
                all_ngrams.append((context, word, count))

        all_ngrams.sort(key=lambda x: x[2], reverse=True)
        return all_ngrams[:k]

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'NGramModel':
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model
