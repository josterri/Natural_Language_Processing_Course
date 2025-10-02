# 5-Gram Language Model

This folder contains a complete implementation of a 5-gram language model trained on the extended headlines dataset (10,000 headlines).

## Overview

An n-gram language model predicts the next word based on the previous (n-1) words. For a 5-gram model:
- **Context**: 4 previous words
- **Prediction**: The 5th word
- **Training corpus**: 10,000 news headlines from extended dataset
- **Vocabulary**: 939 unique words
- **Total n-grams**: 80,381

## Files

### Core Implementation
- **ngram_model.py** - Main NGramModel class with training and generation methods
- **train_5gram.py** - Script to train the model on extended headlines dataset
- **generate_text.py** - CLI tool for generating text with various options
- **generate_half_page.py** - Generates 200-word samples (half a page)
- **ngram_analysis.ipynb** - Comprehensive Jupyter notebook with analysis and visualization

### Model and Outputs
- **models/5gram_extended.pkl** - Trained 5-gram model (902 KB)
- **samples/** - Directory containing generated text samples

## Model Statistics

- N-gram size: 5
- Vocabulary size: 939 words
- Unique contexts: 18,378
- Total n-grams: 80,381
- Smoothing: Add-k (k=0.01)

## Usage

### 1. Train the Model

```bash
python train_5gram.py
```

This will:
- Load the extended headlines dataset
- Train a 5-gram model
- Display statistics and top n-grams
- Save the model to models/5gram_extended.pkl

### 2. Generate Text

Generate 200-word samples (half a page):
```bash
python generate_half_page.py
```

Or use the flexible generation script:
```bash
python generate_text.py --words 200 --samples 3 --save
```

Options:
- `--words N` - Number of words to generate (default: 200)
- `--samples N` - Number of samples (default: 3)
- `--seed TEXT` - Seed text to start generation
- `--save` - Save samples to files

### 3. Interactive Analysis

Open the Jupyter notebook:
```bash
jupyter notebook ngram_analysis.ipynb
```

The notebook includes:
- Model training and statistics
- N-gram frequency analysis
- Conditional probability examples
- Text generation with various parameters
- Comparison of 3-gram, 4-gram, and 5-gram models

## Example Output

```
netsphere commits to carbon neutral processor manufacturing . india imposes
new sanctions on japan . budget committee approves 250 million for climate
change . hackers exploit console flaw affecting 25 devices . developers
create innovative healthcare using quantum computing . controversy erupts
as hawks protests referee decision . hockey legend brown inducted into hall
of fame . emma stone stars in new horror film new dawn . platform update
brings security to millions of users . global artificial intelligence
conference showcases latest innovations . foreign minister discusses trade
agreement with portugal . senate votes on agriculture reform bill . phoenix
dynasty continues with another trophy win . lawmakers approve 25 billion
tax package . scarlett johansson wins all-star for performance in eternal .
president smith faces impeachment over education allegations . critics
praise rising tide for groundbreaking production design . draft pick young
impresses in debut for eagles . italy and france reach compromise on
education . foreign minister discusses technology accord with switzerland .
```

## Technical Details

### Smoothing
The model uses add-k smoothing (k=0.01) to handle unseen n-grams:

```
P(word | context) = (Count(context, word) + k) / (Count(context) + k * V)
```

Where V is the vocabulary size.

### Backoff Strategy
If a 5-gram context is not found, the model backs off to shorter contexts:
- Try 5-gram (4 words context)
- If not found, try 4-gram (3 words context)
- If not found, try 3-gram (2 words context)
- If still not found, sample randomly from vocabulary

### Multi-Sentence Generation
Since headlines are short (avg 7 words), the model uses a multi-sentence strategy:
- When '<END>' is encountered, adds a period
- Resets context to '<START>'
- Continues generating until reaching target word count

## Limitations

1. **Template-based training data**: The headlines were generated from templates, creating repetitive patterns
2. **Short training sequences**: Headlines average 7 words, limiting long-range coherence
3. **No semantic understanding**: The model only captures statistical patterns, not meaning
4. **Local context only**: Only considers previous 4 words, not document-level context
5. **Domain-specific**: Trained on news headlines, may not generalize to other domains

## Future Improvements

- Train on longer texts (articles dataset) for better coherence
- Implement Kneser-Ney smoothing for better probability estimation
- Add perplexity evaluation on held-out test set
- Implement beam search for more coherent generation
- Compare with neural language models (RNN, Transformer)
