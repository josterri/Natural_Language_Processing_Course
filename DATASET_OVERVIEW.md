# NLP Datasets - Comprehensive Overview

## Introduction

This repository contains three progressive synthetic news datasets designed for BSc-level Natural Language Processing education. The datasets progress from basic text analysis to advanced NLP tasks including embeddings, language modeling, and summarization.

## Dataset Summary

| Dataset | Records | Total Tokens | Unique Words | Zipf Slope | Primary Use Case |
|---------|---------|--------------|--------------|------------|------------------|
| Basic Headlines | 400 | 2,855 | 372 | -0.486 | Intro text analysis |
| Extended Headlines | 10,000 | 70,381 | 937 | -0.609 | Embeddings, LM |
| Articles | 1,000 | 69,810 | 1,012 | -0.807 | Summarization |

---

## 1. Basic Headlines Dataset

### Purpose
Introductory text analysis exercises for students learning NLP fundamentals.

### Specifications
- **File:** `basic/news_headlines_dataset.csv`
- **Size:** 400 headlines
- **Categories:** Politics (100), Sports (100), Technology (100), Entertainment (100)
- **Total tokens:** 2,855
- **Unique words:** 372
- **Average length:** 7.14 words per headline

### Columns
- `headline_id` - Unique identifier (1-400)
- `headline` - News headline text
- `category` - One of 4 categories
- `word_count` - Number of words in headline
- `has_number` - Boolean indicating if headline contains digits

### Statistical Properties
- **Type-token ratio:** 0.130
- **Zipf slope:** -0.486 (flatter than ideal due to small corpus)
- **R² (log-log fit):** 0.955
- **Duplicates:** 32 (8%)

### Example Headlines
```
machine learning market expected to grow 15 percent
Warriors secures playoff spot with win over Rangers
President Smith announces new education initiative
The Office renewed for fourth season
```

### Use Cases
- Introduction to tokenization
- Word frequency analysis
- Basic text statistics
- Zipf's law demonstration (showing limitations of small corpora)
- Category-based text comparison

### Notebooks
- `nlp_basics_homework.ipynb` - Student exercises with TODO sections
- `nlp_basics_solutions.ipynb` - Complete solutions for instructors

---

## 2. Extended Headlines Dataset

### Purpose
Larger corpus for word embeddings training and language modeling tasks.

### Specifications
- **File:** `extended/news_headlines_extended.csv`
- **Size:** 10,000 headlines
- **Categories:** Politics (2,500), Sports (2,500), Technology (2,500), Entertainment (2,500)
- **Total tokens:** 70,381
- **Unique words:** 937
- **Average length:** 7.04 words per headline

### Columns
- `headline_id` - Unique identifier (1-10,000)
- `headline` - News headline text
- `category` - One of 4 categories
- `word_count` - Number of words in headline
- `has_number` - Boolean indicating if headline contains digits

### Statistical Properties
- **Type-token ratio:** 0.0133
- **Zipf slope:** -0.609 (improved but still flatter than ideal)
- **R² (log-log fit):** 0.9626 (excellent linear fit)
- **Duplicates:** 2,512 (25.1%) - expected with template generation

### Top 10 Most Frequent Words
1. 'for': 1,500
2. 'with': 1,278
3. 'to': 1,154
4. 'in': 1,059
5. 'new': 906
6. 'on': 860
7. 'million': 597
8. 'of': 531
9. 'the': 512
10. 'after': 486

### Train/Validation/Test Splits
The dataset is split using stratified sampling to maintain balanced category distributions:

| Split | File | Size | Categories (each) |
|-------|------|------|-------------------|
| Train | `extended/train.csv` | 7,000 (70%) | 1,750 |
| Validation | `extended/val.csv` | 1,500 (15%) | 375 |
| Test | `extended/test.csv` | 1,500 (15%) | 375 |

**No data leakage:** Splits are completely disjoint with no overlapping headline IDs.

### Use Cases
- **Word embeddings:** Train Word2Vec, GloVe, FastText models
- **N-gram language models:** Sufficient data for trigram/4-gram models
- **Neural language models:** Basic RNN/LSTM training
- **Text classification:** Category prediction tasks
- **Perplexity evaluation:** Proper train/val/test methodology

### Limitations
- Zipf slope (-0.609) indicates flatter distribution than natural text
- Short headlines (7 words) limit context for embeddings
- Template-based generation creates some repetitive patterns
- Vocabulary size (937) is limited compared to real news corpora

### Recommendations
- Suitable for introductory embeddings exercises
- Good for demonstrating language modeling concepts
- May need supplementation for production-quality embeddings
- Ideal for BSc-level projects, less so for research

---

## 3. Articles Dataset

### Purpose
Summarization tasks with article-headline pairs showing natural discourse structure.

### Specifications
- **File:** `articles/news_articles_dataset.csv`
- **Size:** 1,000 article-headline pairs
- **Categories:** Politics (250), Sports (250), Technology (250), Entertainment (250)

#### Article Statistics
- **Total tokens:** 69,810
- **Unique words:** 1,012
- **Average length:** 69.81 words per article
- **Word count range:** 62-84 words

#### Headline Statistics
- **Total tokens:** 7,107
- **Unique words:** 280
- **Average length:** 7.11 words per headline
- **Word count range:** 5-11 words

### Columns
- `article_id` - Unique identifier (1-1,000)
- `headline` - Summary headline for article
- `article` - Full article text (4-5 sentences)
- `category` - One of 4 categories
- `headline_word_count` - Number of words in headline
- `article_word_count` - Number of words in article
- `has_number` - Boolean indicating if article contains digits

### Statistical Properties

#### Article Text
- **Type-token ratio:** 0.0145
- **Zipf slope:** -0.807 (moderate Zipfian behavior - much better!)
- **R² (log-log fit):** 0.9699 (excellent)
- **Duplicates:** 2 (0.2%)

#### Headline Text
- **Type-token ratio:** 0.0394
- **Zipf slope:** -0.454
- **R² (log-log fit):** 0.8445

### Key Strength: Article Zipf's Law
The articles show **much better Zipf adherence (slope = -0.807)** compared to headlines because:
1. **Longer texts** (69 words vs 7 words) allow natural word distributions
2. **Discourse structure** with connectives creates realistic function word patterns
3. **Contextual richness** from multi-sentence format

### Top 10 Words (Article Text)
1. 'the': 5,512
2. 'and': 1,973
3. 'to': 1,782
4. 'a': 1,708
5. 'in': 1,690
6. 'of': 943
7. 'for': 800
8. 'will': 644
9. 'with': 619
10. 'on': 544

### Article Structure
Each article follows a 4-5 sentence template:
1. **Sentence 1:** Main event (aligned with headline)
2. **Sentence 2:** Background/context
3. **Sentence 3:** Quote, details, or statistics
4. **Sentence 4:** Impact or reaction
5. **Sentence 5:** Future outlook (optional)

### Discourse Features
- **Connectives:** However, Additionally, Therefore, Moreover, etc.
- **Entity consistency:** Pronouns and co-references maintained
- **Temporal coherence:** Logical time progression
- **Causal structure:** Events linked with appropriate connectives

### Example Article-Headline Pair

**Headline:**
"AppWorks addresses security concerns in popular platform"

**Article:**
"AppWorks released an urgent security update for its platform after vulnerabilities were discovered. The issues affected an estimated 30 million users worldwide and prompted immediate action. However cybersecurity experts had alerted the company to potential exploits several weeks ago. The company apologized to customers and assured them that no data breaches have been detected. All users are strongly advised to install the latest update to protect their devices and information."

### Train/Validation/Test Splits

| Split | File | Size | Categories (each) |
|-------|------|------|-------------------|
| Train | `articles/train.csv` | 700 (70%) | 175 |
| Validation | `articles/val.csv` | 150 (15%) | ~38 |
| Test | `articles/test.csv` | 150 (15%) | ~38 |

### Use Cases
- **Extractive summarization:** Select important sentences
- **Abstractive summarization:** Generate new headlines from articles
- **Seq2seq models:** Encoder-decoder architectures
- **ROUGE evaluation:** Standard summarization metrics
- **Headline generation:** Conditional language modeling
- **Document understanding:** Multi-sentence context analysis

### Advantages Over Headlines
1. **Better Zipf's law adherence** (slope -0.807 vs -0.609)
2. **Richer vocabulary** with natural distribution
3. **Discourse structure** for context-aware models
4. **Realistic summarization task** with proper article-headline pairing
5. **Longer context** for better embeddings and representations

---

## Generation Details

### Template-Based Synthesis
All datasets use template-based generation with random entity substitution:

| Dataset | Templates per Category | Vocabulary Size | Total Combinations |
|---------|------------------------|-----------------|-------------------|
| Basic | 15 | Standard | ~60 per category |
| Extended | 50 | 2-3x larger | ~200 per category |
| Articles | 5 multi-sentence | Large | ~25 per category |

### Random Seed
All generators use `random.seed(42)` for reproducibility.

### Quality Control
- Stratified sampling ensures balanced categories
- Duplicate checking performed (acceptable levels for educational use)
- Zipf's law analysis verifies linguistic realism
- Type-token ratio computed to assess vocabulary diversity

---

## Comparison: Headlines vs Articles

| Metric | Headlines (10K) | Articles (1K) | Winner |
|--------|-----------------|---------------|---------|
| Zipf slope | -0.609 | -0.807 | Articles |
| R² (fit quality) | 0.9626 | 0.9699 | Articles |
| Unique words | 937 | 1,012 | Articles |
| Type-token ratio | 0.0133 | 0.0145 | Articles |
| Avg text length | 7 words | 69.8 words | Articles |
| Discourse structure | None | Rich | Articles |

**Conclusion:** Articles dataset is linguistically superior due to longer texts and discourse structure, making it ideal for advanced NLP tasks.

---

## Usage Recommendations

### For Beginners (Weeks 1-2)
- Start with **Basic Headlines (400)**
- Focus on word frequency, tokenization, basic statistics
- Learn Zipf's law principles (and its limitations)
- Practice with simple visualizations

### For Intermediate Students (Weeks 3-4)
- Use **Extended Headlines (10K)** train/val splits
- Train simple word embeddings (Word2Vec, GloVe)
- Build n-gram language models
- Understand train/validation methodology

### For Advanced Students (Weeks 5-8)
- **Language Modeling:** Extended Headlines with proper splits
- **Summarization:** Articles dataset with article-headline pairs
- Compare extractive vs abstractive approaches
- Evaluate with ROUGE metrics

---

## Limitations and Considerations

### General Limitations
1. **Synthetic data:** Template-based, not natural text
2. **Domain-specific:** News headlines only
3. **Vocabulary size:** Limited compared to real corpora
4. **Duplicates:** Template combinations create some repeats
5. **Simplified language:** Educational focus, not research-grade

### Zipf's Law Deviations
- **Basic (slope -0.486):** Too small for realistic distribution
- **Extended (slope -0.609):** Better but still flatter than natural text
- **Articles (slope -0.807):** Best approximation, nearly ideal

**Why?** Short headlines and template generation create more uniform distributions than natural language.

### When to Use Real Data
- Research publications
- Production systems
- Fine-tuning pre-trained models
- Benchmarking against state-of-the-art
- Understanding real-world data challenges

---

## Future Enhancements

Possible extensions to improve datasets:
1. **Scale up:** Generate 20K-50K headlines for better Zipf adherence
2. **Sentiment labels:** Add positive/negative/neutral annotations
3. **Named entities:** Mark person, organization, location entities
4. **Paraphrase pairs:** Create semantically similar headline variants
5. **Longer articles:** Extend to 150-200 words for document tasks
6. **Multiple references:** Provide 2-3 reference summaries per article
7. **Real data augmentation:** Mix synthetic with real news (with proper licensing)

---

## Citation

If you use these datasets in academic work, please cite:

```
NLP Educational Datasets
Generated: October 2025
Purpose: BSc-level Natural Language Processing education
Features: Progressive difficulty from basic analysis to summarization
```

---

## License

These synthetic datasets are provided for educational purposes only. No real news content is included.

---

## Contact

For questions, issues, or suggestions, please refer to the repository documentation or contact the course instructor.

---

## Appendix: Quick Reference

### File Locations
```
basic/news_headlines_dataset.csv          # 400 headlines
extended/news_headlines_extended.csv      # 10,000 headlines
extended/train.csv                        # 7,000 training headlines
extended/val.csv                          # 1,500 validation headlines
extended/test.csv                         # 1,500 test headlines
articles/news_articles_dataset.csv        # 1,000 article-headline pairs
articles/train.csv                        # 700 training articles
articles/val.csv                          # 150 validation articles
articles/test.csv                         # 150 test articles
```

### Generator Scripts
```
generators/generate_headlines.py          # Basic dataset (400)
generators/generate_extended_headlines.py # Extended dataset (10K)
generators/generate_articles.py           # Articles dataset (1K)
generators/create_splits.py               # Train/val/test splitting
generators/verify_datasets.py             # Comprehensive verification
```

### Total Corpus Statistics
- **Total records:** 11,400 (10,400 headlines + 1,000 articles)
- **Total tokens:** ~147,000 (headlines + articles)
- **Unique words:** ~2,000 across all datasets
- **Categories:** 4 (balanced distribution)
- **Languages:** English only
