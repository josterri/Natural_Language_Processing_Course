# Sentence Embeddings Beamer Tutorial

## Overview

Professional Beamer presentation converting the embeddings notebook into a pedagogical tutorial. Focuses on Hugging Face's `SentenceTransformer('all-MiniLM-L6-v2')` model with comprehensive visualizations.

## Files

```
presentation/
├── 20251002_1001_embeddings_tutorial.tex    # Main Beamer presentation
├── 20251002_1001_embeddings_tutorial.pdf    # Compiled PDF (28 slides, 947 KB)
├── generate_charts.py                        # Chart generation script
├── charts/                                   # Generated visualization PDFs
│   ├── embedding_concept.pdf
│   ├── cosine_similarity_example.pdf
│   ├── pca_visualization.pdf
│   ├── pca_variance_explained.pdf           # NEW
│   ├── tsne_visualization.pdf
│   ├── clustering_comparison.pdf
│   ├── similarity_distribution.pdf
│   ├── similarity_heatmap.pdf               # NEW
│   └── semantic_neighborhood.pdf            # NEW
├── temp/                                     # Auxiliary LaTeX files
└── previous/                                 # Previous versions
```

## Presentation Structure (28 Slides)

### Part 1: Introduction & Motivation (4 slides)
1. Title slide
2. Learning objectives
3. The problem: How computers understand text
4. Section divider

### Part 2: Core Concepts (3 slides)
5. What are embeddings? (with diagram)
6. From words to sentences
7. Cosine similarity explained (with geometric visualization)

### Part 3: The Model - Hugging Face (5 slides)
8. Introducing sentence-transformers
9. The model: all-MiniLM-L6-v2 specifications
10. Code: Loading the model
11. Code: Generating embeddings
12. Understanding the output

### Part 4: Visualizations & Analysis (10 slides)
13. Our dataset (10,000 headlines)
14. Dimensionality reduction challenge
15. **PCA variance explained (scree plot)** ← NEW
16. PCA visualization
17. t-SNE visualization
18. Clustering analysis
19. Similarity patterns
20. **Similarity heatmap (concrete examples)** ← NEW

### Part 5: Practical Applications (5 slides)
21. Semantic search in action (with code)
22. **Semantic neighborhood (top 10 results)** ← NEW
23. Real-world applications
24. Advantages over traditional methods

### Part 6: Summary & Next Steps (4 slides)
25. Key takeaways
26. From notebook to production (3 steps)
27. Further exploration
28. Resources & references + Thank you

## Key Features

### Pedagogical Framework
- **Learning Objectives**: Clear goals at start
- **Progressive Complexity**: Simple → Complex
- **Concrete Examples**: Real code and results
- **Visual Learning**: Charts on most slides
- **Hands-on Code**: Actual working examples
- **Repetition**: Key concepts reinforced

### Code Examples
All code examples are real and runnable:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(headlines)
```

### Visualizations
Nine publication-quality charts:
1. **Embedding Concept**: Text → Vector transformation
2. **Cosine Similarity**: Geometric interpretation
3. **PCA Variance Explained**: Scree plot showing dimensionality trade-offs ← NEW
4. **PCA**: Linear dimensionality reduction
5. **t-SNE**: Non-linear clustering visualization
6. **Clustering**: Actual vs K-means comparison
7. **Similarity Distribution**: Within/between category patterns
8. **Similarity Heatmap**: Concrete example headlines with scores ← NEW
9. **Semantic Neighborhood**: Top 10 similar headlines for query ← NEW

### Design
- **Theme**: Madrid (Beamer)
- **Font**: 8pt
- **Aspect Ratio**: 16:9
- **Color Scheme**: Purple/lavender palette
- **Style**: Clean, professional, academic

## Usage

### Viewing the Presentation
```bash
# Open the PDF
open 20251002_1001_embeddings_tutorial.pdf
```

### Regenerating Charts
```bash
cd presentation/
python generate_charts.py
```

This will:
- Generate all 9 charts as PDFs
- Takes ~3-4 minutes (includes t-SNE computation)
- Saves to `charts/` folder

### Recompiling LaTeX
```bash
pdflatex 20251002_1001_embeddings_tutorial.tex
```

Or use your LaTeX editor (TeXstudio, Overleaf, etc.)

## Chart Details

### 1. Embedding Concept Diagram
- Shows: Text → SentenceTransformer → 384D vectors
- Purpose: Visual explanation of embedding process
- Size: Optimized for slide

### 2. Cosine Similarity Example
- Shows: Two vectors with angle θ
- Formula: cos(θ) = (A·B) / (||A|| × ||B||)
- Purpose: Geometric interpretation

### 3. PCA Visualization
- Data: 10,000 embeddings → 2D
- Colors: 4 categories (Politics, Sports, Tech, Entertainment)
- Shows: Some category separation
- Variance explained: PC1 + PC2 ≈ 15%

### 4. t-SNE Visualization
- Data: 2,000 sample embeddings → 2D
- Method: Non-linear dimensionality reduction
- Shows: Clear category clusters
- Better separation than PCA

### 5. Clustering Comparison
- Left: Actual categories
- Right: K-means clusters (unsupervised)
- Shows: Clusters align with categories (97%+ accuracy)
- Validates: Embeddings capture semantic structure

### 6. Similarity Distribution
- Boxplots: Within-category vs between-category
- Finding: Same category = 35% more similar
- Within: ~0.62 average
- Between: ~0.46 average

### 7. PCA Variance Explained ← NEW
- Scree plot: First 50 principal components
- Bar chart: Individual variance per component
- Line chart: Cumulative variance percentage
- Shows: Need ~35 dimensions for 50% variance
- Purpose: Understand dimensionality vs information trade-off

### 8. Similarity Heatmap ← NEW
- 4×4 matrix: One example headline per category
- Color-coded: Purple gradient (0.0 to 1.0)
- Text annotations: Exact similarity scores
- Shows: Within-category (diagonal) vs between-category differences
- Purpose: Concrete examples of semantic similarity

### 9. Semantic Neighborhood ← NEW
- Horizontal bar chart: Top 10 most similar headlines
- Query: Politics headline about president/announcements
- Color-coded by category (Politics, Sports, Tech, Entertainment)
- Similarity scores: 0.75-0.85 range
- Purpose: Demonstrates practical semantic search results

## Educational Notes

### Target Audience
- **Level**: Undergraduate
- **Background**: Basic Python, basic linear algebra
- **Duration**: 30-40 minutes presentation

### Learning Outcomes
After this presentation, students can:
1. Explain what embeddings are
2. Use sentence-transformers in 3 lines of code
3. Calculate cosine similarity
4. Visualize embeddings with PCA/t-SNE
5. Apply to semantic search tasks

### Key Messages
1. **Embeddings capture meaning** in numbers
2. **Hugging Face makes it easy** (just 3 lines!)
3. **Cosine similarity** measures semantic distance
4. **Visualizations reveal structure** (PCA, t-SNE)
5. **Real applications** (search, clustering, classification)

## Technical Specifications

### LaTeX Packages Used
- `beamer` (Madrid theme)
- `graphicx` (images)
- `listings` (code)
- `amsmath` (formulas)
- `xcolor` (colors)

### Python Dependencies (for chart generation)
- `sentence-transformers`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

### Chart Generation
- Format: PDF (vector graphics)
- DPI: 300
- Style: Publication quality
- Colors: Match template palette

## Customization

### Change the Model
Edit in slides and regenerate:
```python
# Instead of:
model = SentenceTransformer('all-MiniLM-L6-v2')

# Try:
model = SentenceTransformer('all-mpnet-base-v2')  # Higher quality
model = SentenceTransformer('paraphrase-multilingual-*')  # Multilingual
```

### Adjust Chart Sizes
In `generate_charts.py`:
```python
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust width, height
```

### Modify Colors
In LaTeX preamble:
```latex
\definecolor{mlpurple}{RGB}{51,51,178}  % Change RGB values
```

## Tips for Presenters

### Slide Timing
- Introduction: 5 minutes
- Concepts: 5 minutes
- Model & Code: 10 minutes
- Visualizations: 10 minutes
- Applications: 5 minutes
- Summary: 5 minutes
- **Total**: ~40 minutes

### Key Points to Emphasize
1. **Slide 7**: Embeddings = meaningful numbers (core concept)
2. **Slide 10**: Just 3 lines of code (accessibility)
3. **Slide 16**: t-SNE shows clear clusters (validation)
4. **Slide 20**: Semantic search finds "chancellor" when query is "president" (power)
5. **Slide 24**: Production ready in 3 steps (practical)

### Live Demo Options
- Run semantic search in notebook
- Generate embeddings for custom sentences
- Visualize with t-SNE interactively

## Troubleshooting

### PDF Not Compiling
- Ensure all packages installed: `beamer`, `listings`, etc.
- Check charts exist in `charts/` folder
- Verify no special characters in paths

### Charts Not Generating
- Install Python dependencies: `pip install -r ../requirements.txt`
- Check embeddings exist: `../data/headlines_embeddings.npy`
- Run from `presentation/` directory

### Fonts Look Wrong
- Install CM Super fonts for better rendering
- Or use different LaTeX engine (XeLaTeX, LuaLaTeX)

## Version History

- **20251002_1001**: Current version
  - Removed BSc mentions (now "Undergraduate")
  - 28 slides
  - 9 visualizations
  - No other changes

- **20251002_0919**: Enhanced version
  - 28 slides (+3 new)
  - 9 visualizations (+3 new)
  - Added: PCA variance explained (scree plot)
  - Added: Similarity heatmap (concrete examples)
  - Added: Semantic neighborhood (top 10 results)
  - More pedagogical depth and concrete examples

- **20251002_1940**: Initial version
  - 25 slides
  - 6 visualizations
  - Complete pedagogical framework
  - All code examples included

## License & Attribution

- **Template**: Based on Beamer Madrid theme
- **Content**: Original educational material
- **Code**: sentence-transformers (Apache 2.0)
- **Model**: all-MiniLM-L6-v2 (Apache 2.0)

## Related Materials

- **Notebook**: `../embedding_analysis.ipynb` (source material)
- **Scripts**: `../generate_embeddings.py`, `../semantic_search.py`
- **README**: `../README.md` (embeddings documentation)
- **Dataset**: `../../extended/news_headlines_extended.csv`

## Contact & Feedback

For questions or improvements, refer to the main NLP_Data project documentation.

---

**Generated**: 2025-10-02 (Updated: 10:01)
**Format**: Beamer presentation (Madrid theme, 16:9)
**Pages**: 28 (was 25, added 3 new slides)
**Size**: 947 KB
**Charts**: 9 (was 6, added 3 new visualizations)
**Changes**: Removed BSc references
**Status**: Ready for teaching
