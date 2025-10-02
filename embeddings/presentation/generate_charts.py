import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os

# Template color scheme
MLPURPLE = (51/255, 51/255, 178/255)
MLLAVENDER = (173/255, 173/255, 224/255)
MLLAVENDER2 = (193/255, 193/255, 232/255)
MLLAVENDER3 = (204/255, 204/255, 235/255)
MLBLUE = (0/255, 102/255, 204/255)
MLORANGE = (255/255, 127/255, 14/255)
MLGREEN = (44/255, 160/255, 44/255)
MLRED = (214/255, 39/255, 40/255)
MLGRAY = (127/255, 127/255, 127/255)

# Set style for clean, publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette([MLPURPLE, MLBLUE, MLORANGE, MLGREEN])

def ensure_charts_dir():
    os.makedirs('charts', exist_ok=True)

def load_data_and_embeddings():
    print("Loading data and embeddings...")

    # Check if embeddings exist
    if os.path.exists('../data/headlines_embeddings.npy'):
        embeddings = np.load('../data/headlines_embeddings.npy')
        df = pd.read_csv('../../extended/news_headlines_extended.csv')
        print(f"  Loaded {len(embeddings):,} embeddings")
        return df, embeddings
    else:
        print("  Embeddings not found. Generating...")
        df = pd.read_csv('../../extended/news_headlines_extended.csv')
        model = SentenceTransformer('all-MiniLM-L6-v2')
        headlines = df['headline'].tolist()
        embeddings = model.encode(headlines, batch_size=32, show_progress_bar=True)

        os.makedirs('../data', exist_ok=True)
        np.save('../data/headlines_embeddings.npy', embeddings)
        print(f"  Generated and saved {len(embeddings):,} embeddings")
        return df, embeddings

def generate_embedding_concept_diagram():
    print("\n[1/6] Generating embedding concept diagram...")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'From Text to Embeddings',
            ha='center', fontsize=18, fontweight='bold', color=MLPURPLE)

    # Left: Text examples
    texts = [
        '"Team wins championship"',
        '"President announces policy"',
        '"New AI breakthrough"'
    ]

    for i, text in enumerate(texts):
        y_pos = 0.75 - i*0.2
        ax.text(0.15, y_pos, text, ha='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=MLLAVENDER3,
                         edgecolor=MLPURPLE, linewidth=1.5))

    # Arrow
    ax.annotate('', xy=(0.45, 0.5), xytext=(0.3, 0.5),
                arrowprops=dict(arrowstyle='->', lw=3, color=MLPURPLE))
    ax.text(0.375, 0.55, 'SentenceTransformer', ha='center', fontsize=10, color=MLPURPLE)

    # Right: Vector representations
    ax.text(0.75, 0.75, '[0.12, -0.45, 0.78, ..., 0.34]', ha='center', fontsize=9,
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=MLLAVENDER2,
                     edgecolor=MLBLUE, linewidth=1.5))
    ax.text(0.75, 0.55, '[0.89, 0.23, -0.12, ..., -0.56]', ha='center', fontsize=9,
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=MLLAVENDER2,
                     edgecolor=MLBLUE, linewidth=1.5))
    ax.text(0.75, 0.35, '[-0.34, 0.67, 0.91, ..., 0.45]', ha='center', fontsize=9,
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=MLLAVENDER2,
                     edgecolor=MLBLUE, linewidth=1.5))

    # Bottom note
    ax.text(0.5, 0.1, '384 dimensions per sentence', ha='center', fontsize=10,
            style='italic', color=MLGRAY)

    plt.savefig('charts/embedding_concept.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: charts/embedding_concept.pdf")

def generate_cosine_similarity_diagram():
    print("\n[2/6] Generating cosine similarity diagram...")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw two vectors
    origin = np.array([0, 0])
    v1 = np.array([3, 2])
    v2 = np.array([2, 3])

    # Plot vectors
    ax.quiver(*origin, *v1, angles='xy', scale_units='xy', scale=1,
             width=0.008, color=MLBLUE, label='Vector A')
    ax.quiver(*origin, *v2, angles='xy', scale_units='xy', scale=1,
             width=0.008, color=MLORANGE, label='Vector B')

    # Draw angle arc
    angle = np.linspace(np.arctan2(v1[1], v1[0]), np.arctan2(v2[1], v2[0]), 50)
    r = 0.8
    ax.plot(r*np.cos(angle), r*np.sin(angle), 'k--', linewidth=1.5)
    ax.text(0.6, 1.2, r'$\theta$', fontsize=14, color='black')

    # Annotations
    ax.text(v1[0]+0.2, v1[1]+0.2, 'Sentence 1\nembedding', fontsize=10, color=MLBLUE)
    ax.text(v2[0]+0.2, v2[1]-0.3, 'Sentence 2\nembedding', fontsize=10, color=MLORANGE)

    # Formula
    ax.text(1.5, -1.5, r'$\cos(\theta) = \frac{A \cdot B}{||A|| \times ||B||}$',
           fontsize=14, bbox=dict(boxstyle='round,pad=0.8',
                                  facecolor=MLLAVENDER3, edgecolor=MLPURPLE))

    ax.set_xlim(-0.5, 4)
    ax.set_ylim(-2, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('Dimension 1', fontsize=11)
    ax.set_ylabel('Dimension 2', fontsize=11)
    ax.set_title('Cosine Similarity: Measuring Angle Between Vectors',
                fontsize=13, fontweight='bold', color=MLPURPLE)
    ax.legend(loc='upper left', fontsize=10)

    plt.savefig('charts/cosine_similarity_example.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: charts/cosine_similarity_example.pdf")

def generate_pca_visualization(df, embeddings):
    print("\n[3/6] Generating PCA visualization...")

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 7))

    categories = df['category'].unique()
    colors = [MLPURPLE, MLBLUE, MLORANGE, MLGREEN]

    for i, category in enumerate(categories):
        mask = df['category'] == category
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  c=[colors[i]], label=category, alpha=0.6, s=15, edgecolors='none')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('PCA: 384D Embeddings Projected to 2D',
                fontsize=14, fontweight='bold', color=MLPURPLE)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('charts/pca_visualization.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: charts/pca_visualization.pdf")

def generate_tsne_visualization(df, embeddings):
    print("\n[4/6] Generating t-SNE visualization (this may take 2-3 minutes)...")

    # Use subset for speed
    n_samples = 2000
    np.random.seed(42)
    sample_indices = np.random.choice(len(embeddings), n_samples, replace=False)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings[sample_indices])

    fig, ax = plt.subplots(figsize=(11, 8))

    df_sample = df.iloc[sample_indices]
    categories = df['category'].unique()
    colors = [MLPURPLE, MLBLUE, MLORANGE, MLGREEN]

    for i, category in enumerate(categories):
        mask = df_sample['category'] == category
        mask_array = mask.values
        ax.scatter(embeddings_2d[mask_array, 0], embeddings_2d[mask_array, 1],
                  c=[colors[i]], label=category, alpha=0.6, s=20, edgecolors='none')

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('t-SNE: Non-linear Reduction Reveals Clear Clusters',
                fontsize=14, fontweight='bold', color=MLPURPLE)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('charts/tsne_visualization.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: charts/tsne_visualization.pdf")

def generate_clustering_comparison(df, embeddings):
    print("\n[5/6] Generating clustering comparison...")

    # PCA for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    categories = df['category'].unique()
    cat_colors = [MLPURPLE, MLBLUE, MLORANGE, MLGREEN]
    cluster_colors = plt.cm.Set2(np.linspace(0, 1, 4))

    # Left: Actual categories
    for i, category in enumerate(categories):
        mask = df['category'] == category
        axes[0].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=[cat_colors[i]], label=category, alpha=0.6, s=12, edgecolors='none')

    axes[0].set_xlabel('PC1', fontsize=11)
    axes[0].set_ylabel('PC2', fontsize=11)
    axes[0].set_title('Actual Categories', fontsize=13, fontweight='bold', color=MLPURPLE)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.2)

    # Right: K-means clusters
    for i in range(4):
        mask = clusters == i
        axes[1].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=[cluster_colors[i]], label=f'Cluster {i}', alpha=0.6, s=12, edgecolors='none')

    axes[1].set_xlabel('PC1', fontsize=11)
    axes[1].set_ylabel('PC2', fontsize=11)
    axes[1].set_title('K-Means Clusters (Unsupervised)', fontsize=13, fontweight='bold', color=MLPURPLE)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('charts/clustering_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: charts/clustering_comparison.pdf")

def generate_similarity_distribution(df, embeddings):
    print("\n[6/6] Generating similarity distribution...")

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    category_similarities = {}
    between_category_sims = []

    for cat in df['category'].unique():
        cat_indices = df[df['category'] == cat].index[:100].tolist()

        within_sims = []
        for i in range(len(cat_indices)):
            for j in range(i+1, min(i+15, len(cat_indices))):
                sim = cosine_similarity(embeddings[cat_indices[i]], embeddings[cat_indices[j]])
                within_sims.append(sim)

        category_similarities[cat] = within_sims

        # Between categories
        other_cats = df[df['category'] != cat].index[:50].tolist()
        for i in cat_indices[:25]:
            for j in other_cats[:10]:
                sim = cosine_similarity(embeddings[i], embeddings[j])
                between_category_sims.append(sim)

    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot = [sims for sims in category_similarities.values()]
    data_to_plot.append(between_category_sims)

    labels = list(category_similarities.keys()) + ['Between\nCategories']
    colors_box = [MLPURPLE, MLBLUE, MLORANGE, MLGREEN, MLGRAY]

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)

    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors_box[i])
        box.set_alpha(0.7)
        box.set_edgecolor(MLPURPLE)
        box.set_linewidth(1.5)

    for whisker in bp['whiskers']:
        whisker.set(color=MLPURPLE, linewidth=1.2)
    for cap in bp['caps']:
        cap.set(color=MLPURPLE, linewidth=1.2)
    for median in bp['medians']:
        median.set(color='darkred', linewidth=2)

    ax.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
    ax.set_title('Within-Category vs Between-Category Similarity',
                fontsize=14, fontweight='bold', color=MLPURPLE)
    ax.grid(True, alpha=0.2, axis='y')
    plt.xticks(fontsize=11)

    plt.tight_layout()
    plt.savefig('charts/similarity_distribution.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: charts/similarity_distribution.pdf")

def generate_pca_variance_explained(embeddings):
    print("\n[7/9] Generating PCA variance explained (scree plot)...")

    # Compute PCA with 50 components
    pca = PCA(n_components=50)
    pca.fit(embeddings)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar chart of individual variance
    ax.bar(range(1, 51), pca.explained_variance_ratio_,
           alpha=0.6, color=MLLAVENDER2, edgecolor=MLPURPLE, linewidth=1)

    # Line chart of cumulative variance
    ax2 = ax.twinx()
    ax2.plot(range(1, 51), cumulative_variance * 100,
            color=MLPURPLE, linewidth=2.5, marker='o', markersize=4, label='Cumulative')

    # Highlight key points
    ax2.axhline(y=50, color=MLRED, linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.text(52, 50, '50%', fontsize=10, color=MLRED, va='center')

    # Find components needed for 50% variance
    n_components_50 = np.argmax(cumulative_variance >= 0.5) + 1
    ax2.axvline(x=n_components_50, color=MLRED, linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.text(n_components_50, 10, f'{n_components_50} dims', fontsize=9,
            color=MLRED, ha='center', rotation=90)

    ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variance Explained (Individual)', fontsize=11, color=MLLAVENDER2)
    ax2.set_ylabel('Cumulative Variance (%)', fontsize=11, color=MLPURPLE, fontweight='bold')
    ax.set_title('PCA: How Many Dimensions Do We Need?',
                fontsize=14, fontweight='bold', color=MLPURPLE, pad=15)
    ax.set_xlim(0, 51)
    ax2.set_ylim(0, 100)
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis='y', labelcolor=MLLAVENDER2)
    ax2.tick_params(axis='y', labelcolor=MLPURPLE)

    plt.tight_layout()
    plt.savefig('charts/pca_variance_explained.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: charts/pca_variance_explained.pdf")

def generate_similarity_heatmap(df, embeddings):
    print("\n[8/9] Generating similarity heatmap...")

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Select one example headline from each category
    categories = df['category'].unique()
    examples = []
    example_embeddings = []

    for cat in categories:
        idx = df[df['category'] == cat].index[0]
        headline = df.loc[idx, 'headline']
        # Truncate long headlines
        if len(headline) > 35:
            headline = headline[:32] + '...'
        examples.append(headline)
        example_embeddings.append(embeddings[idx])

    # Compute similarity matrix
    n = len(examples)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = cosine_similarity(
                example_embeddings[i], example_embeddings[j]
            )

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(similarity_matrix, cmap='RdPu', vmin=0, vmax=1, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', fontsize=11, fontweight='bold')

    # Add text annotations
    for i in range(n):
        for j in range(n):
            text_color = 'white' if similarity_matrix[i, j] > 0.6 else 'black'
            ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                   ha='center', va='center', color=text_color,
                   fontsize=11, fontweight='bold')

    # Set ticks and labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(examples, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(examples, fontsize=9)

    # Add category labels
    for i, cat in enumerate(categories):
        ax.text(-0.7, i, cat, fontsize=8, color=MLPURPLE,
               fontweight='bold', ha='right', va='center')
        ax.text(i, -0.7, cat, fontsize=8, color=MLPURPLE,
               fontweight='bold', ha='center', va='bottom', rotation=45)

    ax.set_title('Similarity Between Example Headlines',
                fontsize=14, fontweight='bold', color=MLPURPLE, pad=20)

    plt.tight_layout()
    plt.savefig('charts/similarity_heatmap.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: charts/similarity_heatmap.pdf")

def generate_semantic_neighborhood(df, embeddings):
    print("\n[9/9] Generating semantic neighborhood...")

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Find a good query example about politics/president
    query_idx = None
    for idx, row in df.iterrows():
        if 'president' in row['headline'].lower() or 'announces' in row['headline'].lower():
            if row['category'] == 'Politics':
                query_idx = idx
                break

    # Fallback: just use first politics headline
    if query_idx is None:
        query_idx = df[df['category'] == 'Politics'].index[0]

    query_headline = df.loc[query_idx, 'headline']
    query_embedding = embeddings[query_idx]

    # Compute similarities to all other headlines
    similarities = []
    for idx in range(len(embeddings)):
        if idx != query_idx:
            sim = cosine_similarity(query_embedding, embeddings[idx])
            similarities.append((idx, sim))

    # Sort and get top 10
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_10 = similarities[:10]

    # Prepare data for plotting
    neighbor_headlines = []
    neighbor_sims = []
    neighbor_categories = []

    for idx, sim in top_10:
        headline = df.loc[idx, 'headline']
        if len(headline) > 45:
            headline = headline[:42] + '...'
        neighbor_headlines.append(headline)
        neighbor_sims.append(sim)
        neighbor_categories.append(df.loc[idx, 'category'])

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(11, 7))

    # Color bars by category
    colors_map = {'Politics': MLPURPLE, 'Sports': MLBLUE,
                  'Technology': MLORANGE, 'Entertainment': MLGREEN}
    bar_colors = [colors_map.get(cat, MLGRAY) for cat in neighbor_categories]

    y_pos = np.arange(len(neighbor_headlines))
    bars = ax.barh(y_pos, neighbor_sims, color=bar_colors, alpha=0.7,
                   edgecolor=MLPURPLE, linewidth=1.5)

    # Add similarity values
    for i, (bar, sim) in enumerate(zip(bars, neighbor_sims)):
        ax.text(sim + 0.01, i, f'{sim:.3f}', va='center', fontsize=9,
               fontweight='bold', color=MLPURPLE)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(neighbor_headlines, fontsize=9)
    ax.set_xlabel('Cosine Similarity', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.2, axis='x')

    # Title with query
    query_short = query_headline if len(query_headline) <= 50 else query_headline[:47] + '...'
    ax.set_title(f'Top 10 Similar Headlines\nQuery: ``{query_short}``',
                fontsize=13, fontweight='bold', color=MLPURPLE, pad=15)

    # Add legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_map[cat], edgecolor=MLPURPLE,
                            label=cat, alpha=0.7)
                      for cat in ['Politics', 'Sports', 'Technology', 'Entertainment']]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig('charts/semantic_neighborhood.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: charts/semantic_neighborhood.pdf")

def main():
    print("=" * 70)
    print("Generating Charts for Beamer Presentation")
    print("=" * 70)

    ensure_charts_dir()

    # Generate conceptual diagrams first (no data needed)
    generate_embedding_concept_diagram()
    generate_cosine_similarity_diagram()

    # Load data and embeddings
    df, embeddings = load_data_and_embeddings()

    # Generate original visualizations
    generate_pca_visualization(df, embeddings)
    generate_tsne_visualization(df, embeddings)
    generate_clustering_comparison(df, embeddings)
    generate_similarity_distribution(df, embeddings)

    # Generate new visualizations
    generate_pca_variance_explained(embeddings)
    generate_similarity_heatmap(df, embeddings)
    generate_semantic_neighborhood(df, embeddings)

    print("\n" + "=" * 70)
    print("All charts generated successfully!")
    print("=" * 70)
    print("\nGenerated files in charts/:")
    print("  1. embedding_concept.pdf")
    print("  2. cosine_similarity_example.pdf")
    print("  3. pca_visualization.pdf")
    print("  4. tsne_visualization.pdf")
    print("  5. clustering_comparison.pdf")
    print("  6. similarity_distribution.pdf")
    print("  7. pca_variance_explained.pdf")
    print("  8. similarity_heatmap.pdf")
    print("  9. semantic_neighborhood.pdf")
    print("\nReady for Beamer presentation!")

if __name__ == "__main__":
    main()
