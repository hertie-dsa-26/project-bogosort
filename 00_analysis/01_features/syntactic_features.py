import pandas as pd
import re
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# NEGATION FEATURE
# ---------------------------------------------------------------------------

NEGATION_WORDS = {
    "not", "never", "no", "nobody", "nothing", "neither",
    "nowhere", "nor", "cannot", "can't", "won't", "wouldn't",
    "shouldn't", "couldn't", "didn't", "doesn't", "don't",
    "isn't", "aren't", "wasn't", "weren't", "hadn't", "hasn't",
    "haven't", "needn't", "mustn't"
}

def count_negations(text: str) -> int:
    """Count the number of negation words in a comment."""
    if not isinstance(text, str):
        return 0
    tokens = re.findall(r"\b\w+(?:'\w+)?\b", text.lower())
    return sum(1 for token in tokens if token in NEGATION_WORDS)


# ---------------------------------------------------------------------------
# SENTENCE COUNT FEATURE
# ---------------------------------------------------------------------------

def count_sentences(text: str) -> int:
    """Count the number of sentences in a comment."""
    if not isinstance(text, str):
        return 0
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    return len(sentences)


# ---------------------------------------------------------------------------
# AVERAGE SENTENCE LENGTH FEATURE
# ---------------------------------------------------------------------------

def average_sentence_length(text: str) -> float:
    """Calculate the average number of words per sentence in a comment."""
    if not isinstance(text, str):
        return 0.0
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) == 0:
        return 0.0
    word_counts = [len(s.split()) for s in sentences]
    return sum(word_counts) / len(word_counts)


# ---------------------------------------------------------------------------
# COMBINED FEATURE ADDER
# ---------------------------------------------------------------------------

def add_syntactic_features(df: pd.DataFrame, text_col: str = "comment_text") -> pd.DataFrame:
    """Add all 3 syntactic features to the dataframe."""
    df = df.copy()
    df["negation_count"] = df[text_col].apply(count_negations)
    df["sentence_count"] = df[text_col].apply(count_sentences)
    df["avg_sentence_length"] = df[text_col].apply(average_sentence_length)
    return df


# ---------------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(BASE_DIR, "..", "..", "..", "Dataset", "train.csv"))

    df = add_syntactic_features(df)

    features = ["negation_count", "sentence_count", "avg_sentence_length"]
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    # -----------------------------------------------------------------------
    # OVERALL MEANS
    # -----------------------------------------------------------------------
    print("=== Overall Means ===")
    for feature in features:
        print(f"{feature}: {df[feature].mean():.3f}")

    # -----------------------------------------------------------------------
    # MEAN PER LABEL
    # -----------------------------------------------------------------------
    print("\n=== Mean per Label ===")
    for feature in features:
        print(f"\n--- {feature} ---")
        for label in labels:
            means = df.groupby(label)[feature].mean()
            print(f"  {label}: non-toxic={means[0]:.3f}, toxic={means[1]:.3f}")

    # -----------------------------------------------------------------------
    # CORRELATION BETWEEN FEATURES
    # -----------------------------------------------------------------------
    print("\n=== Correlation Between Features ===")
    print(df[features].corr().round(3))

    # -----------------------------------------------------------------------
    # TOP 5 MOST EXTREME COMMENTS PER FEATURE
    # -----------------------------------------------------------------------
    print("\n=== Top 5 Most Extreme Comments Per Feature ===")
    for feature in features:
        print(f"\n--- Top 5 highest {feature} ---")
        top5 = df.nlargest(5, feature)[["comment_text", feature]]
        for _, row in top5.iterrows():
            preview = row["comment_text"][:80].replace("\n", " ")
            print(f"  [{row[feature]:.1f}] {preview}...")

    # -----------------------------------------------------------------------
    # PLOTS
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 6, figsize=(20, 10))
    for i, feature in enumerate(features):
        for j, label in enumerate(labels):
            df.groupby(label)[feature].mean().plot(
                kind="bar", ax=axes[i][j],
                title=f"{label}",
                color=["steelblue", "tomato"]
            )
            axes[i][j].set_xlabel("")
            axes[i][j].set_xticklabels(["Non-toxic", "Toxic"], rotation=0)
            if j == 0:
                axes[i][j].set_ylabel(feature, fontsize=9)

    plt.suptitle("Mean Syntactic Features by Label", fontsize=14)
    plt.tight_layout()
    plt.savefig("syntactic_analysis.png")
    print("\nPlot saved as syntactic_analysis.png")

    # Correlation heatmap
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    corr = df[features].corr()
    im = ax2.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax2.set_xticks(range(len(features)))
    ax2.set_yticks(range(len(features)))
    ax2.set_xticklabels(features, rotation=45, ha="right")
    ax2.set_yticklabels(features)
    for i in range(len(features)):
        for j in range(len(features)):
            ax2.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=12)
    plt.colorbar(im, ax=ax2)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    print("Correlation heatmap saved as correlation_heatmap.png")