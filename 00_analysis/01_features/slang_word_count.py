# =============================================================================
# SLANG WORD COUNT FEATURE
# =============================================================================
# Purpose: This script creates a new feature (column) called 'slang_word_count'
# for the Jigsaw toxic comment dataset. It counts how many toxic/offensive
# slang words appear in each comment.
#
# Author: Helena Kandjumbwa
# Branch: feature/slang-word-count
#
# =============================================================================
# REFERENCES & METHODOLOGY
# =============================================================================
# The toxic slang list used in this script was curated using the following
# sources and methodology:
#
# [1] Watanabe, H., Bouazizi, M., & Ohtsuki, T. (2018). "Hate Speech on
#     Twitter: A Pragmatic Approach to Collect Hateful and Offensive Expressions
#     and Perform Hate Speech Detection." IEEE Access, 6, pp. 13825-13835.
#     DOI: 10.1109/ACCESS.2018.2806394
#     --> Establishes the methodology of using slang words as a sentiment-based
#         feature for hate speech detection. The authors extract features
#         including "slang words, emoticons, hashtags" to train classifiers.
#
# [2] Wiktionary: Appendix of English Internet Slang
#     URL: https://en.wiktionary.org/wiki/Appendix:English_internet_slang
#     --> A community-maintained comprehensive list of internet slang and
#         abbreviations. Used as a base reference for identifying internet-
#         specific acronyms and slang terms.
#
# [3] NoSwearing.com (part of AllSlang family)
#     URL: https://www.noswearing.com/
#     --> A crowdsourced dictionary of profane and offensive language. Used
#         to cross-reference and filter slang terms for toxicity relevance.
#
# [4] HateBase.org (retired 2022)
#     URL: https://hatebase.org/
#     --> A formerly active collaborative repository of multilingual hate
#         speech, built in partnership with the Dark Data Project and The
#         Sentinel Project. While no longer actively maintained, its
#         methodology of cataloging hate-related vocabulary informed our
#         term selection approach.
#
# [5] Keshari, N., Malladi, D., & Mittal, U. (2023). "Hate Speech Detection
#     Using Natural Language Processing." Stanford CS224N Custom Project.
#     URL: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/
#          final-reports/final-report-169358304.pdf
#     --> References Watanabe et al. [1] and uses sentiment-based features
#         (including slang) for hate speech classification. Provides the
#         broader methodological context for our feature engineering approach.
#
# METHODOLOGY:
# Terms were drawn from Wiktionary [2] and NoSwearing [3], filtered for
# toxicity relevance using the feature extraction approach described by
# Watanabe et al. [1], and cross-referenced with the HateBase [4] framework
# of offensive vocabulary classification. The final list represents a curated
# subset of toxic/offensive internet slang — NOT an exhaustive collection.
#
# KNOWN LIMITATIONS:
# - Slang evolves constantly; this list captures a snapshot, not all terms.
# - Some terms have dual meanings depending on context (e.g., "gtfo" can
#   express surprise OR hostility). Some noise is expected.
# - The list focuses on English-language internet slang abbreviations and
#   may not capture slang from other languages or dialects.
# =============================================================================


# --- STEP 1: IMPORT LIBRARIES -----------------------------------------------
# pandas is the main library for working with tabular data (like spreadsheets)
# re is Python's built-in library for "regular expressions" (pattern matching in text)
import pandas as pd
import re


# --- STEP 2: LOAD THE DATASET -----------------------------------------------
# We use a relative path from this file's location (00_analysis/01_features/)
# to the data folder (01_data/00_raw/jigsaw-dataset/train.csv)
# The ".." means "go up one folder"
#   From 01_features/ -> go up to 00_analysis/ -> go up to project root/
#   Then down into 01_data/00_raw/jigsaw-dataset/
DATA_PATH = "../../01_data/00_raw/jigsaw-dataset/train.csv"

# Read the CSV file into a pandas DataFrame (basically a table)
df = pd.read_csv(DATA_PATH)

# Let's print the first few rows so we can see what the data looks like
print("=== First 5 rows of the dataset ===")
print(df.head())
print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Column names: {list(df.columns)}")


# --- STEP 3: DEFINE THE TOXIC SLANG LIST ------------------------------------
# This is a curated list of slang terms commonly associated with toxic or
# offensive online speech. These are abbreviations and internet slang that
# a standard profanity filter might miss.
#
# Each term is annotated with:
#   - Its meaning
#   - Its primary source(s) from the references above
#
# The terms are grouped into categories for clarity.

TOXIC_SLANG_LIST = [

    # ----- Category A: Direct threats / self-harm slang -----
    # These are unambiguously toxic. Sources: [2], [3], [4]
    "kys",      # "kill yourself" — extremely toxic, direct threat
    "kms",      # "kill myself" — self-harm related

    # ----- Category B: Aggressive dismissals / insults -----
    # Abbreviations that tell someone to go away aggressively or shut up.
    # Sources: [2], [3]
    "stfu",     # "shut the f*** up"
    "gtfo",     # "get the f*** out" — NOTE: can also express surprise
    "foh",      # "f*** outta here"
    "gfys",     # "go f*** yourself"
    "gfy",      # short version of above
    "stfd",     # "sit the f*** down"

    # ----- Category C: Death wishes / extreme hostility -----
    # Sources: [2], [3]
    "diaf",     # "die in a fire"
    "esad",     # "eat s*** and die"
    "foad",     # "f*** off and die"

    # ----- Category D: Profane expressions of frustration/anger -----
    # These contain profanity and often appear in hostile contexts,
    # though some can also appear in non-toxic usage.
    # Sources: [2], [3]
    "wtf",      # "what the f***"
    "ffs",      # "for f***'s sake"
    "omfg",     # "oh my f***ing god"
    "jfc",      # "Jesus f***ing Christ"
    "tf",       # "the f***" — often aggressive
    "smfh",     # "shaking my f***ing head"
    "fml",      # "f*** my life"

    # ----- Category E: Derogatory name-calling via slang -----
    # Terms used specifically to demean or insult people.
    # Sources: [3], [4]
    "pos",      # "piece of s***"
    "sob",      # "son of a b****"
    "thot",     # derogatory slang for a person (usually women)
    "incel",    # used in toxic/hateful contexts
    "simp",     # used to mock or demean
    "cuck",     # derogatory political/social slang
    "libtard",  # political slur combining "liberal" + "retard"
    "retard",   # ableist slur frequently used as internet slang
    "tard",     # shortened version of above

    # ----- Category F: Dismissive / contemptuous expressions -----
    # These express indifference or contempt. Sources: [2], [3]
    "idgaf",    # "I don't give a f***"
    "dgaf",     # short version of above
    "rtfm",     # "read the f***ing manual" — dismissive

    # ----- Category G: Sexually aggressive slang -----
    # Sources: [3], [4]
    "smd",      # "suck my d***"

    # ----- Category H: Mocking / aggressive reactions -----
    # These CAN appear in non-toxic contexts but frequently accompany
    # toxic comments. Included with the caveat that they add noise.
    # Sources: [2], [3]
    "lmao",     # "laughing my a** off" — often used in mocking
    "lmfao",    # variation of above
    "smh",      # "shaking my head" — disapproval/dismissal
    "af",       # "as f***" — intensifier, often in hostile remarks
]

# Remove any accidental duplicates from the list
# (set() removes duplicates, list() converts back)
TOXIC_SLANG_LIST = list(set(TOXIC_SLANG_LIST))

# Print the total number of terms for transparency
print(f"\n=== Toxic slang list: {len(TOXIC_SLANG_LIST)} unique terms ===")


# --- STEP 4: BUILD THE SLANG COUNTING FUNCTION ------------------------------
# This function takes a single comment (a string of text) and returns
# an integer: how many words in that comment match our slang list.

def count_slang_words(comment):
    """
    Count the number of toxic slang words in a given comment.

    Methodology follows Watanabe et al. (2018) [1], who used slang words
    as one of several sentiment-based features for hate speech detection.

    Parameters:
        comment (str): A single comment text from the dataset.

    Returns:
        int: The number of slang words found in the comment.
    """

    # Safety check: if the comment is not a string (e.g., NaN/missing value),
    # return 0 because there's nothing to count
    if not isinstance(comment, str):
        return 0

    # Convert the comment to lowercase so matching is case-insensitive
    # e.g., "KYS" and "kys" should both be caught
    comment_lower = comment.lower()

    # Use regex to split the comment into individual words
    # \b means "word boundary" — this splits on spaces, punctuation, etc.
    # re.findall(r'\b\w+\b', text) extracts all "words" from the text
    # For example: "you're a pos!!!" becomes ["you", "re", "a", "pos"]
    words = re.findall(r'\b\w+\b', comment_lower)

    # Count how many of those words appear in our slang list
    # This loops through each word and checks: is it in TOXIC_SLANG_LIST?
    # If yes, it gets counted. If no, it's skipped.
    slang_count = sum(1 for word in words if word in TOXIC_SLANG_LIST)

    return slang_count


# --- STEP 5: APPLY THE FUNCTION TO THE ENTIRE DATASET -----------------------
# .apply() runs our function on every single row in the 'comment_text' column
# and stores the result in a new column called 'slang_word_count'
# This is like dragging a formula down in Excel — it runs for all ~160k rows

print("\n=== Counting slang words across all comments... ===")
df['slang_word_count'] = df['comment_text'].apply(count_slang_words)
print("Done!")


# --- STEP 6: INSPECT THE RESULTS --------------------------------------------

# Show basic statistics about the new feature
print("\n=== Slang Word Count — Summary Statistics ===")
print(df['slang_word_count'].describe())

# How many comments have at least one slang word?
comments_with_slang = (df['slang_word_count'] > 0).sum()
total_comments = len(df)
print(f"\nComments with at least 1 slang word: {comments_with_slang} out of {total_comments}")
print(f"That's {comments_with_slang / total_comments * 100:.2f}% of all comments")

# Show some examples of comments that scored high on slang count
print("\n=== Top 10 comments by slang word count ===")
top_slang = df.nlargest(10, 'slang_word_count')[['comment_text', 'slang_word_count']]
for index, row in top_slang.iterrows():
    # Only print first 100 characters of each comment to keep output readable
    preview = row['comment_text'][:100] + "..." if len(row['comment_text']) > 100 else row['comment_text']
    print(f"  Slang count: {row['slang_word_count']} | Comment: {preview}")

# Show a few examples comparing toxic vs non-toxic comments
print("\n=== Average slang count: Toxic vs Non-Toxic comments ===")
print(f"  Toxic comments (toxic=1):     {df[df['toxic'] == 1]['slang_word_count'].mean():.3f}")
print(f"  Non-toxic comments (toxic=0): {df[df['toxic'] == 0]['slang_word_count'].mean():.3f}")

# Show the first few rows with the new column
print("\n=== First 10 rows with new slang_word_count column ===")
print(df[['comment_text', 'toxic', 'slang_word_count']].head(10).to_string())


# --- STEP 7: SAVE THE RESULT ------------------------------------------------
# Save the updated dataset with the new feature to the processed data folder
OUTPUT_PATH = "../../01_data/01_processed/train_with_slang_count.csv"
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n=== Saved updated dataset to: {OUTPUT_PATH} ===")
print("You now have a new column 'slang_word_count' in your dataset!")
