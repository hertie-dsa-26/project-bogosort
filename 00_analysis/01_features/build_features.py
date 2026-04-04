"""
build_features.py — sklearn-compatible feature pipeline for the Jigsaw toxicity dataset.

All transformers follow the sklearn API (BaseEstimator + TransformerMixin),
meaning they plug directly into sklearn Pipeline and handle train/test
separation automatically.

Usage
-----
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from build_features import DenseFeatureTransformer, TfidfTransformer

    pipe = Pipeline([
        ("dense", DenseFeatureTransformer()),
        ("tfidf", TfidfTransformer()),
        ("model", RandomForestClassifier()),
    ])

    pipe.fit(df_train, y_train)
    pipe.predict(df_test)

Feature outputs (DenseFeatureTransformer)
-----------------------------------------
  1.  Sentiment      vader_compound, vader_neg, vader_pos,
                     vader_is_negative, vader_intensity, vader_pos_minus_neg
  2.  Second-person  has_second_person, second_person_count, second_person_density
  3.  Profanity      profanity_count, obfuscated_profanity_count
  4.  Slang          slang_count
  5.  Text shape     char_count, word_count, exclamation_count, uppercase_ratio
  6.  Unique words   unique_word_ratio
  7.  Elongation     elongated_token_count, consecutive_punct_count
  8.  URLs / IPs     url_count, ip_count, has_url_or_ip
  9.  Syntactic      negation_count, sentence_count, avg_sentence_length
  10. Identity       identity_mention_count, identity_race, identity_gender,
                     identity_sexuality, identity_religion, identity_disability,
                     identity_nationality
"""

import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

TEXT_COL = "comment_text"  # Name of data column, in case we change later


# ===========================================================================
# Lexicons and compiled patterns
# Defined at module level so they are built exactly once on import.
# frozenset gives O(1) membership checks and is immutable (cannot be
# accidentally modified at runtime).
# ===========================================================================

_PROFANITY_LEXICON = frozenset({
    "ass", "asshole", "bastard", "bitch", "bullshit", "crap", "cunt",
    "damn", "dick", "dumbass", "fuck", "fucker", "fucking", "idiot",
    "jackass", "loser", "moron", "retard", "shit", "slut", "stupid",
    "trash", "whore",
})

# Maps obfuscation characters to letter equivalents for leetspeak normalisation.
# Used with str.translate() for a single-pass substitution (faster than chained replace).
_LEETSPEAK_MAP = str.maketrans({
    "@": "a", "$": "s", "5": "s", "0": "o",
    "1": "i", "!": "i", "3": "e", "7": "t", "+": "t",
})

_TOXIC_SLANG = frozenset({
    # Direct threats / self-harm
    "kys", "kms",
    # Aggressive dismissals
    "stfu", "gtfo", "gtfoh", "foh", "gfys", "gfy", "stfd", "gth",
    "rekt", "pwned", "noob",
    # Death wishes
    "diaf", "esad", "foad",
    # Profane expressions (abbreviations a standard filter misses)
    "wtf", "ffs", "omfg", "jfc", "tf", "smfh", "smmfh", "fml",
    "mf", "mofo", "bs", "fu", "fy",
    # Derogatory name-calling
    "pos", "sob", "thot", "incel", "simp", "cuck", "libtard", "retard", "tard",
    "snowflake", "feminazi", "sjw", "npc",
    # Dismissive / contemptuous
    "idgaf", "dgaf", "rtfm", "cope", "seethe",
    # Sexually aggressive
    "smd",
    # Transphobic slang
    "dilate",
    # Mocking reactions
    "lmao", "lmfao", "smh", "af", "kek",
})

_NEGATION_WORDS = frozenset({
    "not", "never", "no", "nobody", "nothing", "neither", "nowhere", "nor",
    "cannot", "can't", "won't", "wouldn't", "shouldn't", "couldn't",
    "didn't", "doesn't", "don't", "isn't", "aren't", "wasn't", "weren't",
    "hadn't", "hasn't", "haven't", "needn't", "mustn't",
})

# Pre-compiled regex patterns — built once, reused across all rows.

# Feature 2 — has_second_person, second_person_count, second_person_density
# \b = word boundary, ensures "you" does not match inside "youtube".
_SECOND_PERSON_RE = re.compile(
    r"\b(you|your|yours|yourself|yourselves"
    r"|you're|you'll|you've|you'd|ur|u)\b",
    re.IGNORECASE,
)

# Feature 8 — url_count, has_url_or_ip
_URL_RE = re.compile(r"(?:https?://[^\s]+)|(www\.[^\s]+)", re.IGNORECASE)

# Feature 8 — ip_count, has_url_or_ip
# Each octet validated to 0-255 via alternation pattern.
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
)

# Feature 3 — profanity_count
# Matches whole words made of letters and apostrophes.
_WORD_RE = re.compile(r"\b[a-zA-Z']+\b")

# Feature 9 — negation_count
# Matches whole words including contractions (e.g. "can't", "wouldn't").
_NEGATION_RE = re.compile(r"\b\w+(?:'\w+)?\b")

# Feature 7 — elongated_token_count
# Matches any character that repeats 3+ times consecutively (e.g. "coooool").
_ELONGATE_RE = re.compile(r"(.)\1{2,}", re.IGNORECASE)

# Feature 7 — consecutive_punct_count
# Matches runs of 2+ consecutive punctuation characters (e.g. "!!", "???").
_PUNCT_RE = re.compile(r"[^\w\s]{2,}")

# Feature 9 — sentence_count, avg_sentence_length
# Splits text into sentences on . ! or ? characters.
_SENTENCE_RE = re.compile(r"[.!?]+")


# ===========================================================================
# Identity group lexicon
#
# Six categories, each compiled into its own regex pattern.
# Presence alone is a weak signal on Wikipedia data — the feature earns
# value in combination with sentiment features.
# ===========================================================================

_IDENTITY_CATEGORIES: dict = {

    "race": [
        "black", "african", "african american", "afro-american", "negro", "colored",
        "white", "caucasian", "european",
        "asian", "east asian", "southeast asian", "south asian",
        "chinese", "japanese", "korean", "vietnamese", "filipino", "thai",
        "indonesian", "malaysian", "singaporean",
        "indian", "pakistani", "bangladeshi", "sri lankan", "nepalese",
        "hispanic", "latino", "latina", "latinx", "chicano", "chicana", "mestizo",
        "mexican", "puerto rican", "cuban", "dominican", "salvadoran", "guatemalan",
        "colombian", "venezuelan", "peruvian", "argentinian", "brazilian", "bolivian",
        "ecuadorian",
        "arab", "arabic", "middle eastern", "persian", "iranian", "turkish",
        "kurdish", "armenian", "lebanese", "syrian", "egyptian", "moroccan",
        "algerian", "libyan", "tunisian",
        "jewish", "jew",
        "slavic", "russian", "ukrainian", "polish", "romanian", "hungarian",
        "czech", "slovak", "serbian", "croatian",
        "indigenous", "native american", "american indian", "first nations",
        "aboriginal",
        "pacific islander", "hawaiian", "samoan", "tongan", "fijian", "maori",
        "romani", "gypsy",
        "biracial", "multiracial", "mixed race",
    ],

    "gender": [
        "woman", "women", "female", "girl", "lady", "ladies", "femme",
        "man", "men", "male", "boy",
        "transgender", "trans", "transwoman", "transman", "trans woman", "trans man",
        "transsexual",
        "nonbinary", "non-binary", "genderqueer", "gender queer",
        "genderfluid", "gender fluid", "gender nonconforming",
        "agender", "bigender", "enby",
        "intersex",
        "cisgender", "cis",
        "butch",
    ],

    "sexuality": [
        "gay", "lesbian", "bisexual", "bi", "pansexual", "pan",
        "asexual", "ace", "queer", "homosexual", "heterosexual", "straight",
        "demisexual", "aromantic",
        "lgbt", "lgbtq", "lgbtqia", "lgbtq+",
        "same-sex", "same sex",
        "sodomite",
    ],

    "religion": [
        "christian", "christianity", "catholic", "catholicism", "protestant",
        "evangelical", "baptist", "lutheran", "methodist", "presbyterian",
        "anglican", "episcopal", "orthodox",
        "mormon", "latter-day saint", "jehovah",
        "muslim", "islam", "islamic", "sunni", "shia", "shiite", "islamist",
        "jewish", "judaism", "hasidic", "zionist", "antisemit",
        "hindu", "hinduism",
        "buddhist", "buddhism",
        "sikh", "sikhism",
        "atheist", "atheism", "agnostic", "agnosticism", "secular", "irreligious",
        "pagan", "wiccan", "wicca", "scientologist", "scientology",
        "satanist",
    ],

    "disability": [
        "disabled", "disability", "handicapped",
        "autistic", "autism", "autism spectrum", "asperger", "adhd",
        "attention deficit", "dyslexic", "dyslexia",
        "bipolar", "schizophrenic", "schizophrenia", "mentally ill",
        "mental illness", "mental health", "psychotic", "paranoid",
        "blind", "visually impaired", "deaf", "hard of hearing",
        "wheelchair", "wheelchair user", "amputee", "paraplegic",
        "quadriplegic", "cerebral palsy",
        "down syndrome", "downs syndrome", "alzheimer", "dementia",
        "epileptic", "epilepsy", "chronic illness", "chronic pain",
    ],

    "nationality": [
        "immigrant", "immigrants", "migrant", "migrants", "migration",
        "refugee", "refugees", "asylum seeker", "asylum seekers",
        "undocumented", "illegal immigrant", "illegal immigrants",
        "illegal alien", "foreigner", "foreigners", "deported",
        "american", "british", "french", "german",
        "irish", "italian", "greek", "spanish",
    ],
}


def _build_identity_pattern(terms: list) -> re.Pattern:
    """
    Build one compiled regex from a list of identity terms.
    Multi-word terms (e.g. 'native american') use \\s+ so they match
    regardless of exact spacing. Single-word terms get plain \\b boundaries.
    """
    parts = []
    for term in terms:
        words = term.split()
        if len(words) == 1:
            parts.append(re.escape(term))
        else:
            parts.append(r"\s+".join(re.escape(w) for w in words))
    return re.compile(r"\b(" + "|".join(parts) + r")\b", re.IGNORECASE)


# One pattern per category (for per-category binary flags)
_IDENTITY_PATTERNS: dict = {
    cat: _build_identity_pattern(terms)
    for cat, terms in _IDENTITY_CATEGORIES.items()
}

# One combined pattern (for total mention count across all categories)
_IDENTITY_ALL_RE: re.Pattern = _build_identity_pattern([
    term for terms in _IDENTITY_CATEGORIES.values() for term in terms
])


# ===========================================================================
# Row-level feature functions (private — called by DenseFeatureTransformer)
# Each accepts a single pre-cleaned string and returns a scalar or dict.
# ===========================================================================

# --- 1. Sentiment ---
# SentimentIntensityAnalyzer loads a lexicon from disk. Using it as a
# default argument creates it once at function definition time — not per call.
def _extract_sentiment(text: str, _sia=SentimentIntensityAnalyzer()) -> dict:
    scores   = _sia.polarity_scores(text)
    compound = scores["compound"]
    neg, pos = scores["neg"], scores["pos"]
    return {
        "vader_compound":      compound,
        "vader_neg":           neg,
        "vader_pos":           pos,
        "vader_is_negative":   int(compound < -0.05),  # VADER's own negativity threshold
        "vader_intensity":     abs(compound),           # emotional strength, sign-free
        "vader_pos_minus_neg": round(pos - neg, 6),    # net positivity; negative = more negative
    }


# --- 2. Second-person pronouns ---
def _extract_second_person(text: str) -> dict:
    matches    = _SECOND_PERSON_RE.findall(text)
    count      = len(matches)
    word_count = max(len(text.split()), 1)  # guard against empty string / div-by-zero
    return {
        "has_second_person":     int(count > 0),
        "second_person_count":   count,
        "second_person_density": round(count / word_count, 6),
    }


# --- 3. Profanity ---
def _normalize_leetspeak(token: str) -> str:
    # Translate obfuscation chars (e.g. $ -> s), then strip any remaining non-alpha.
    return re.sub(r"[^a-z]", "", token.lower().translate(_LEETSPEAK_MAP))

def _profanity_count(text: str) -> int:
    tokens = _WORD_RE.findall(text.lower())
    return sum(1 for t in tokens if t in _PROFANITY_LEXICON)

def _obfuscated_profanity_count(text: str) -> int:
    # Counts tokens that only match the lexicon after leetspeak translation —
    # i.e. the word was intentionally obfuscated to evade a plain filter.
    count = 0
    for raw in text.split():
        plain      = re.sub(r"[^a-z]", "", raw.lower())
        normalised = _normalize_leetspeak(raw)
        if normalised in _PROFANITY_LEXICON and plain not in _PROFANITY_LEXICON:
            count += 1
    return count


# --- 4. Slang ---
def _slang_count(text: str) -> int:
    words = re.findall(r"\b\w+\b", text.lower())
    return sum(1 for w in words if w in _TOXIC_SLANG)


# --- 5. Text shape ---
def _uppercase_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    upper = [w for w in words if w.isupper() and len(w) > 1]
    return len(upper) / len(words)


# --- 6. Unique word ratio ---
def _unique_word_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


# --- 7. Elongation + consecutive punctuation ---
def _elongated_token_count(text: str) -> int:
    return sum(1 for tok in text.split() if _ELONGATE_RE.search(tok))

def _consecutive_punct_count(text: str) -> int:
    return len(_PUNCT_RE.findall(text))


# --- 8. URLs / IPs ---
def _url_count(text: str) -> int:
    return len(_URL_RE.findall(text))

def _ip_count(text: str) -> int:
    return len(_IPV4_RE.findall(text))


# --- 9. Syntactic ---
def _negation_count(text: str) -> int:
    tokens = _NEGATION_RE.findall(text.lower())
    return sum(1 for t in tokens if t in _NEGATION_WORDS)

def _sentence_count(text: str) -> int:
    sents = [s for s in _SENTENCE_RE.split(text.strip()) if s.strip()]
    return len(sents)

def _avg_sentence_length(text: str) -> float:
    sents = [s for s in _SENTENCE_RE.split(text.strip()) if s.strip()]
    if not sents:
        return 0.0
    return sum(len(s.split()) for s in sents) / len(sents)


# --- 10. Identity group mentions ---
def _extract_identity(text: str) -> dict:
    total   = len(_IDENTITY_ALL_RE.findall(text))
    results = {"identity_mention_count": total}
    for cat, pattern in _IDENTITY_PATTERNS.items():
        results[f"identity_{cat}"] = int(bool(pattern.search(text)))
    return results


# ===========================================================================
# sklearn-compatible Transformers
#
# Each class inherits from BaseEstimator and TransformerMixin, giving it
# the fit() / transform() / fit_transform() interface sklearn expects.
#
# The Pipeline calls fit_transform() on train and transform() on test.
# DenseFeatureTransformer and BertTransformer are stateless so fit() does
# nothing. TfidfTransformer is stateful — fit() learns the vocabulary from
# training data, transform() applies it without refitting.
# ===========================================================================

class DenseFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Computes all row-level features and appends them as new columns.
    Stateless — fit() does nothing, transform() applies all feature functions.

    Input : pd.DataFrame with a 'comment_text' column.
    Output: pd.DataFrame with all original columns + feature columns appended.
    """

    def fit(self, X, y=None):
        return self  # nothing to learn

    def transform(self, X):
        df           = X.copy()
        df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
        texts        = df[TEXT_COL].tolist()

        # Single pass through all rows — collecting every feature into one dict
        records = []
        for text in texts:
            row = {}
            row.update(_extract_sentiment(text))       # 1. vader_*
            row.update(_extract_second_person(text))   # 2. second_person_*

            # 3. Profanity
            row["profanity_count"]            = _profanity_count(text)
            row["obfuscated_profanity_count"] = _obfuscated_profanity_count(text)

            # 4. Slang
            row["slang_count"] = _slang_count(text)

            # 5. Text shape
            words = text.split()
            row["char_count"]        = len(text)
            row["word_count"]        = len(words)
            row["exclamation_count"] = text.count("!")
            row["uppercase_ratio"]   = _uppercase_ratio(text)

            # 6. Unique word ratio
            row["unique_word_ratio"] = _unique_word_ratio(text)

            # 7. Elongation + consecutive punctuation
            row["elongated_token_count"]   = _elongated_token_count(text)
            row["consecutive_punct_count"] = _consecutive_punct_count(text)

            # 8. URLs / IPs
            url_n = _url_count(text)
            ip_n  = _ip_count(text)
            row["url_count"]     = url_n
            row["ip_count"]      = ip_n
            row["has_url_or_ip"] = int(url_n > 0 or ip_n > 0)

            # 9. Syntactic
            row["negation_count"]      = _negation_count(text)
            row["sentence_count"]      = _sentence_count(text)
            row["avg_sentence_length"] = _avg_sentence_length(text)

            row.update(_extract_identity(text))        # 10. identity_*

            records.append(row)

        feature_df = pd.DataFrame(records, index=df.index)
        return pd.concat([df, feature_df], axis=1)


class TfidfTransformer(BaseEstimator, TransformerMixin):
    """
    Word n-gram + character n-gram TF-IDF features.

    fit()       learns vocabulary from training texts.
    transform() applies the learned vocabulary without refitting.

    The Pipeline calls fit_transform() on train and transform() on test,
    so the vocabulary is never contaminated by test data.

    Input : pd.DataFrame with a 'comment_text' column.
    Output: scipy sparse matrix of shape (n, word_max_features + char_max_features).
    """

    def __init__(self, word_max_features=20_000, char_max_features=30_000):
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features

    def fit(self, X, y=None):
        texts = X[TEXT_COL].fillna("").astype(str).tolist()
        self.word_vec_ = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            stop_words="english", max_features=self.word_max_features,
        )
        self.char_vec_ = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5),
            max_features=self.char_max_features,
        )
        self.word_vec_.fit(texts)
        self.char_vec_.fit(texts)
        return self

    def transform(self, X):
        texts = X[TEXT_COL].fillna("").astype(str).tolist()
        return hstack([
            self.word_vec_.transform(texts),
            self.char_vec_.transform(texts),
        ])


class BertTransformer(BaseEstimator, TransformerMixin):
    """
    Mean-pooled BERT embeddings (bert-base-uncased).
    Stateless — pretrained weights are fixed, not fine-tuned here.

    Input : pd.DataFrame with a 'comment_text' column.
    Output: np.ndarray of shape (n, 768).
    """

    def __init__(self, model_name="bert-base-uncased", batch_size=32, max_length=128):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

    def fit(self, X, y=None):
        return self  # pretrained weights — nothing to fit

    def transform(self, X):
        try:
            import torch
            from transformers import BertModel, BertTokenizer
        except ImportError:
            raise ImportError("Run: pip install transformers torch")

        device    = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        model     = BertModel.from_pretrained(self.model_name).to(device)
        model.eval()

        texts   = X[TEXT_COL].fillna("").astype(str).tolist()
        all_emb = []
        for i in range(0, len(texts), self.batch_size):
            batch  = texts[i : i + self.batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_emb.append(emb.cpu().numpy())
        return np.vstack(all_emb)
