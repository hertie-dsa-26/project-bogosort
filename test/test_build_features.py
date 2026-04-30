"""
Tests for build_features.py — unit and integration level.

Run with:
    uv run pytest test/test_build_features.py -v
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis.features.build_features import (
    DenseFeatureTransformer,
    TfidfTransformer,
    _avg_sentence_length,
    _consecutive_punct_count,
    _elongated_token_count,
    _extract_identity,
    _extract_second_person,
    _extract_sentiment,
    _ip_count,
    _negation_count,
    _normalize_leetspeak,
    _obfuscated_profanity_count,
    _profanity_count,
    _sentence_count,
    _slang_count,
    _unique_word_ratio,
    _uppercase_ratio,
    _url_count,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_df(*texts):
    """Build a minimal DataFrame that DenseFeatureTransformer expects."""
    return pd.DataFrame({"comment_text": list(texts)})


# ===========================================================================
# Unit tests — private feature functions
# ===========================================================================

class TestNormalizeLeetspeak:
    def test_substitutes_symbols(self):
        assert _normalize_leetspeak("$h1t") == "shit"

    def test_substitutes_at_sign(self):
        assert _normalize_leetspeak("@$$") == "ass"

    def test_plain_word_unchanged(self):
        assert _normalize_leetspeak("hello") == "hello"

    def test_strips_remaining_non_alpha(self):
        # Characters not in the map are stripped
        assert _normalize_leetspeak("f**k") == "fk"


class TestExtractSentiment:
    def test_returns_all_keys(self):
        result = _extract_sentiment("hello")
        expected_keys = {
            "vader_compound", "vader_neg", "vader_pos",
            "vader_is_negative", "vader_intensity", "vader_pos_minus_neg",
        }
        assert set(result.keys()) == expected_keys

    def test_negative_text_flags_correctly(self):
        result = _extract_sentiment("I hate you, you worthless piece of garbage!")
        assert result["vader_is_negative"] == 1
        assert result["vader_compound"] < -0.05

    def test_positive_text_not_flagged(self):
        result = _extract_sentiment("I love this, it is wonderful and amazing!")
        assert result["vader_is_negative"] == 0
        assert result["vader_compound"] > 0.05

    def test_intensity_is_absolute(self):
        result = _extract_sentiment("terrible awful disaster")
        assert result["vader_intensity"] >= 0

    def test_pos_minus_neg_direction(self):
        # Positive text → pos_minus_neg should be positive
        positive = _extract_sentiment("wonderful happy joyful")
        negative = _extract_sentiment("terrible awful disgusting")
        assert positive["vader_pos_minus_neg"] > negative["vader_pos_minus_neg"]

    def test_empty_string(self):
        result = _extract_sentiment("")
        assert result["vader_compound"] == 0.0


class TestExtractSecondPerson:
    def test_detects_you(self):
        result = _extract_second_person("You should leave now")
        assert result["has_second_person"] == 1
        assert result["second_person_count"] == 1

    def test_detects_your(self):
        result = _extract_second_person("Your opinion is wrong")
        assert result["second_person_count"] == 1

    def test_no_second_person(self):
        result = _extract_second_person("He went to the store")
        assert result["has_second_person"] == 0
        assert result["second_person_count"] == 0

    def test_multiple_pronouns(self):
        result = _extract_second_person("you and your friend, yourselves")
        assert result["second_person_count"] == 3

    def test_density_proportional_to_word_count(self):
        short = _extract_second_person("you")         # 1/1
        long  = _extract_second_person("you are a great person indeed")  # 1/6
        assert short["second_person_density"] > long["second_person_density"]

    def test_youtube_not_matched(self):
        # "you" inside "youtube" should not match due to word boundary
        result = _extract_second_person("check out youtube")
        assert result["second_person_count"] == 0

    def test_empty_string(self):
        result = _extract_second_person("")
        assert result["has_second_person"] == 0
        assert result["second_person_density"] == 0.0


class TestProfanityCount:
    def test_counts_profanity(self):
        assert _profanity_count("you are a fucking idiot") == 2

    def test_clean_text_is_zero(self):
        assert _profanity_count("the weather is nice today") == 0

    def test_case_insensitive(self):
        assert _profanity_count("SHIT happens") == 1

    def test_empty_string(self):
        assert _profanity_count("") == 0


class TestObfuscatedProfanityCount:
    def test_detects_leetspeak(self):
        # "sh1t" → normalises to "shit", plain "sht" not in lexicon → obfuscated
        assert _obfuscated_profanity_count("sh1t") == 1

    def test_plain_profanity_not_counted(self):
        # "shit" plain is already in the lexicon → not obfuscated
        assert _obfuscated_profanity_count("shit") == 0

    def test_dollar_sign_obfuscation(self):
        # "$hit" → plain "hit" not in lexicon, normalised "shit" is
        assert _obfuscated_profanity_count("$hit") == 1

    def test_clean_text(self):
        assert _obfuscated_profanity_count("hello world") == 0


class TestSlangCount:
    def test_counts_slang(self):
        assert _slang_count("kys you noob") == 2

    def test_clean_text_is_zero(self):
        assert _slang_count("hello world") == 0

    def test_case_insensitive(self):
        assert _slang_count("KYS") == 1

    def test_empty_string(self):
        assert _slang_count("") == 0


class TestUppercaseRatio:
    def test_all_uppercase(self):
        assert _uppercase_ratio("HELLO WORLD") == 1.0

    def test_no_uppercase(self):
        assert _uppercase_ratio("hello world") == 0.0

    def test_half_uppercase(self):
        assert _uppercase_ratio("HELLO world") == pytest.approx(0.5)

    def test_single_char_words_excluded(self):
        # Single-char words like "I" are excluded from the uppercase check
        assert _uppercase_ratio("I am here") == 0.0

    def test_empty_string(self):
        assert _uppercase_ratio("") == 0.0


class TestUniqueWordRatio:
    def test_all_unique(self):
        assert _unique_word_ratio("one two three") == pytest.approx(1.0)

    def test_all_same(self):
        assert _unique_word_ratio("the the the") == pytest.approx(1 / 3)

    def test_empty_string(self):
        assert _unique_word_ratio("") == 0.0


class TestElongatedTokenCount:
    def test_detects_elongation(self):
        assert _elongated_token_count("coooool") == 1

    def test_multiple_elongated(self):
        assert _elongated_token_count("noooo waaaay") == 2

    def test_normal_text_is_zero(self):
        assert _elongated_token_count("normal text here") == 0

    def test_two_repeats_not_counted(self):
        # Only 3+ consecutive repeats count
        assert _elongated_token_count("cool") == 0


class TestConsecutivePunctCount:
    def test_double_exclamation(self):
        assert _consecutive_punct_count("wow!!") == 1

    def test_triple_question(self):
        assert _consecutive_punct_count("really???") == 1

    def test_single_punct_not_counted(self):
        assert _consecutive_punct_count("hello.") == 0

    def test_empty_string(self):
        assert _consecutive_punct_count("") == 0


class TestUrlCount:
    def test_detects_https_url(self):
        assert _url_count("visit https://example.com today") == 1

    def test_detects_www_url(self):
        assert _url_count("go to www.example.com") == 1

    def test_no_url(self):
        assert _url_count("no links here") == 0

    def test_multiple_urls(self):
        assert _url_count("https://a.com and https://b.com") == 2


class TestIpCount:
    def test_detects_valid_ip(self):
        assert _ip_count("logged from 192.168.1.1") == 1

    def test_no_ip(self):
        assert _ip_count("no ip address here") == 0

    def test_invalid_octet_not_matched(self):
        # 999 is not a valid octet
        assert _ip_count("999.999.999.999") == 0


class TestNegationCount:
    def test_counts_not(self):
        assert _negation_count("I did not do it") == 1

    def test_counts_never(self):
        assert _negation_count("I never said that") == 1

    def test_counts_contraction(self):
        assert _negation_count("I can't and won't") == 2

    def test_clean_text_is_zero(self):
        assert _negation_count("I did it") == 0


class TestSentenceCount:
    def test_two_sentences(self):
        assert _sentence_count("Hello world. Goodbye world.") == 2

    def test_question_and_exclamation(self):
        assert _sentence_count("Really? Yes!") == 2

    def test_single_sentence_no_punct(self):
        assert _sentence_count("no punctuation here") == 1

    def test_empty_string(self):
        assert _sentence_count("") == 0


class TestAvgSentenceLength:
    def test_two_equal_sentences(self):
        # "Hello world" (2 words) + "Bye now" (2 words) → avg 2.0
        assert _avg_sentence_length("Hello world. Bye now.") == pytest.approx(2.0)

    def test_empty_string(self):
        assert _avg_sentence_length("") == 0.0


class TestExtractIdentity:
    def test_race_detected(self):
        result = _extract_identity("black people deserve respect")
        assert result["identity_race"] == 1
        assert result["identity_mention_count"] >= 1

    def test_sexuality_detected(self):
        result = _extract_identity("gay rights matter")
        assert result["identity_sexuality"] == 1

    def test_religion_detected(self):
        result = _extract_identity("the muslim community")
        assert result["identity_religion"] == 1

    def test_no_identity_mention(self):
        result = _extract_identity("the weather is nice today")
        assert result["identity_mention_count"] == 0
        assert result["identity_race"] == 0
        assert result["identity_gender"] == 0

    def test_returns_all_keys(self):
        result = _extract_identity("hello")
        expected_keys = {
            "identity_mention_count",
            "identity_race", "identity_gender", "identity_sexuality",
            "identity_religion", "identity_disability", "identity_nationality",
        }
        assert set(result.keys()) == expected_keys

    def test_mention_count_accumulates(self):
        result = _extract_identity("black gay muslim")
        assert result["identity_mention_count"] >= 3


# ===========================================================================
# Integration tests — DenseFeatureTransformer
# ===========================================================================

EXPECTED_FEATURE_COLS = {
    "vader_compound", "vader_neg", "vader_pos",
    "vader_is_negative", "vader_intensity", "vader_pos_minus_neg",
    "has_second_person", "second_person_count", "second_person_density",
    "profanity_count", "obfuscated_profanity_count",
    "slang_count",
    "char_count", "word_count", "exclamation_count", "uppercase_ratio",
    "unique_word_ratio",
    "elongated_token_count", "consecutive_punct_count",
    "url_count", "ip_count", "has_url_or_ip",
    "negation_count", "sentence_count", "avg_sentence_length",
    "identity_mention_count", "identity_race", "identity_gender",
    "identity_sexuality", "identity_religion", "identity_disability",
    "identity_nationality",
}


class TestDenseFeatureTransformer:
    def test_produces_all_feature_columns(self):
        df = _make_df("hello world", "you suck", "I hate you")
        result = DenseFeatureTransformer().fit_transform(df)
        assert EXPECTED_FEATURE_COLS.issubset(set(result.columns))

    def test_row_count_unchanged(self):
        df = _make_df("a", "b", "c", "d")
        result = DenseFeatureTransformer().fit_transform(df)
        assert len(result) == 4

    def test_original_columns_preserved(self):
        df = _make_df("hello")
        df["label"] = 1
        result = DenseFeatureTransformer().fit_transform(df)
        assert "comment_text" in result.columns
        assert "label" in result.columns

    def test_does_not_mutate_input(self):
        df = _make_df("hello world")
        original_cols = list(df.columns)
        DenseFeatureTransformer().fit_transform(df)
        assert list(df.columns) == original_cols

    def test_handles_nan(self):
        df = pd.DataFrame({"comment_text": [None, float("nan"), "hello"]})
        result = DenseFeatureTransformer().fit_transform(df)
        assert result["word_count"].notna().all()

    def test_no_nan_in_feature_columns(self):
        df = _make_df("hello", "you idiot", "I love you", "")
        result = DenseFeatureTransformer().fit_transform(df)
        feature_cols = [c for c in result.columns if c in EXPECTED_FEATURE_COLS]
        assert not result[feature_cols].isna().any().any()

    def test_fit_returns_self(self):
        transformer = DenseFeatureTransformer()
        assert transformer.fit(_make_df("hello")) is transformer

    def test_char_count_correct(self):
        df = _make_df("hello")
        result = DenseFeatureTransformer().fit_transform(df)
        assert result["char_count"].iloc[0] == 5

    def test_empty_string_row(self):
        df = _make_df("")
        result = DenseFeatureTransformer().fit_transform(df)
        assert result["word_count"].iloc[0] == 0
        assert result["vader_compound"].iloc[0] == 0.0


# ===========================================================================
# Integration tests — TfidfTransformer
# ===========================================================================

class TestTfidfTransformer:
    def _train_test_dfs(self):
        train = _make_df("the quick brown fox", "hello world", "foo bar baz")
        test  = _make_df("the lazy dog", "world hello")
        return train, test

    def test_output_is_sparse_matrix(self):
        import scipy.sparse
        train, _ = self._train_test_dfs()
        result = TfidfTransformer().fit_transform(train)
        assert scipy.sparse.issparse(result)

    def test_train_row_count(self):
        train, _ = self._train_test_dfs()
        result = TfidfTransformer().fit_transform(train)
        assert result.shape[0] == len(train)

    def test_test_has_same_columns_as_train(self):
        train, test = self._train_test_dfs()
        tfidf = TfidfTransformer()
        train_out = tfidf.fit_transform(train)
        test_out  = tfidf.transform(test)
        assert train_out.shape[1] == test_out.shape[1]

    def test_vocabulary_not_refit_on_test(self):
        # After fit(), the vocabulary is fixed. Calling transform() on test
        # should use the exact same vectorizers — we check they are the same objects.
        train, test = self._train_test_dfs()
        tfidf = TfidfTransformer()
        tfidf.fit(train)
        word_vec_before = tfidf.word_vec_
        tfidf.transform(test)
        assert tfidf.word_vec_ is word_vec_before

    def test_fit_returns_self(self):
        train, _ = self._train_test_dfs()
        tfidf = TfidfTransformer()
        assert tfidf.fit(train) is tfidf

    def test_max_features_respected(self):
        train, _ = self._train_test_dfs()
        tfidf = TfidfTransformer(word_max_features=5, char_max_features=10)
        result = tfidf.fit_transform(train)
        assert result.shape[1] <= 15  # 5 word + 10 char

    def test_handles_nan_in_text(self):
        train = pd.DataFrame({"comment_text": [None, "hello world"]})
        tfidf = TfidfTransformer()
        result = tfidf.fit_transform(train)
        assert result.shape[0] == 2
