"""Fixtures for service layer tests."""
import pytest
import numpy as np


# ===== Sample Data Fixtures =====

@pytest.fixture
def sample_toxic_words():
    """Sample toxic words with counts for sorting tests."""
    return [
        ("toxic", 150),
        ("hate", 120),
        ("abuse", 100),
        ("spam", 80),
        ("rude", 60),
        ("bad", 50),
    ]




# ===== File/Path Fixtures =====

@pytest.fixture
def temp_npy_file(tmp_path, sample_toxic_words):
    """Create a temporary .npy file with toxic words data."""
    npy_path = tmp_path / "toxic_words.npy"
    data = np.array(sample_toxic_words, dtype=object)
    np.save(npy_path, data)
    return npy_path


# ===== Parametrize Fixtures =====

@pytest.fixture(
    params=[
        "https://en.wikipedia.org/wiki/Test_Article",
        "https://simple.wikipedia.org/wiki/Simple_Test",
    ]
)
def valid_wikipedia_urls(request):
    """Valid Wikipedia URLs for parametrized tests."""
    return request.param


@pytest.fixture(
    params=[
        "https://example.com/wiki/Article",
        "https://en.wikipedia.org/Article",
        "ftp://en.wikipedia.org/wiki/Article",
        "",
        None,
    ]
)
def invalid_wikipedia_urls(request):
    """Invalid Wikipedia URLs for parametrized tests."""
    return request.param


@pytest.fixture(
    params=[
        (0.3, 0.1),  # Below both thresholds
        (0.5, 0.7),  # Between thresholds
        (0.9, 0.95),  # Above both
        (0.75, 0.75),  # On both thresholds
        (0.0, 1.0),  # Boundary values
    ]
)
def threshold_pairs(request):
    """Parametrized threshold pairs (auto, manual) for decision logic testing."""
    return request.param
