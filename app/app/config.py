import os

class Config:
    """Base configuration."""
    #SQL_URI = os.environ.get("SQL_URI", "sqlite:///jigsaw_dataset.db")
    #ML_API_URL = os.environ.get("ML_API_URL", "http://some-ml-service/predict")
    SECRET_KEY = os.environ.get("SECRET_KEY", "jigsaw_secret_key")
    DEBUG = False
    TESTING = False

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
    #SQL_URI = os.environ.get("TEST_SQL_URI", "sqlite:///test_jigsaw_dataset.db")

class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = os.environ.get("SECRET_KEY")  # Must be set in environment