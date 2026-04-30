import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

class DataPipeline:
    def __init__(
        self,
        data_path="data/processed/train_data.pkl",
        test_data_path="data/processed/test_data.pkl",
        processed_path="data/processed/test_train_data.pkl",
        text_column="comment_text",
        label_columns=None
    ):

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.train_path = data_path
        self.test_path = test_data_path
        self.processed_path = processed_path
        self.text_column = text_column

        self.label_columns = label_columns or [
            "toxic", "severe_toxic", "obscene",
            "threat", "insult", "identity_hate"
        ]

    def _load(self):
        with open(self.processed_path, "rb") as f:
            data = pickle.load(f)

        self.X_train = data["X_train"]
        self.X_test = data["X_test"]
        self.y_train = data["y_train"]
        self.y_test = data["y_test"]

    def _load_raw(self, data_path='data/raw/jigsaw-dataset/train.csv'):
        df = pd.read_csv(data_path)

        null_counts = df[[self.text_column] + self.label_columns].isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls > 0:
            print("Null values found:")
            print(null_counts[null_counts > 0])
            # drop rows with nulls in required columns
            df = df.dropna(subset=[self.text_column] + self.label_columns)

        return df

    def _save(self):
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)

        data = {
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test
        }

        with open(self.processed_path, "wb") as f:
            pickle.dump(data, f)

    def _split(self, df, split_size=0.2):
        #X = df[self.text_column].values
        #y = df[self.label_columns].values
        X = df[self.text_column]
        y = df[self.label_columns]

        return train_test_split(
            X,
            y,
            test_size=split_size,
            random_state=42,
            stratify=y
        )

    def get_data(self, force_rebuild=False):
        if os.path.exists(self.processed_path) and not force_rebuild:
            print("Loading already processed data...")
            self._load()
        #else:
        #    print("Processing raw data...")
        #    df = self._load_raw()
            #self.X_train, self.X_test, self.y_train, self.y_test = self._split(df)
        #    self._save()
            #print(self.X_train.shape, self.y_train.shape)
        #    print("Processed data saved!")

        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_infer_data(self, infer_path=None):
        """
        Returns inference features (text column only).
        If labels exist in file, they are ignored.
        """
        path = infer_path or self.test_data_path
        
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.X_test = data["X_test"]
        self.y_test = data["y_test"]

        if self.text_column not in self.X_test.columns:
            raise KeyError(f"Missing required text column: {self.text_column}")

        X = self.X_test[self.text_column].fillna("").astype(str)
        return X, self.y_test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    dp = DataPipeline(data_path=args.data_path)
    dp.get_data(force_rebuild=args.force_rebuild)

    if args.force_rebuild:
        dp.get_data(force_rebuild=True)
    else:
        dp.get_data()