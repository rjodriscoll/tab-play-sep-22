import pandas as pd
from typing import List, Tuple
from constants import X_LABELS, Y_LABEL
from sklearn.preprocessing import LabelEncoder


class PreProcess:
    def __init__(self, data: pd.DataFrame, label_encoder_cols: List[str]) -> None:
        self.label_encoder_cols = label_encoder_cols
        self.data = data
        self.test_cols = X_LABELS
        self.train_cols = self.test_cols + [Y_LABEL]
        self._run()

    def _set_index(self) -> pd.DataFrame:
        self.data.set_index("row_id", inplace=True)

    def _df_dt_features(self) -> pd.DataFrame:
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.data["year"] = self.data["date"].dt.year
        self.data["month"] = self.data["date"].dt.month
        self.data["day_of_week"] = self.data["date"].dt.day_of_week
        self.data["is_month_end"] = self.data["date"].dt.is_month_end
        self.data["is_year_end"] = self.data["date"].dt.is_year_end

    def _encoder(self):
        for col in self.label_encoder_cols:
            self.data[col] = LabelEncoder().fit_transform(self.data[col])

    @property
    def train_test(self):
        return self.train, self.test

    def _train_test_set(self):
        self.train = self.data[self.data["is_training"] == True].copy()
        self.test = self.data[self.data["is_training"] == False].copy()
        self.train = self.train[self.train_cols]
        self.test = self.test[self.test_cols]

    def _run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._set_index()
        self._df_dt_features()
        self._encoder()
        self._train_test_set()

    @property
    def train_test(self):
        return self.train, self.test
