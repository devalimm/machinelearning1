"""
preprocessing.py
----------------
Veri kümesini yükleme, inceleme, eğitim/test kümelerine ayırma ve
Z-Score standartlaştırma işlemlerini içerir.
"""

import numpy as np
import pandas as pd


def load_dataset(csv_path: str):
    """CSV dosyasını okur ve temel veri bilgilerini döndürür."""
    df = pd.read_csv(csv_path)

    info = {
        "n_samples": int(len(df)),
        "n_features": int(df.shape[1] - 1),
        "columns": list(df.columns),
        "missing_values": {col: int(val) for col, val in df.isnull().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {int(k): int(v) for k, v in df["Purchased"].value_counts().sort_index().items()},
        "age_stats": {
            "min": float(df["Age"].min()),
            "max": float(df["Age"].max()),
            "mean": float(df["Age"].mean()),
            "std": float(df["Age"].std()),
        },
        "salary_stats": {
            "min": float(df["EstimatedSalary"].min()),
            "max": float(df["EstimatedSalary"].max()),
            "mean": float(df["EstimatedSalary"].mean()),
            "std": float(df["EstimatedSalary"].std()),
        },
        "correlations": {
            "Age_Purchased": float(df["Age"].corr(df["Purchased"])),
            "EstimatedSalary_Purchased": float(df["EstimatedSalary"].corr(df["Purchased"])),
            "Age_EstimatedSalary": float(df["Age"].corr(df["EstimatedSalary"])),
        },
    }

    X = df[["Age", "EstimatedSalary"]].values.astype(np.float64)
    y = df["Purchased"].values.astype(np.float64)
    return df, X, y, info


def stratified_train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, random_state: int = 42):
    """
    Sınıf oranlarını (0 ve 1 dağılımını) koruyarak veriyi eğitim ve test olarak ayırır.
    Örn: %80 Eğitim, %20 Test.
    """
    rng = np.random.default_rng(random_state)
    train_indices = []
    test_indices = []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_test = int(round(len(cls_idx) * test_ratio))
        test_indices.extend(cls_idx[:n_test])
        train_indices.extend(cls_idx[n_test:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


class StandardScaler:
    """
    Z-Score Standartlaştırma: z = (x - ortalama) / standart_sapma

    Neden gereklidir?
    Yaş (18-60) ile Maaş (15.000-150.000) çok farklı ölçeklerdedir.
    Ölçeklendirme yapılmazsa Gradient Descent (Gradyan İnişi) maaş ekseninde
    çok büyük adımlar atarak yakınsayamaz.

    Önemli kural: Ortalama ve standart sapma SADECE eğitim kümesinden hesaplanır,
    test kümesi aynı değerlerle dönüştürülür (veri sızıntısı / data leakage önlenir).
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
