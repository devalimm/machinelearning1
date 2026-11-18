"""
models.py
---------
Doğrusal Regresyon (Linear Regression) ve Lojistik Regresyon (Logistic Regression)
sınıflandırma algoritmalarının NumPy matris işlemleri ile sıfırdan uygulanması.
"""

import numpy as np


class LinearRegressionClassifier:
    """
    Doğrusal Regresyon tabanlı iki sınıflı (binary) sınıflandırıcı.

    Matematiksel Model:
        y_hat = X @ w + b

    Maliyet Fonksiyonu (Ortalama Kare Hata - MSE):
        J(w, b) = (1 / (2 * m)) * sum((y_hat - y)^2)

    Gradyanlar (Kısmi Türevler):
        dw = (1 / m) * X^T @ (y_hat - y)
        db = (1 / m) * sum(y_hat - y)

    Sınıflandırma Kuralı:
        y_hat >= threshold (varsayılan 0.5) -> 1, aksi halde 0
    """

    def __init__(self, learning_rate: float = 0.05, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Gradient Descent (Gradyan İnişi) ile ağırlıkları (w) ve sapmayı (b) öğrenir."""
        m, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            # 1. İleri Yayılım (Tahmin): y_hat = Xw + b
            y_pred = np.dot(X, self.weights) + self.bias

            # 2. Hata ve Maliyet (MSE) Hesabı
            error = y_pred - y
            loss = (1.0 / (2.0 * m)) * np.sum(error ** 2)
            self.loss_history.append(float(loss))

            # 3. Gradyanların (Türevlerin) Hesaplanması
            dw = (1.0 / m) * np.dot(X.T, error)
            db = (1.0 / m) * np.sum(error)

            # 4. Parametre Güncelleme
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Ham doğrusal tahmin skorlarını döndürür (0-1 aralığı dışına çıkabilir)."""
        return np.dot(X, self.weights) + self.bias

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Eşik değerine (0.5) göre 0 veya 1 sınıf tahmini döndürür."""
        scores = self.predict_scores(X)
        return (scores >= threshold).astype(int)


class LogisticRegressionClassifier:
    """
    Lojistik Regresyon tabanlı iki sınıflı (binary) sınıflandırıcı.

    Matematiksel Model:
        z = X @ w + b
        y_hat = sigmoid(z) = 1 / (1 + e^(-z))   [0 ile 1 arasında olasılık üretir]

    Maliyet Fonksiyonu (Binary Cross-Entropy / Log-Loss):
        J(w, b) = -(1 / m) * sum( y * log(y_hat) + (1 - y) * log(1 - y_hat) )

    Gradyanlar (Kısmi Türevler):
        dw = (1 / m) * X^T @ (y_hat - y)
        db = (1 / m) * sum(y_hat - y)

    Sınıflandırma Kuralı:
        P(y=1 | x) = y_hat >= threshold (varsayılan 0.5) -> 1, aksi halde 0
    """

    def __init__(self, learning_rate: float = 0.05, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Sayısal taşmayı (overflow) önleyerek Sigmoid fonksiyonunu hesaplar."""
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Gradient Descent (Gradyan İnişi) ile ağırlıkları (w) ve sapmayı (b) öğrenir."""
        m, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        self.loss_history = []
        eps = 1e-15

        for _ in range(self.n_iterations):
            # 1. Doğrusal Birleşim ve Sigmoid Aktivasyonu
            z = np.dot(X, self.weights) + self.bias
            y_prob = self._sigmoid(z)

            # 2. Maliyet (Binary Cross-Entropy / Log-Loss) Hesabı
            y_prob_safe = np.clip(y_prob, eps, 1.0 - eps)
            loss = -(1.0 / m) * np.sum(
                y * np.log(y_prob_safe) + (1.0 - y) * np.log(1.0 - y_prob_safe)
            )
            self.loss_history.append(float(loss))

            # 3. Gradyanların (Türevlerin) Hesaplanması
            error = y_prob - y
            dw = (1.0 / m) * np.dot(X.T, error)
            db = (1.0 / m) * np.sum(error)

            # 4. Parametre Güncelleme
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Sınıf 1 olasılık değerlerini [0, 1] aralığında döndürür."""
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Eşik değerine (0.5) göre 0 veya 1 sınıf tahmini döndürür."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
