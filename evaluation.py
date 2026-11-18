"""
evaluation.py
-------------
Sınıflandırma performans metriklerinin (Accuracy, Precision, Recall,
Specificity, F1-Skoru, ROC-AUC, Karışıklık Matrisi) hesaplanması ve
karşılaştırmalı grafiklerin üretilmesi.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Karışıklık matrisi elemanlarını (TN, FP, FN, TP) hesaplar."""
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray):
    """Farklı eşik değerleri için ROC eğrisini (FPR, TPR) ve AUC alanını hesaplar."""
    y_true = y_true.astype(int)
    thresholds = np.concatenate(([np.inf], np.sort(np.unique(y_scores))[::-1], [-np.inf]))
    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)

    tprs = []
    fprs = []
    for thr in thresholds:
        preds = (y_scores >= thr).astype(int)
        tp = np.sum((y_true == 1) & (preds == 1))
        fp = np.sum((y_true == 0) & (preds == 1))
        tprs.append(tp / total_pos if total_pos > 0 else 0.0)
        fprs.append(fp / total_neg if total_neg > 0 else 0.0)

    fprs = np.array(fprs)
    tprs = np.array(tprs)
    # Trapez kuralı ile eğri altındaki alan (AUC)
    auc = float(np.trapezoid(tprs, fprs))
    return fprs, tprs, auc


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> dict:
    """Tüm temel sınıflandırma metriklerini hesaplar ve sözlük olarak döndürür."""
    cm = compute_confusion_matrix(y_true, y_pred)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fprs, tprs, roc_auc = compute_roc_auc(y_true, y_scores)

    return {
        "confusion_matrix": cm,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "roc_curve": {"fpr": fprs.tolist(), "tpr": tprs.tolist()},
        "score_min": float(np.min(y_scores)),
        "score_max": float(np.max(y_scores)),
    }


def _configure_plot_style():
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#FAFAFA"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["font.family"] = "DejaVu Sans"


def plot_dataset_distribution(X: np.ndarray, y: np.ndarray, save_path: str):
    """1. Grafik: Veri kümesinin 2D dağılımı (Yaş vs Tahmini Maaş)."""
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    cls0 = y == 0
    cls1 = y == 1
    ax.scatter(
        X[cls0, 0], X[cls0, 1] / 1000.0,
        c="#2E86AB", alpha=0.75, edgecolors="white", linewidth=0.6, s=55,
        label="Satın Almadı (0)"
    )
    ax.scatter(
        X[cls1, 0], X[cls1, 1] / 1000.0,
        c="#E63946", alpha=0.80, edgecolors="white", linewidth=0.6, s=55,
        marker="^", label="Satın Aldı (1)"
    )

    ax.set_title("Social Network Ads Veri Kümesi Dağılımı (n = 400)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Yaş (Age)", fontsize=10.5)
    ax.set_ylabel("Tahmini Maaş (Bin $)", fontsize=10.5)
    ax.legend(frameon=True, facecolor="white", loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_loss_curves(lin_losses: list, log_losses: list, save_path: str):
    """2. Grafik: Eğitim sürecinde maliyet fonksiyonu (Loss) değişimleri."""
    _configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(lin_losses, color="#D97706", linewidth=2.2)
    axes[0].set_title("Doğrusal Regresyon - MSE Maliyet Eğrisi", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("İterasyon")
    axes[0].set_ylabel("MSE Maliyeti")

    axes[1].plot(log_losses, color="#2563EB", linewidth=2.2)
    axes[1].set_title("Lojistik Regresyon - Log-Loss Maliyet Eğrisi", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("İterasyon")
    axes[1].set_ylabel("Binary Cross-Entropy Maliyeti")

    fig.suptitle("Gradient Descent Eğitim Yakınsama Eğrileri", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_decision_boundaries(X_test_raw, y_test, scaler, lin_model, log_model, save_path: str):
    """3. Grafik: İki algoritmanın Test Verisi üzerindeki karar sınırları."""
    _configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    age_min, age_max = X_test_raw[:, 0].min() - 3, X_test_raw[:, 0].max() + 3
    sal_min, sal_max = X_test_raw[:, 1].min() - 8000, X_test_raw[:, 1].max() + 8000

    xx, yy = np.meshgrid(
        np.linspace(age_min, age_max, 250),
        np.linspace(sal_min, sal_max, 250)
    )
    grid_raw = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid_raw)

    configs = [
        (axes[0], lin_model, "Doğrusal Regresyon Karar Sınırı (Test)"),
        (axes[1], log_model, "Lojistik Regresyon Karar Sınırı (Test)"),
    ]

    for ax, model, title in configs:
        Z = model.predict(grid_scaled).reshape(xx.shape)
        ax.contourf(xx, yy / 1000.0, Z, alpha=0.22, levels=[-0.5, 0.5, 1.5], colors=["#2E86AB", "#E63946"])
        ax.contour(xx, yy / 1000.0, Z, levels=[0.5], colors=["#111827"], linewidths=1.8, linestyles="--")

        cls0 = y_test == 0
        cls1 = y_test == 1
        ax.scatter(
            X_test_raw[cls0, 0], X_test_raw[cls0, 1] / 1000.0,
            c="#2E86AB", edgecolors="white", linewidth=0.7, s=60, label="Gerçek: 0 (Almadı)"
        )
        ax.scatter(
            X_test_raw[cls1, 0], X_test_raw[cls1, 1] / 1000.0,
            c="#E63946", edgecolors="white", linewidth=0.7, s=60, marker="^", label="Gerçek: 1 (Aldı)"
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Yaş (Age)")
        ax.set_ylabel("Tahmini Maaş (Bin $)")
        ax.legend(frameon=True, facecolor="white", loc="upper left", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrices(lin_cm: dict, log_cm: dict, save_path: str):
    """4. Grafik: İki algoritmanın Test Karışıklık Matrisleri yan yana."""
    _configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))

    items = [
        (axes[0], lin_cm, "Doğrusal Regresyon - Karışıklık Matrisi", "Oranges"),
        (axes[1], log_cm, "Lojistik Regresyon - Karışıklık Matrisi", "Blues"),
    ]

    for ax, cm, title, cmap in items:
        mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=max(mat.max(), 1) * 1.15)
        ax.grid(False)

        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                text_color = "white" if val > mat.max() * 0.55 else "#111827"
                ax.text(
                    j, i, f"{labels[i][j]}\n{val}",
                    ha="center", va="center", color=text_color, fontsize=12, fontweight="bold"
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Tahmin: 0", "Tahmin: 1"])
        ax.set_yticklabels(["Gerçek: 0", "Gerçek: 1"])
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_performance_comparison(lin_metrics: dict, log_metrics: dict, save_path: str):
    """5. Grafik: ROC Eğrileri ve Metrik Karşılaştırma Çubuk Grafiği."""
    _configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Sol panel: ROC Eğrisi
    ax_roc = axes[0]
    ax_roc.plot(
        lin_metrics["roc_curve"]["fpr"], lin_metrics["roc_curve"]["tpr"],
        color="#D97706", linewidth=2.2,
        label=f"Doğrusal Regresyon (AUC = {lin_metrics['roc_auc']:.4f})"
    )
    ax_roc.plot(
        log_metrics["roc_curve"]["fpr"], log_metrics["roc_curve"]["tpr"],
        color="#2563EB", linewidth=2.2, linestyle="-.",
        label=f"Lojistik Regresyon (AUC = {log_metrics['roc_auc']:.4f})"
    )
    ax_roc.plot([0, 1], [0, 1], color="#6B7280", linestyle="--", linewidth=1.2, label="Rastgele Tahmin (AUC = 0.50)")
    ax_roc.set_title("ROC Eğrisi Karşılaştırması (Test Kümesi)", fontsize=11, fontweight="bold")
    ax_roc.set_xlabel("Yanlış Pozitif Oranı (FPR)")
    ax_roc.set_ylabel("Doğru Pozitif Oranı (TPR / Recall)")
    ax_roc.legend(loc="lower right", frameon=True, facecolor="white", fontsize=8.5)

    # Sağ panel: Metrik Çubuk Grafiği
    ax_bar = axes[1]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Skoru", "ROC-AUC"]
    lin_vals = [
        lin_metrics["accuracy"], lin_metrics["precision"], lin_metrics["recall"],
        lin_metrics["f1_score"], lin_metrics["roc_auc"]
    ]
    log_vals = [
        log_metrics["accuracy"], log_metrics["precision"], log_metrics["recall"],
        log_metrics["f1_score"], log_metrics["roc_auc"]
    ]

    x = np.arange(len(metric_names))
    width = 0.35
    bars1 = ax_bar.bar(x - width / 2, lin_vals, width, label="Doğrusal Regresyon", color="#F59E0B")
    bars2 = ax_bar.bar(x + width / 2, log_vals, width, label="Lojistik Regresyon", color="#2563EB")

    for bar in bars1:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax_bar.set_ylim(0, 1.18)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names)
    ax_bar.set_ylabel("Skor")
    ax_bar.set_title("Test Kümesi Performans Metrikleri", fontsize=11, fontweight="bold")
    ax_bar.legend(loc="upper center", ncol=2, frameon=True, facecolor="white", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
