"""
main.py
-------
Ana çalıştırıcı betik.
1. Social_Network_Ads.csv veri kümesini yükler ve inceler.
2. Veriyi %80 Eğitim - %20 Test olarak ayırır ve Z-Score standartlaştırma uygular.
3. Sıfırdan yazılmış Doğrusal Regresyon ve Lojistik Regresyon modellerini eğitir.
4. Aynı test kümesi üzerinde iki modeli değerlendirir ve karşılaştırır.
5. Tüm grafikleri figures/ klasörüne, sayısal sonuçları results.json dosyasına kaydeder.
"""

import json
import os
import numpy as np

from preprocessing import load_dataset, stratified_train_test_split, StandardScaler
from models import LinearRegressionClassifier, LogisticRegressionClassifier
from evaluation import (
    evaluate_model,
    plot_dataset_distribution,
    plot_loss_curves,
    plot_decision_boundaries,
    plot_confusion_matrices,
    plot_performance_comparison,
)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "Social_Network_Ads.csv")
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 65)
    print("1. VERİ KÜMESİ YÜKLEME VE ÖZET BİLGİLER")
    print("=" * 65)
    df, X, y, data_info = load_dataset(csv_path)
    print(f"Toplam örnek sayısı : {data_info['n_samples']}")
    print(f"Özellik sayısı      : {data_info['n_features']} (Age, EstimatedSalary)")
    print(f"Eksik değer sayısı  : {sum(data_info['missing_values'].values())}")
    print(f"Mükerrer satır      : {data_info['duplicate_rows']}")
    print(f"Sınıf dağılımı      : 0 (Satın Almadı) = {data_info['class_counts'][0]}, "
          f"1 (Satın Aldı) = {data_info['class_counts'][1]}")
    print(f"Age aralığı         : {data_info['age_stats']['min']:.0f} - {data_info['age_stats']['max']:.0f} "
          f"(Ort: {data_info['age_stats']['mean']:.2f}, Std: {data_info['age_stats']['std']:.2f})")
    print(f"Maaş aralığı        : {data_info['salary_stats']['min']:.0f} - {data_info['salary_stats']['max']:.0f} "
          f"(Ort: {data_info['salary_stats']['mean']:.2f}, Std: {data_info['salary_stats']['std']:.2f})")

    print("\n" + "=" * 65)
    print("2. ÖN İŞLEME (STRATIFIED %80/%20 AYRIM VE Z-SCORE ÖLÇEKLENDİRME)")
    print("=" * 65)
    X_train_raw, X_test_raw, y_train, y_test = stratified_train_test_split(
        X, y, test_ratio=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"Eğitim kümesi boyutu: {len(y_train)} (Sınıf 0: {int(np.sum(y_train==0))}, Sınıf 1: {int(np.sum(y_train==1))})")
    print(f"Test kümesi boyutu  : {len(y_test)} (Sınıf 0: {int(np.sum(y_test==0))}, Sınıf 1: {int(np.sum(y_test==1))})")
    print(f"Eğitim ortalamaları : Age={scaler.mean_[0]:.2f}, EstimatedSalary={scaler.mean_[1]:.2f}")
    print(f"Eğitim std sapmaları: Age={scaler.std_[0]:.2f}, EstimatedSalary={scaler.std_[1]:.2f}")

    print("\n" + "=" * 65)
    print("3. MODELLERİN EĞİTİMİ (GRADIENT DESCENT - SIFIRDAN KODLAMA)")
    print("=" * 65)
    learning_rate = 0.05
    n_iterations = 1500

    lin_model = LinearRegressionClassifier(learning_rate=learning_rate, n_iterations=n_iterations)
    lin_model.fit(X_train, y_train)

    log_model = LogisticRegressionClassifier(learning_rate=learning_rate, n_iterations=n_iterations)
    log_model.fit(X_train, y_train)

    print(f"[Doğrusal Regresyon]  w = [{lin_model.weights[0]:.4f}, {lin_model.weights[1]:.4f}], "
          f"b = {lin_model.bias:.4f}, Son MSE = {lin_model.loss_history[-1]:.4f}")
    print(f"[Lojistik Regresyon]  w = [{log_model.weights[0]:.4f}, {log_model.weights[1]:.4f}], "
          f"b = {log_model.bias:.4f}, Son Log-Loss = {log_model.loss_history[-1]:.4f}")

    print("\n" + "=" * 65)
    print("4. TEST KÜMESİ PERFORMANS KARŞILAŞTIRMASI")
    print("=" * 65)

    lin_train_metrics = evaluate_model(y_train, lin_model.predict(X_train), lin_model.predict_scores(X_train))
    log_train_metrics = evaluate_model(y_train, log_model.predict(X_train), log_model.predict_proba(X_train))

    lin_test_scores = lin_model.predict_scores(X_test)
    lin_test_preds = lin_model.predict(X_test)
    lin_test_metrics = evaluate_model(y_test, lin_test_preds, lin_test_scores)

    log_test_probs = log_model.predict_proba(X_test)
    log_test_preds = log_model.predict(X_test)
    log_test_metrics = evaluate_model(y_test, log_test_preds, log_test_probs)

    header = f"{'Metrik':<22} {'Doğrusal Reg.':>15} {'Lojistik Reg.':>15}"
    print(header)
    print("-" * len(header))
    rows = [
        ("Eğitim Doğruluğu", lin_train_metrics["accuracy"], log_train_metrics["accuracy"]),
        ("Test Doğruluğu (Acc)", lin_test_metrics["accuracy"], log_test_metrics["accuracy"]),
        ("Precision (Kesinlik)", lin_test_metrics["precision"], log_test_metrics["precision"]),
        ("Recall (Duyarlılık)", lin_test_metrics["recall"], log_test_metrics["recall"]),
        ("Specificity (Özgüllük)", lin_test_metrics["specificity"], log_test_metrics["specificity"]),
        ("F1-Skoru", lin_test_metrics["f1_score"], log_test_metrics["f1_score"]),
        ("ROC-AUC", lin_test_metrics["roc_auc"], log_test_metrics["roc_auc"]),
    ]
    for name, v1, v2 in rows:
        print(f"{name:<22} {v1:>15.4f} {v2:>15.4f}")

    print("\nTest Karışıklık Matrisi (Doğrusal):", lin_test_metrics["confusion_matrix"])
    print("Test Karışıklık Matrisi (Lojistik):", log_test_metrics["confusion_matrix"])
    print(f"\nDoğrusal Regresyon ham tahmin aralığı (Test): [{lin_test_metrics['score_min']:.3f}, {lin_test_metrics['score_max']:.3f}]")
    print(f"Lojistik Regresyon olasılık aralığı   (Test): [{log_test_metrics['score_min']:.3f}, {log_test_metrics['score_max']:.3f}]")

    print("\n" + "=" * 65)
    print("5. GRAFİKLER OLUŞTURULUYOR...")
    print("=" * 65)
    plot_dataset_distribution(X, y, os.path.join(figures_dir, "1_veri_dagilimi.png"))
    plot_loss_curves(lin_model.loss_history, log_model.loss_history, os.path.join(figures_dir, "2_maliyet_egrileri.png"))
    plot_decision_boundaries(X_test_raw, y_test, scaler, lin_model, log_model, os.path.join(figures_dir, "3_karar_sinirlari.png"))
    plot_confusion_matrices(
        lin_test_metrics["confusion_matrix"],
        log_test_metrics["confusion_matrix"],
        os.path.join(figures_dir, "4_karisiklik_matrisleri.png"),
    )
    plot_performance_comparison(lin_test_metrics, log_test_metrics, os.path.join(figures_dir, "5_performans_karsilastirma.png"))
    print("5 grafik figures/ klasörüne kaydedildi.")

    results = {
        "dataset": data_info,
        "split": {
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "train_class_0": int(np.sum(y_train == 0)),
            "train_class_1": int(np.sum(y_train == 1)),
            "test_class_0": int(np.sum(y_test == 0)),
            "test_class_1": int(np.sum(y_test == 1)),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_std": scaler.std_.tolist(),
        },
        "hyperparameters": {
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "threshold": 0.5,
            "random_state": 42,
        },
        "linear_regression": {
            "weights": lin_model.weights.tolist(),
            "bias": float(lin_model.bias),
            "initial_loss": float(lin_model.loss_history[0]),
            "final_loss": float(lin_model.loss_history[-1]),
            "train_accuracy": lin_train_metrics["accuracy"],
            "test_metrics": {k: v for k, v in lin_test_metrics.items() if k != "roc_curve"},
        },
        "logistic_regression": {
            "weights": log_model.weights.tolist(),
            "bias": float(log_model.bias),
            "initial_loss": float(log_model.loss_history[0]),
            "final_loss": float(log_model.loss_history[-1]),
            "train_accuracy": log_train_metrics["accuracy"],
            "test_metrics": {k: v for k, v in log_test_metrics.items() if k != "roc_curve"},
        },
    }

    with open(os.path.join(base_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Sonuçlar results.json dosyasına kaydedildi.")


if __name__ == "__main__":
    main()
