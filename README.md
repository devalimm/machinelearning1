# Doğrusal ve Lojistik Regresyon ile İki Sınıflı Sınıflandırma

Bu projede, Kaggle'dan alınan **Social Network Ads** veri kümesi (`n = 400`, iki sınıflı, doğrusal olmayan) üzerinde **Doğrusal Regresyon** ve **Lojistik Regresyon** algoritmaları hazır makine öğrenimi kütüphanesi kullanılmadan, yalnızca **NumPy** ile sıfırdan kodlanarak uygulanmış ve aynı test kümesi üzerinde karşılaştırılmıştır.

## Proje Dosya Yapısı

- `Social_Network_Ads.csv` — 400 örneklik iki sınıflı veri kümesi (`Age`, `EstimatedSalary`, `Purchased`)
- `preprocessing.py` — Veri inceleme, tabakalı (stratified) %80/%20 eğitim-test ayrımı ve Z-Score standartlaştırma
- `models.py` — Sıfırdan kodlanmış `LinearRegressionClassifier` ve `LogisticRegressionClassifier` (Gradient Descent)
- `evaluation.py` — Karışıklık matrisi, Accuracy, Precision, Recall, Specificity, F1-Skoru, ROC-AUC ve görselleştirmeler
- `main.py` — Tüm akışı çalıştıran, metrikleri raporlayan ve `results.json` ile `figures/` çıktılarını üreten ana betik
- `generate_report.py` — Akademik Türkçe PDF raporunu (`Proje_Raporu.pdf`) oluşturan betik
- `Proje_Raporu.pdf` — Veri analizi, yöntemler, grafikler ve karşılaştırmalı değerlendirmeyi içeren proje raporu

## Kurulum ve Çalıştırma

```bash
pip install numpy pandas matplotlib reportlab
python main.py
python generate_report.py
```

## Test Kümesi Sonuç Özeti (n = 80)

| Metrik | Doğrusal Regresyon | Lojistik Regresyon |
| :--- | :---: | :---: |
| Test Doğruluğu (Accuracy) | %80.00 | **%81.25** |
| Kesinlik (Precision) | 0.8095 | **0.8182** |
| Duyarlılık (Recall) | 0.5862 | **0.6207** |
| Özgüllük (Specificity) | 0.9216 | 0.9216 |
| F1-Skoru | 0.6800 | **0.7059** |
| ROC-AUC | 0.8932 | 0.8911 |
| Test Çıktı Aralığı | `[-0.323, 1.044]` | `[0.001, 0.991]` |
