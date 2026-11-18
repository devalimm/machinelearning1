"""
generate_report.py
------------------
results.json ve figures/ klasöründeki grafikleri kullanarak akademik formatta,
Türkçe karakter destekli Proje_Raporu.pdf dosyasını oluşturur.
"""

import json
import os
import matplotlib

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def register_fonts():
    """Türkçe karakter ve matematik sembollerini destekleyen DejaVuSans fontlarını yükler."""
    font_dir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(font_dir, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(font_dir, "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Italic", os.path.join(font_dir, "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Mono", os.path.join(font_dir, "DejaVuSansMono.ttf")))
    pdfmetrics.registerFontFamily(
        "DejaVu",
        normal="DejaVu",
        bold="DejaVu-Bold",
        italic="DejaVu-Italic",
        boldItalic="DejaVu-Bold",
    )


def add_page_number(canvas, doc):
    """Her sayfanın altına sayfa numarası ve üstüne ince başlık çizgisi ekler."""
    canvas.saveState()
    page_num = canvas.getPageNumber()
    width, height = A4

    # Üst bilgi (ilk sayfa hariç)
    if page_num > 1:
        canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
        canvas.setLineWidth(0.6)
        canvas.line(2 * cm, height - 1.6 * cm, width - 2 * cm, height - 1.6 * cm)
        canvas.setFont("DejaVu", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(2 * cm, height - 1.4 * cm, "Doğrusal ve Lojistik Regresyon ile İki Sınıflı Sınıflandırma Raporu")

    # Alt bilgi: sayfa numarası
    canvas.setFont("DejaVu", 8.5)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawCentredString(width / 2.0, 1.2 * cm, f"Sayfa {page_num}")
    canvas.restoreState()


def build_styles():
    styles = getSampleStyleSheet()
    navy = colors.HexColor("#0F172A")
    accent = colors.HexColor("#1D4ED8")
    dark_text = colors.HexColor("#1E293B")
    muted = colors.HexColor("#475569")

    custom = {
        "Title": ParagraphStyle(
            "CustomTitle",
            parent=styles["Normal"],
            fontName="DejaVu-Bold",
            fontSize=17,
            leading=22,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
        "Subtitle": ParagraphStyle(
            "CustomSubtitle",
            parent=styles["Normal"],
            fontName="DejaVu",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#DBEAFE"),
            alignment=TA_LEFT,
        ),
        "H1": ParagraphStyle(
            "CustomH1",
            parent=styles["Normal"],
            fontName="DejaVu-Bold",
            fontSize=12.5,
            leading=16,
            textColor=accent,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "H2": ParagraphStyle(
            "CustomH2",
            parent=styles["Normal"],
            fontName="DejaVu-Bold",
            fontSize=10.5,
            leading=14,
            textColor=navy,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "Body": ParagraphStyle(
            "CustomBody",
            parent=styles["Normal"],
            fontName="DejaVu",
            fontSize=9.5,
            leading=14.5,
            textColor=dark_text,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "Bullet": ParagraphStyle(
            "CustomBullet",
            parent=styles["Normal"],
            fontName="DejaVu",
            fontSize=9.3,
            leading=14,
            textColor=dark_text,
            leftIndent=14,
            firstLineIndent=-8,
            spaceAfter=3.5,
        ),
        "Formula": ParagraphStyle(
            "CustomFormula",
            parent=styles["Normal"],
            fontName="DejaVu-Mono",
            fontSize=8.8,
            leading=13.5,
            textColor=navy,
            backColor=colors.HexColor("#F1F5F9"),
            borderColor=colors.HexColor("#CBD5E1"),
            borderWidth=0.6,
            borderPadding=7,
            spaceBefore=4,
            spaceAfter=7,
        ),
        "Caption": ParagraphStyle(
            "CustomCaption",
            parent=styles["Normal"],
            fontName="DejaVu-Italic",
            fontSize=8.5,
            leading=11.5,
            textColor=muted,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=10,
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader",
            parent=styles["Normal"],
            fontName="DejaVu-Bold",
            fontSize=8.8,
            leading=11,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            parent=styles["Normal"],
            fontName="DejaVu",
            fontSize=8.8,
            leading=11.5,
            textColor=dark_text,
            alignment=TA_CENTER,
        ),
        "TableCellLeft": ParagraphStyle(
            "TableCellLeft",
            parent=styles["Normal"],
            fontName="DejaVu",
            fontSize=8.8,
            leading=11.5,
            textColor=dark_text,
            alignment=TA_LEFT,
        ),
    }
    return custom


def styled_table(data, col_widths, styles, left_cols=(0,)):
    """Temiz kurumsal tablo oluşturur."""
    formatted_rows = []
    for r_idx, row in enumerate(data):
        f_row = []
        for c_idx, cell in enumerate(row):
            if r_idx == 0:
                f_row.append(Paragraph(str(cell), styles["TableHeader"]))
            elif c_idx in left_cols:
                f_row.append(Paragraph(str(cell), styles["TableCellLeft"]))
            else:
                f_row.append(Paragraph(str(cell), styles["TableCell"]))
        formatted_rows.append(f_row)

    tbl = Table(formatted_rows, colWidths=col_widths)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return tbl


def fit_image(path: str, max_width_cm: float, max_height_cm: float = 10.5):
    """En-boy oranını koruyarak görseli sayfaya sığdırır."""
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w_px, h_px = im.size
    aspect = h_px / float(w_px)
    w = max_width_cm * cm
    h = w * aspect
    if h > max_height_cm * cm:
        h = max_height_cm * cm
        w = h / aspect
    img = Image(path, width=w, height=h)
    img.hAlign = "CENTER"
    return img


def generate_pdf():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base_dir, "results.json")
    figures_dir = os.path.join(base_dir, "figures")
    output_pdf = os.path.join(base_dir, "Proje_Raporu.pdf")

    with open(results_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    register_fonts()
    styles = build_styles()

    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=2.1 * cm,
        bottomMargin=2.0 * cm,
        title="Doğrusal ve Lojistik Regresyon ile İki Sınıflı Sınıflandırma Raporu",
    )

    ds = res["dataset"]
    sp = res["split"]
    hp = res["hyperparameters"]
    lin = res["linear_regression"]
    log = res["logistic_regression"]
    lin_m = lin["test_metrics"]
    log_m = log["test_metrics"]

    story = []

    repo_url = "https://github.com/devalimm/machinelearning1"
    banner_content = [
        [Paragraph("Makine Öğrenmesi Proje Raporu  |  Ali Mammadli (23060005)", styles["Subtitle"])],
        [Paragraph("İki Sınıflı Doğrusal Olmayan Veri Kümesi Üzerinde<br/>Doğrusal ve Lojistik Regresyon Karşılaştırması", styles["Title"])],
        [Paragraph(
            f'Veri Kümesi: Social Network Ads (Kaggle, n = 400)  |  Dil: Python (Sıfırdan NumPy)<br/>'
            f'GitHub Deposu: <link href="{repo_url}" color="#93C5FD"><u>{repo_url}</u></link>',
            styles["Subtitle"],
        )],
    ]
    banner = Table(banner_content, colWidths=[17.0 * cm])
    banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1E3A8A")),
                ("TOPPADDING", (0, 0), (0, 0), 14),
                ("BOTTOMPADDING", (0, -1), (0, -1), 14),
                ("LEFTPADDING", (0, 0), (-1, -1), 16),
                ("RIGHTPADDING", (0, 0), (-1, -1), 16),
                ("TOPPADDING", (0, 1), (0, 1), 4),
                ("BOTTOMPADDING", (0, 1), (0, 1), 6),
            ]
        )
    )
    story.append(banner)
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("1. Proje Amacı ve Problem Tanımı", styles["H1"]))
    story.append(
        Paragraph(
            "Bu projede, iki sınıflı (binary) ve doğrusal olarak ayrılamayan (non-linear) gerçek dünya "
            "verisi üzerinde <b>Doğrusal Regresyon (Linear Regression)</b> ve <b>Lojistik Regresyon "
            "(Logistic Regression)</b> algoritmaları kullanılarak sınıflandırma gerçekleştirilmiş ve "
            "her iki yöntemin aynı test kümesi üzerindeki başarımı karşılaştırılmıştır.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "Proje kuralları gereği her iki sınıflandırma algoritmasının ileri yayılım (tahmin), "
            "maliyet fonksiyonu, kısmi türev (gradyan) hesabı ve Gradient Descent parametre güncelleme "
            "adımları hazır makine öğrenimi kütüphaneleri kullanılmadan, yalnızca <b>NumPy</b> matris "
            "işlemleriyle sıfırdan kodlanmıştır. Veri inceleme için <i>Pandas</i>, görselleştirme için "
            "<i>Matplotlib</i> yardımcı kütüphanelerinden yararlanılmıştır.",
            styles["Body"],
        )
    )

    story.append(Paragraph("2. Veri Kümesi ve Özellikleri", styles["H1"]))
    story.append(
        Paragraph(
            "Çalışmada Kaggle açık veri platformundan alınan <b>Social Network Ads</b> veri kümesi "
            "kullanılmıştır. Veri kümesi, bir sosyal ağ reklamını gören kullanıcıların yaş ve tahmini "
            "yıllık maaş bilgilerine göre reklamı yapılan ürünü satın alıp almadıklarını (<b>Purchased: 0 veya 1</b>) "
            "içermektedir.",
            styles["Body"],
        )
    )

    ds_table_data = [
        ["Özellik / Nitelik", "Açıklama / Değer"],
        ["Kaynak", "Kaggle — Social Network Ads (Social_Network_Ads.csv)"],
        ["Toplam Örnek Sayısı", f"{ds['n_samples']} satır (şart: en az 100 örnek)"],
        ["Bağımsız Değişkenler (X)", "Age (Yaş) ve EstimatedSalary (Tahmini Yıllık Maaş, $)"],
        ["Hedef Değişken (y)", f"Purchased — İki sınıflı: 0 (Satın Almadı: {ds['class_counts']['0']}, %{ds['class_counts']['0']/ds['n_samples']*100:.1f}), 1 (Satın Aldı: {ds['class_counts']['1']}, %{ds['class_counts']['1']/ds['n_samples']*100:.1f})"],
        ["Eksik Değer / Mükerrer", f"Eksik değer: {sum(ds['missing_values'].values())} | Aynı yaş-maaş-sonuç kombinasyonuna sahip satır: {ds['duplicate_rows']}"],
        ["Age İstatistikleri", f"Min: {ds['age_stats']['min']:.0f}, Maks: {ds['age_stats']['max']:.0f}, Ort: {ds['age_stats']['mean']:.2f}, Std: {ds['age_stats']['std']:.2f}"],
        ["EstimatedSalary İstatistikleri", f"Min: {ds['salary_stats']['min']:,.0f} $, Maks: {ds['salary_stats']['max']:,.0f} $, Ort: {ds['salary_stats']['mean']:,.2f} $, Std: {ds['salary_stats']['std']:,.2f} $"],
        ["Hedefle Korelasyon (r)", f"Age-Purchased: {ds['correlations']['Age_Purchased']:.3f} | EstimatedSalary-Purchased: {ds['correlations']['EstimatedSalary_Purchased']:.3f}"],
    ]
    story.append(styled_table(ds_table_data, [5.4 * cm, 11.6 * cm], styles, left_cols=(0, 1)))
    story.append(Spacer(1, 0.35 * cm))

    story.append(
        KeepTogether(
            [
                Paragraph(
                    "<b>Veri Kümesinin Doğrusal Olmama Özelliği:</b> Şekil 1 incelendiğinde, <i>Satın Almadı (0)</i> "
                    "sınıfının düşük-orta yaş ve düşük-orta maaş bölgesinde yoğunlaştığı; <i>Satın Aldı (1)</i> "
                    "sınıfının ise hem yüksek maaşlı genç/orta yaş grubunda (Yaş ~27–40, Maaş > 90.000 $) hem de "
                    "düşük maaşlı ileri yaş grubunda (Yaş ≥ 45, Maaş 20.000–50.000 $) kümelendiği görülmektedir. "
                    "İki sınıf tek bir düz doğru ile eksiksiz ayrılamadığından veri kümesi <b>doğrusal olmayan</b> bir yapıya sahiptir.",
                    styles["Body"],
                ),
                fit_image(os.path.join(figures_dir, "1_veri_dagilimi.png"), max_width_cm=13.5, max_height_cm=8.5),
                Paragraph("Şekil 1: Social Network Ads veri kümesinin Yaş ve Tahmini Maaş uzayındaki iki sınıflı dağılımı.", styles["Caption"]),
            ]
        )
    )

    story.append(Paragraph("3. Veri Ön İşleme Adımları", styles["H1"]))
    story.append(
        Paragraph(
            "Modellerin kararlı eğitilebilmesi ve adil değerlendirilebilmesi için aşağıdaki ön işleme adımları uygulanmıştır:",
            styles["Body"],
        )
    )
    story.append(Paragraph("• <b>Veri Bütünlüğü Kontrolü:</b> Veri kümesinde hiç eksik (NaN) değer bulunmamaktadır. Demografik veride farklı kullanıcıların aynı yaş ve yuvarlanmış maaş değerine sahip olması doğal olduğundan tüm 400 gözlem korunmuştur.", styles["Bullet"]))
    story.append(Paragraph(f"• <b>Tabakalı Eğitim/Test Ayrımı (Stratified Split):</b> Veri kümesi sınıf oranları korunarak (sabit rastgele tohum <i>random_state={hp['random_state']}</i>) <b>%80 Eğitim ({sp['train_size']} örnek; 0: {sp['train_class_0']}, 1: {sp['train_class_1']})</b> ve <b>%20 Test ({sp['test_size']} örnek; 0: {sp['test_class_0']}, 1: {sp['test_class_1']})</b> olarak ayrılmıştır.", styles["Bullet"]))
    story.append(Paragraph(f"• <b>Z-Score Standartlaştırma:</b> Yaş (18–60) ile Tahmini Maaş (15.000–150.000) arasındaki ölçek farkı Gradient Descent yakınsamasını bozacağından her özellik sıfır ortalama ve birim varyansa dönüştürülmüştür. Veri sızıntısını (data leakage) önlemek amacıyla ortalama (μ = [{sp['scaler_mean'][0]:.2f}, {sp['scaler_mean'][1]:.2f}]) ve standart sapma (σ = [{sp['scaler_std'][0]:.2f}, {sp['scaler_std'][1]:.2f}]) <b>yalnızca eğitim kümesinden</b> hesaplanıp hem eğitim hem test verisine uygulanmıştır.", styles["Bullet"]))
    story.append(Paragraph("z = (x - μ_eğitim) / σ_eğitim", styles["Formula"]))

    story.append(PageBreak())
    story.append(Paragraph("4. Uygulanan Yöntemler (Sıfırdan Algoritma Tasarımı)", styles["H1"]))
    story.append(
        Paragraph(
            f"Her iki algoritma da aynı hiperparametrelerle (öğrenme oranı <b>α = {hp['learning_rate']}</b>, "
            f"iterasyon sayısı <b>T = {hp['n_iterations']}</b>, sınıflandırma eşiği <b>τ = {hp['threshold']}</b>) "
            "Gradient Descent (Gradyan İnişi) yöntemiyle sıfırdan eğitilmiştir.",
            styles["Body"],
        )
    )

    story.append(Paragraph("4.1. Doğrusal Regresyon ile Sınıflandırma (LinearRegressionClassifier)", styles["H2"]))
    story.append(
        Paragraph(
            "Doğrusal regresyon, bağımsız değişkenlerin doğrusal bir birleşimi ile sürekli bir çıktı üretir. "
            "Sınıflandırma amacıyla modelin ürettiği sürekli skor 0.5 eşik değeri ile karşılaştırılarak 0 veya 1 sınıfına atanır:",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "Tahmin Fonksiyonu :  ŷ = X·w + b<br/>"
            "Maliyet (MSE)     :  J(w, b) = (1 / 2m) · Σ (ŷ_i - y_i)²<br/>"
            "Gradyanlar        :  dw = (1 / m) · Xᵀ · (ŷ - y) ,   db = (1 / m) · Σ (ŷ - y)<br/>"
            "Güncelleme        :  w ← w - α·dw ,   b ← b - α·db<br/>"
            "Sınıf Kararı      :  ŷ ≥ 0.5 ise 1 (Satın Aldı), aksi halde 0 (Satın Almadı)",
            styles["Formula"],
        )
    )
    story.append(
        Paragraph(
            f"Eğitim sonunda öğrenilen parametreler: <b>w = [{lin['weights'][0]:.4f}, {lin['weights'][1]:.4f}]</b>, "
            f"<b>b = {lin['bias']:.4f}</b>. Başlangıç MSE maliyeti {lin['initial_loss']:.4f} iken "
            f"{hp['n_iterations']} iterasyon sonunda <b>{lin['final_loss']:.4f}</b> değerine yakınsamıştır.",
            styles["Body"],
        )
    )

    story.append(Paragraph("4.2. Lojistik Regresyon ile Sınıflandırma (LogisticRegressionClassifier)", styles["H2"]))
    story.append(
        Paragraph(
            "Lojistik regresyon, doğrusal birleşimi <b>Sigmoid (Lojistik)</b> fonksiyonundan geçirerek çıktıyı "
            "doğrudan (0, 1) aralığında bir olasılık değerine dönüştürür ve sınıflandırmaya özgü <b>Binary Cross-Entropy "
            "(Log-Loss)</b> maliyet fonksiyonunu minimize eder:",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "Lineer & Sigmoid  :  z = X·w + b ,   ŷ = σ(z) = 1 / (1 + e⁻ᶻ) ∈ (0, 1)<br/>"
            "Maliyet (Log-Loss):  J(w, b) = -(1 / m) · Σ [ y_i·log(ŷ_i) + (1 - y_i)·log(1 - ŷ_i) ]<br/>"
            "Gradyanlar        :  dw = (1 / m) · Xᵀ · (ŷ - y) ,   db = (1 / m) · Σ (ŷ - y)<br/>"
            "Güncelleme        :  w ← w - α·dw ,   b ← b - α·db<br/>"
            "Sınıf Kararı      :  P(y=1|x) = ŷ ≥ 0.5 ise 1 (Satın Aldı), aksi halde 0 (Satın Almadı)",
            styles["Formula"],
        )
    )
    story.append(
        Paragraph(
            f"Eğitim sonunda öğrenilen parametreler: <b>w = [{log['weights'][0]:.4f}, {log['weights'][1]:.4f}]</b>, "
            f"<b>b = {log['bias']:.4f}</b>. Başlangıç Log-Loss maliyeti {log['initial_loss']:.4f} iken "
            f"{hp['n_iterations']} iterasyon sonunda <b>{log['final_loss']:.4f}</b> değerine yakınsamıştır. "
            "Her iki modelde de Yaş ağırlığının (w₁) Maaş ağırlığından (w₂) yaklaşık 1.8 kat büyük olması, "
            "satın alma kararında yaşın maaşa kıyasla daha baskın bir etken olduğunu göstermektedir.",
            styles["Body"],
        )
    )

    story.append(
        KeepTogether(
            [
                fit_image(os.path.join(figures_dir, "2_maliyet_egrileri.png"), max_width_cm=16.5, max_height_cm=6.5),
                Paragraph("Şekil 2: Doğrusal Regresyon (MSE) ve Lojistik Regresyon (Binary Cross-Entropy) eğitim yakınsama eğrileri.", styles["Caption"]),
            ]
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("5. Elde Edilen Sonuçlar ve Performans Değerlendirmesi", styles["H1"]))
    story.append(
        Paragraph(
            "Eğitilen her iki model, eğitim sırasında hiç görmedikleri aynı <b>80 örneklik test kümesi</b> "
            "üzerinde Doğruluk (Accuracy), Kesinlik (Precision), Duyarlılık (Recall), Özgüllük (Specificity), "
            "F1-Skoru ve ROC-AUC metrikleri ile değerlendirilmiştir.",
            styles["Body"],
        )
    )

    perf_table_data = [
        ["Değerlendirme Metriği", "Doğrusal Regresyon", "Lojistik Regresyon", "Fark (Loj. - Doğ.)"],
        ["Eğitim Doğruluğu (Train Acc)", f"{lin['train_accuracy']:.4f} (%{lin['train_accuracy']*100:.2f})", f"{log['train_accuracy']:.4f} (%{log['train_accuracy']*100:.2f})", f"{(log['train_accuracy']-lin['train_accuracy'])*100:+.2f} puan"],
        ["Test Doğruluğu (Accuracy)", f"{lin_m['accuracy']:.4f} (%{lin_m['accuracy']*100:.2f})", f"{log_m['accuracy']:.4f} (%{log_m['accuracy']*100:.2f})", f"{(log_m['accuracy']-lin_m['accuracy'])*100:+.2f} puan"],
        ["Kesinlik (Precision)", f"{lin_m['precision']:.4f}", f"{log_m['precision']:.4f}", f"{log_m['precision']-lin_m['precision']:+.4f}"],
        ["Duyarlılık (Recall / TPR)", f"{lin_m['recall']:.4f}", f"{log_m['recall']:.4f}", f"{log_m['recall']-lin_m['recall']:+.4f}"],
        ["Özgüllük (Specificity / TNR)", f"{lin_m['specificity']:.4f}", f"{log_m['specificity']:.4f}", f"{log_m['specificity']-lin_m['specificity']:+.4f}"],
        ["F1-Skoru", f"{lin_m['f1_score']:.4f}", f"{log_m['f1_score']:.4f}", f"{log_m['f1_score']-lin_m['f1_score']:+.4f}"],
        ["ROC-AUC", f"{lin_m['roc_auc']:.4f}", f"{log_m['roc_auc']:.4f}", f"{log_m['roc_auc']-lin_m['roc_auc']:+.4f}"],
        ["Test Çıktı Aralığı [Min, Maks]", f"[{lin_m['score_min']:.3f}, {lin_m['score_max']:.3f}]", f"[{log_m['score_min']:.3f}, {log_m['score_max']:.3f}]", "Olasılık sınırı [0, 1]"],
    ]
    story.append(styled_table(perf_table_data, [5.5 * cm, 3.8 * cm, 3.8 * cm, 3.9 * cm], styles, left_cols=(0,)))
    story.append(Spacer(1, 0.35 * cm))

    story.append(
        KeepTogether(
            [
                fit_image(os.path.join(figures_dir, "3_karar_sinirlari.png"), max_width_cm=16.8, max_height_cm=7.2),
                Paragraph("Şekil 3: Doğrusal Regresyon ve Lojistik Regresyon modellerinin Test Kümesi üzerindeki karar sınırları.", styles["Caption"]),
            ]
        )
    )

    story.append(
        KeepTogether(
            [
                fit_image(os.path.join(figures_dir, "4_karisiklik_matrisleri.png"), max_width_cm=14.8, max_height_cm=6.2),
                Paragraph("Şekil 4: Test kümesi (n = 80) için Karışıklık Matrisleri (TN: Doğru Negatif, FP: Yanlış Pozitif, FN: Yanlış Negatif, TP: Doğru Pozitif).", styles["Caption"]),
            ]
        )
    )

    story.append(Paragraph("6. İki Algoritmanın Karşılaştırmalı Değerlendirmesi ve Yorumlar", styles["H1"]))

    story.append(
        KeepTogether(
            [
                fit_image(os.path.join(figures_dir, "5_performans_karsilastirma.png"), max_width_cm=16.8, max_height_cm=6.8),
                Paragraph("Şekil 5: Test kümesi üzerinde ROC eğrileri ve temel performans metriklerinin karşılaştırması.", styles["Caption"]),
            ]
        )
    )

    story.append(Paragraph("1. Sınıflandırma Performansı Karşılaştırması:", styles["H2"]))
    story.append(
        Paragraph(
            f"Aynı test kümesi üzerinde <b>Lojistik Regresyon</b>; Doğruluk (%{log_m['accuracy']*100:.2f} vs %{lin_m['accuracy']*100:.2f}), "
            f"Kesinlik ({log_m['precision']:.4f} vs {lin_m['precision']:.4f}), Duyarlılık ({log_m['recall']:.4f} vs {lin_m['recall']:.4f}) ve "
            f"F1-Skoru ({log_m['f1_score']:.4f} vs {lin_m['f1_score']:.4f}) metriklerinin tamamında Doğrusal Regresyonu geride bırakmıştır. "
            f"Karışıklık matrislerinde her iki model de 51 negatif örneğin 47'sini doğru bilmiş (Specificity = {log_m['specificity']:.4f}, FP = 4), "
            f"ancak Lojistik Regresyon pozitif sınıfta 1 ek doğru tahmin (TP: {log_m['confusion_matrix']['TP']} vs {lin_m['confusion_matrix']['TP']}) "
            f"yaparak F1-Skorunda +0.0259 puanlık anlamlı bir üstünlük sağlamıştır.",
            styles["Body"],
        )
    )

    story.append(Paragraph("2. Olasılık Yorumlanabilirliği ve Çıktı Sınırları:", styles["H2"]))
    story.append(
        Paragraph(
            f"Doğrusal Regresyon test kümesinde <b>[{lin_m['score_min']:.3f}, {lin_m['score_max']:.3f}]</b> aralığında ham değerler "
            "üretmiştir. Negatif (-0.323) ve 1'den büyük (1.044) tahminler olasılık teorisine aykırıdır; bu durum Doğrusal Regresyonun "
            "sınıflandırma için tasarlanmadığının somut bir göstergesidir. Buna karşılık Lojistik Regresyon, Sigmoid fonksiyonu sayesinde "
            f"tüm çıktıları <b>[{log_m['score_min']:.3f}, {log_m['score_max']:.3f}]</b> aralığında tutarak doğrudan satın alma olasılığı "
            "P(y=1|x) şeklinde yorumlanabilir sonuçlar üretmiştir.",
            styles["Body"],
        )
    )

    story.append(Paragraph("3. Doğrusal Olmayan Veri Kümesinde Karar Sınırı Kısıtı:", styles["H2"]))
    story.append(
        Paragraph(
            "Şekil 3'te görüldüğü üzere hem Doğrusal Regresyon hem de Lojistik Regresyon, ham özellikler (Age, EstimatedSalary) "
            "üzerinde <b>doğrusal bir karar sınırı (düz doğru)</b> oluşturmaktadır. Veri kümesindeki <i>Yaş 45–50, Maaş 20.000–45.000 $</i> "
            "bölgesi ile <i>Yaş 27–32, Maaş 100.000–140.000 $</i> bölgesindeki satın alan kullanıcılar bu düz doğrunun solunda kaldığı için "
            f"her iki modelde de <b>Yanlış Negatif (FN: {lin_m['confusion_matrix']['FN']} ve {log_m['confusion_matrix']['FN']})</b> sayısı "
            "yüksek çıkmış ve Duyarlılık ~%59–62 bandında kalmıştır. Ayrıca her iki modelin eğitim doğruluğu (~%85.5) ile test "
            "doğruluğu (~%80–81) arasındaki fark küçük olduğundan aşırı öğrenme gözlenmemiştir; performans sınırı doğrudan "
            "verinin doğrusal olmayan yapısından kaynaklanmaktadır.",
            styles["Body"],
        )
    )

    story.append(Paragraph("4. Maliyet Fonksiyonu ve ROC-AUC Değerlendirmesi:", styles["H2"]))
    story.append(
        Paragraph(
            f"ROC-AUC değerleri iki model için birbirine çok yakındır (Doğrusal: {lin_m['roc_auc']:.4f}, Lojistik: {log_m['roc_auc']:.4f}). "
            "Bunun nedeni iki boyutlu uzayda her iki modelin de benzer eğime sahip doğrusal sıralama skorları üretmesidir. Ancak sabit 0.5 "
            "karar eşiğinde Binary Cross-Entropy (Log-Loss) optimizasyonu sınıf sınırına yakın noktalarda yanlış tahminleri logaritmik olarak "
            "daha güçlü cezalandırdığı için Lojistik Regresyon, karar sınırını daha dengeli konumlandırmış ve sınıflandırmada daha yüksek "
            "Doğruluk ve F1-Skoru elde etmiştir.",
            styles["Body"],
        )
    )

    story.append(Paragraph("7. Sonuç ve Proje Dosya Yapısı", styles["H1"]))
    story.append(
        Paragraph(
            "Yapılan deneyler sonucunda; iki sınıflı sınıflandırma problemlerinde <b>Lojistik Regresyonun</b> hem teorik (geçerli olasılık "
            "çıktısı, Log-Loss maliyeti, uç değerlere dayanıklılık) hem de pratik (daha yüksek Accuracy, Precision, Recall ve F1-Skoru) "
            "açıdan Doğrusal Regresyona göre daha uygun bir algoritma olduğu doğrulanmıştır. Doğrusal olmayan veri kümelerinde bu iki modelin "
            "başarımını daha da artırmak için ileri çalışma olarak polinomsal özellik dönüşümü veya doğrusal olmayan sınıflandırıcılar "
            "(örn. RBF Çekirdekli SVM, Karar Ağaçları, Yapay Sinir Ağları) kullanılabilir.",
            styles["Body"],
        )
    )

    story.append(
        Paragraph(
            f'<b>Kaynak Kod Deposu (GitHub):</b> '
            f'<link href="{repo_url}" color="#1D4ED8"><u>{repo_url}</u></link>',
            styles["Body"],
        )
    )

    files_table = [
        ["Dosya Adı", "Görevi"],
        ["Social_Network_Ads.csv", "Kaggle'dan alınan 400 örneklik iki sınıflı veri kümesi"],
        ["preprocessing.py", "Veri yükleme, istatistik özeti, Stratified %80/%20 ayrım ve Z-Score standartlaştırma"],
        ["models.py", "LinearRegressionClassifier ve LogisticRegressionClassifier (sıfırdan NumPy kodu)"],
        ["evaluation.py", "Karışıklık matrisi, Accuracy, Precision, Recall, F1, ROC-AUC ve 5 adet grafik üretimi"],
        ["main.py", "Tüm süreci uçtan uca çalıştıran, metrikleri raporlayan ve results.json üreten ana betik"],
    ]
    story.append(Spacer(1, 0.15 * cm))
    story.append(styled_table(files_table, [4.6 * cm, 12.4 * cm], styles, left_cols=(0, 1)))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"PDF raporu başarıyla oluşturuldu: {output_pdf}")


if __name__ == "__main__":
    generate_pdf()
