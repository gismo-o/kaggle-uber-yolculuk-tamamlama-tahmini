# Uber Yolculuk Tamamlama Tahmini

Amaç: Bir rezervasyonun tamamlanıp tamamlanmayacağını (Completed=1, aksi=0) erken tahmin ederek; iptal/yarım kalma riski yüksek kayıtlar için teşvik, kuyruk yönetimi veya bilgilendirme gibi operasyonel aksiyonları mümkün kılmak.

Bu doğrultuda çalışma, [***Uber Ride Analytics Dataset***](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard/data) verisi üzerinde bir tamamlama (Completed) tahmin modeli geliştirir.


### ADIMLAR:
1. Veri Temizleme
- Booking Status normalizasyonu ve ikili hedefe (Completed=1, diğer durumlar=0) map etme.

- Booking ID duplikasyonları: aynı ID’nin en son statüsü tutularak tekilleştirme.

- "CNR...", "CID..." gibi tırnaklı ID alanlarının strip edilmesi.

- Date + Time → booking_dt zaman damgası; buradan hour, dayofweek, is_weekend türetildi.

- Eksik değerler: skor ve parasal alanlarda gerektiği yerde impute (median) ya da kullanılmayan kolonlardan çıkarım.

- 176 pickup ve 176 drop lokasyon ismi: trim/format kontrolü.

2. Keşifsel Veri Analizi (EDA)

- Hedef dağılımı & durumlar
  - Booking Status dağılımı: Completed çoğunlukta; iptaller Cancelled by Customer/Driver, No Driver Found, Incomplete olarak kümeleniyor.
  - Driver Ratings / Customer Rating ≈ 56.5K eksik → modelde kullanılmadı (veya gerekli yerde median ile tamamlandı).
  - Avg CTAT’ta ~47.6K eksik; diğer sayısallarda düşük-orta düzey eksik → median impute.
- Eksik değerler
  - Payment Method ≈ 47.6K eksik (iş kuralları gereği bazı kayıtlarda yok) → modelde kullanılmadı.
  - Driver Ratings / Customer Rating ≈ 56.5K eksik → modelde kullanılmadı (veya gerekli yerde median ile tamamlandı).
  - Avg CTAT’ta ~47.6K eksik; diğer sayısallarda düşük-orta düzey eksik → median impute.
- Aykırı değer (IQR) & dağılımlar
  - Booking Value: sağ kuyruk belirgin; Q3+1.5·IQR üstünde ~3,422 gözlem → log1p dönüşümü uygulandı (Booking Value Log).
  - Avg VTAT (2–20 dk), Avg CTAT (10–45 dk), Ride Distance (1–50 km): aşırı uç yok/çok sınırlı; boxplot kontrolleri temiz.
  - Ratings 1–5 aralığı: [3, 5] aralığında; veri tutarlı
- Lokasyonlar
  - Pickup 176, Drop 176 benzersiz nokta; adresler trim/format sonrası tutarlı.
  - Uzun kuyruk nedeniyle modelde Top-N + “Other” yaklaşımı benimsendi.
- Zaman kalıpları
  - hour, dayofweek ile gündüz saatleri daha yüksek tamamlama; gece (22–23) hafif riskli.
  - Döngüsellik için hour_sin/cos, dow_sin/cos eklendi.

3. Feature Engineering
- Sızıntı yaratabilecek kolonlar çıkarıldı
- Sayısal özellikler: Avg VTAT, Avg CTAT, Ride Distance, Booking Value Log (log1p), hour, dayofweek, is_weekend, döngüsel (hour_sin/cos, dow_sin/cos).
- Kategorikler: Vehicle Type (OHE) + Top-N PickupLocTop/DropLocTop (en sık ~15 + “Other”).
- Ön-işleme: SimpleImputer(median) + OneHotEncoder(handle_unknown="ignore") → ColumnTransformer pipeline.
- Toplam 50 özellik ile modelleme.

4. Modelleme
- Referans: Logistic Regression (baseline).
- **Ana model: LightGBM (LGBMClassifier)**
  - n_estimators=2000, learning_rate=0.03, num_leaves=63, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0
  - Class weights: sınıf dengesizliği için frekansın tersi.
  - Eğitim metriği: AUC.
5. Eşik (Threshold) Optimizasyonu
- Operasyonel gereksinim için class 0 (iptal) F1’i maksimize eden en iyi eşik tarandı.
- Bulunan best_th ≈ 0.19 ile 0.5 eşiğine göre macro-F1 ve accuracy iyileşti.

6. Sızıntı (Leakage) Kontrolü
- TimeSeriesSplit (5-fold) ile zaman-temelli CV: katmanlar arasında AUC ≈ 0.9935 ± 0.0004, metrikler tutarlı → sızıntı işareti yok.
7. Açıklanabilirlik (SHAP)
- En etkili değişkenler: Ride Distance >> Avg CTAT >> Booking Value Log >> Avg VTAT.
- Yorum: kısa ve düşük ücretli yolculuklarda iptal riski yüksek; uzun/“değerli” yolculuklar daha çok tamamlanıyor.


### Sonuç Metrikleri

**ROC-AUC: 0.9937429622079482**

Eşik = 0.5
| class | precision | recall | f1-score | support |
| ----: | --------: | -----: | -------: | ------: |
|     0 |     0.938 |  0.926 |    0.932 |  11 305 |
|     1 |     0.955 |  0.963 |    0.959 |  18 449 |


Eşik = 0.19 (class 0 F1 optimize)
| class | precision | recall | f1-score | support |
| ----: | --------: | -----: | -------: | ------: |
|     0 |     0.997 |  0.892 |    0.942 |  11 305 |
|     1 |     0.938 |  0.998 |    0.967 |  18 449 |

***yorum:*** 0.19 eşiği ile accuracy ve macro-F1 artıyor; class 1 (tamamlananlar) için recall ≈ 0.998 değerine ulaşılıyor. Class 0’da recall biraz düşse de F1 0.942 ile güçlü.
