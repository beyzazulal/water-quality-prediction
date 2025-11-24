# water-quality-prediction
A machine learning project to predict water potability using Decision Tree and Random Forest models.
# 💧 Dünya Su Kalitesi Tahmin Projesi

Bu proje, bir su örneğinin **içilebilir (potable) olup olmadığını** makine öğrenmesi kullanarak tahmin etmeyi amaçlar.  
Veri seti, fiziksel ve kimyasal su ölçümlerinden oluşmaktadır.  

Proje; veri analizi, eksik değer işleme, model eğitimi, değerlendirme ve hiperparametre optimizasyonu gibi uçtan uca makine öğrenmesi adımlarını içermektedir.

---

## 📘 İçerik

1. Veri Analizi (EDA)
2. Eksik Değer Analizi
3. Normalizasyon
4. Karar Ağacı (Decision Tree)
5. Rastgele Orman (Random Forest)
6. Confusion Matrix & Precision Score
7. Hiperparametre Optimizasyonu (RandomizedSearchCV)
8. Model Sonuçları

---

## 📊 Veri Seti

**water_potability.csv**

Özellikler:
- ph  
- Hardness  
- Solids  
- Chloramines  
- Sulfate  
- Conductivity  
- Organic_carbon  
- Trihalomethanes  
- Turbidity  
- Potability (Hedef değişken)

---

## 🔍 EDA (Keşifsel Veri Analizi)

Projede aşağıdaki analizler yapılmıştır:

- Değişkenlerin dağılım grafikleri  
- Kayıp değer analizi  
- Korelasyon matrisinin incelenmesi  
- Potability sınıf dağılımının görselleştirilmesi  

Örnek görseller:
> *(Eğer istersen “images” klasörü açarız, grafikleri oraya koyarız ve README’ye ekleriz.)*

---

## 🛠 Veri Ön İşleme

- Eksik değerler **ortalama ile doldurulmuştur** (ph, Sulfate, Trihalomethanes).
- Özellikler **Min-Max normalizasyonu** ile ölçeklendirilmiştir.
- Veri eğitim (%70) ve test (%30) olarak bölünmüştür.

---

## 🤖 Modeller

Kullanılan modeller:

- **DecisionTreeClassifier (max_depth=5)**
- **RandomForestClassifier**

Her model için:
- Precision Score
- Confusion Matrix hesaplanmıştır.

---

## 📈 Model Sonuçları

| Model | Precision |
|-------|-----------|
| Decision Tree | 0.60 |
| Random Forest | 0.625 |

Random Forest modeli daha yüksek precision ile daha iyi performans göstermiştir.

---

## 🔧 Hiperparametre Tuning

Random Forest modeli için:

- `n_estimators`
- `max_depth`
- `max_features`

parametreleri **RandomizedSearchCV + RepeatedStratifiedKFold** kullanılarak optimize edilmiştir.

**En iyi parametreler:**

```python
{'n_estimators': 50, 'max_features': 'sqrt', 'max_depth': 16}
