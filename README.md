# Cat vs Dog Predictor API

Bu proje, önceden eğitilmiş üç farklı model (`resnet_feature`, `resnet_finetune`, `cnn_functional`) kullanarak bir web API sağlar. FastAPI tabanlı backend ve basit bir HTML/JavaScript ön yüz ile çalışır.

---

## İçindekiler

* [Ön Koşullar](#ön-koşullar)
* [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
* [API Uç Noktaları (Endpoints)](#api-uç-noktaları-endpoints)
* [Ön Yüz (HTML)](#ön-yüz-html)

---

## Veri Kümesi (Dataset)

Bu projede kullanılan veri kümesi, kedi ve köpek resimlerinden oluşan yaygın bir veri setidir. Her resim dosyasının adı `<label>_<id>.jpg` formatındadır:

* `cat` sınıfı için örnek ırklar: Abyssinian, Birman, Bengal, Bombay, British, Egyptian, Maine, Persian, american, english → binary değeri 0
* Diğer tüm ırklar veya resimler → binary değeri 1 (köpek)

Test seti, tüm resim dosyalarının %20'si kullanılarak ayrıştırılmıştır.

---

## Ön Koşullar

* Python 3.10 veya üzeri
* Aşağıdaki paketler (sürüm bilgileri önerilir):

  * `fastapi`
  * `uvicorn` (standard)
  * `tensorflow` (2.18)
  * `numpy` (1.26.x)
  * `pandas`

---


## API sunucusunu başlatın:

   ```bash
uvicorn fast_api:app --reload --host 0.0.0.0 --port 8000
````

---

## API Uç Noktaları (Endpoints)

| Yöntem | Yol                       | Açıklama                                      |
| ------ | ------------------------- | --------------------------------------------- |
| GET    | `/models`                 | Yüklü modelleri listeler                      |
| POST   | `/predict?model_name=<m>` | Tek resim yükleyip tahmin yapar               |
| GET    | `/report?model_name=<m>`  | Seçilen modelin classification raporunu döner |

### Örnek: Model Listesi

```bash
curl http://127.0.0.1:8000/models
```

```
{"available_models":["resnet_feature","resnet_finetune","cnn_functional"]}
```

### Örnek: Tahmin

```bash
curl -X POST "http://127.0.0.1:8000/predict?model_name=cnn_functional" \
  -F "file=@/path/to/image.jpg"
```

```json
{"model":"cnn_functional","prediction":"dog","score":0.94}
```

### Örnek: Rapor

```bash
curl "http://127.0.0.1:8000/report?model_name=resnet_finetune"
```

```
precision    recall  f1-score   support

  cat       0.94      0.96      0.95       960
  dog       0.98      0.97      0.97      1996

accuracy                           0.97      2956
...
```

---

## Ön Yüz (HTML)

* Proje kök dizininde bulunan `index.html` dosyasını tarayıcıda şu adresten açın:
  `http://127.0.0.1:8000/`

* Model seçin, resim yükleyin ve tahmin sonuçlarını canlı görün.

