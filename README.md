# Panduan Penggunaan Model di Google Colab

Berikut adalah langkah-langkah untuk menggunakan model ini di Google Colab:

## 1. Clone Repositori

Pertama, clone repositori ke lingkungan Colab Anda:

```python
!git clone https://github.com/fuji-184/fuji_model.git
```

## 2. Masuk ke Direktori Repositori
```python
cd fuji_model
```

## 3. Install Dependencies
```python
!pip install -r requirements.txt
```

## 4. Import librarinya
```python
from efnetv2_vit import buat_model, latih_model
```

## 5. Buat model dengan jumlah kelas yang diinginkan
```python
model = buat_model(jumlah_kelas=4)
```

## 6. Latih model
```python
model_hasil = latih_model(
    model=model,
    epochs=isi dengan jumlah epochs,
    path_train="isi dengan lokasi dataset training",
    path_val="isi dengan lokasi dataset validasi",
    path_test="isi dengan lokasi dataset testing"
)
```
