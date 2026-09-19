# Handwritten Character Recognition (CNN Classifier)

Classifier berbasis CNN untuk mengenali huruf kapital tulisan tangan (A–Z), dilatih pada dataset [A-Z Handwritten Alphabets](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format) dari Kaggle.

## Gambaran Umum

Ini adalah klasifikasi gambar karakter tunggal — dikasih satu gambar grayscale 28×28 huruf tulisan tangan, prediksi termasuk huruf yang mana dari 26 huruf. Ini tugas yang lebih sederhana dan komplementer dari text recognition berbasis urutan (lihat [`text-recognition-crnn-ctc`](https://github.com/arielshakaramiro/text-recognition-crnn-ctc)): nggak ada LSTM, nggak ada CTC loss, cuma CNN classifier di atas crop karakter tunggal.

## Dataset

[A-Z Handwritten Alphabets in CSV format](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format) — ~370K gambar grayscale 28×28 berlabel huruf kapital tulisan tangan, diunduh lewat `kagglehub`.

![Distribusi kelas](images/class_distribution.png)

> Kelasnya cukup timpang — `O` dan `S` punya puluhan ribu sampel sementara `F` dan `I` cuma sekitar 1.000. Ini ditangani saat training lewat class weighting (lihat di bawah), dan hasil per-kelas lebih jauh di bawah mengonfirmasi itu benar-benar berhasil, bukan cuma diasumsikan.

![Contoh dataset mentah](images/sample_dataset.png)

## Arsitektur

CNN kecil: 3 blok konvolusi (Conv2D + MaxPool) yang masuk ke dense layer (dengan `Dropout(0.3)` untuk regularisasi) dan output softmax 26-kelas. Di-compile dengan Adam (lr=1e-3) dan categorical cross-entropy.

Training memakai:
- **Class weighting** (`sklearn.utils.class_weight.compute_class_weight`) supaya kesalahan di huruf yang kurang terwakili dihitung lebih berat saat training
- **`EarlyStopping`** (monitor `val_accuracy`, patience 4, kembalikan bobot terbaik) dan **`ReduceLROnPlateau`** (monitor `val_loss`, turunkan learning rate setengah kalau mandek) — bukan jumlah epoch tetap tanpa pengawasan
- `train_test_split` yang di-stratify (`stratify=y, random_state=42`) supaya proporsi kelas di split train/test terjaga dan hasilnya reproducible

## Hasil (terverifikasi — hasil run Colab asli)

| Metrik | Nilai |
|---|---|
| Validation accuracy terbaik | 99.10% (epoch 20) |
| Validation loss (epoch sama) | 0.0391 |
| Training accuracy (epoch sama) | 99.18% |
| Training loss (epoch sama) | 0.0251 |
| Epoch dilatih | 20 (EarlyStopping patience 4 tidak terpicu — akurasi masih terus naik) |

![Kurva training](images/training_curves.png)

Kombinasi class weighting, dropout, dan LR scheduling menghasilkan model yang kuat dan general, bukan cuma model yang dilatih lebih lama tanpa arah.

## Evaluasi: Performa Per-Kelas

Angka yang sebelumnya nggak bisa dibuktikan di repo ini — apakah class weighting beneran membantu huruf yang kurang terwakili — sekarang diukur langsung lewat `classification_report` lengkap dan confusion matrix di test set:

| Huruf | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| `F` (minoritas) | 233 | 0.983 | 0.991 | 0.987 |
| `I` (minoritas) | 224 | 0.978 | 0.991 | 0.984 |
| `D` (precision terendah) | 2.027 | 0.916 | 0.990 | 0.951 |
| `O` (mayoritas) | 11.565 | 0.998 | 0.981 | 0.990 |
| `S` (mayoritas) | 9.684 | 0.999 | 0.994 | 0.996 |

Macro-average F1 (0.989) dan weighted-average F1 (0.991) nilainya berdekatan — pertanda bagus, artinya performanya nggak "ditarik naik" oleh kelas mayoritas sementara kelas minoritas ketinggalan jauh. `F` dan `I` (dua huruf paling kurang terwakili) hasilnya ada di rentang F1 yang mirip dengan `O` dan `S` (dua huruf paling banyak sampelnya) — inilah hasil konkret yang memang jadi tujuan class weighting.

Menariknya, huruf `D` — bukan kelas yang kecil — justru punya precision *terendah* di seluruh laporan (0.916), lebih rendah dari kedua huruf minoritas. Confusion matrix-nya kasih tau kenapa: sejumlah sampel `O` (kelas yang jauh lebih besar) salah diprediksi jadi `D`, menurunkan precision `D` meski recall `D` sendiri baik-baik saja. Ini bukan soal class imbalance — ini kemiripan bentuk spesifik antara dua huruf yang nggak diselesaikan oleh class weighting.

![Confusion matrix](images/confusion_matrix.png)

Cek kualitatif pada satu batch gambar test — label prediksi ditampilkan di tiap gambar:

![Contoh hasil prediksi](images/prediction_grid.png)

> **Catatan cakupan:** test set-nya tetap dari distribusi dataset yang sama dengan training (sumber sama, proses pengumpulan sama). Ini belum mengukur bagaimana model menangani gaya tulisan tangan, scanner, atau preprocessing yang beda signifikan dari dataset ini — itu butuh test set eksternal yang genuinely berbeda, yang belum jadi bagian dari run ini.

## Catatan setup

- **Kredensial Kaggle:** notebook ini pertama-tama cek Colab Secrets untuk `KAGGLE_USERNAME` / `KAGGLE_KEY`; kalau belum diset, akan diminta input manual. Ambil dari akun Kaggle kamu → Settings → API → Create New Token.
- **Penyimpanan model:** notebook ini mount Google Drive dan menyimpan model hasil training (`model_hand.h5`) ke `/content/drive/MyDrive/handwritten-character-recognition/`.

## Cara menjalankan

1. Buka `notebooks/handwritten_character_recognition.ipynb` di Google Colab
2. Jalankan semua cell — akan diminta kredensial Kaggle (kalau belum ada di Colab Secrets) dan akses Drive

## Struktur repo

```
.
├── notebooks/
│   └── handwritten_character_recognition.ipynb
├── images/
│   ├── class_distribution.png
│   ├── sample_dataset.png
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── prediction_grid.png
├── LICENSE
└── README.md
```

## Lisensi

MIT — lihat [LICENSE](LICENSE).
