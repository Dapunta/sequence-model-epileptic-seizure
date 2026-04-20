# Sequence Model untuk Epileptic Seizure Recognition

## Ringkasan Project

Project ini membahas klasifikasi **5 kelas** pada dataset **Epileptic Seizure Recognition** menggunakan lima arsitektur sequence model berbasis TensorFlow, yaitu:

1. **RNN**
2. **LSTM**
3. **BiLSTM**
4. **GRU**
5. **BiGRU**

Seluruh eksperimen dijalankan dalam bentuk notebook terpisah agar setiap model dapat dianalisis secara mandiri, tetapi tetap memakai alur eksperimen dan metrik evaluasi yang sama.

---

## 1. Latar Belakang

Sinyal **electroencephalogram (EEG)** merekam aktivitas listrik otak dalam bentuk deret waktu. Pada kasus epilepsi, pola gelombang EEG dapat menunjukkan perbedaan karakteristik antara kondisi kejang, aktivitas pada area otak tertentu, serta kondisi subjek sehat. Karena data berbentuk urutan sinyal, pendekatan **sequence model** seperti RNN, LSTM, GRU, BiLSTM, dan BiGRU menjadi relevan untuk dibandingkan.

## 2. Tujuan

- membangun lima model sequence untuk klasifikasi EEG 5 kelas
- menjaga alur preprocessing, split data, training, dan evaluasi tetap konsisten
- membandingkan performa model dengan metrik yang sama
- melihat karakter kelas mana yang paling mudah dan paling sulit diprediksi
- menyusun baseline eksperimen recurrent model pada dataset epileptic seizure recognition

---

## 3. Dataset

### 3.1 Nama Dataset
**Epileptic Seizure Recognition**

### 3.2 Gambaran Data
Berdasarkan output notebook:

- shape awal data: **(11500, 180)**
- terdapat satu kolom indeks/string tambahan bernama **`Unnamed`**
- setelah pembersihan, data menjadi:
  - **178 fitur numerik** sinyal EEG
  - **1 kolom label target**
- total sampel: **11.500**
- label unik: **1, 2, 3, 4, 5**

### 3.3 Distribusi Kelas
Distribusi kelas pada notebook bersifat **seimbang**:

| Label | Jumlah Sampel |
|------:|--------------:|
| 1 | 2300 |
| 2 | 2300 |
| 3 | 2300 |
| 4 | 2300 |
| 5 | 2300 |

### 3.4 Makna Label
Pada dataset Epileptic Seizure Recognition, label umumnya dipakai sebagai:

| Label | Deskripsi |
|------:|-----------|
| 1 | Seizure activity |
| 2 | EEG dari area tumor |
| 3 | EEG dari area sehat pada pasien epilepsi |
| 4 | Subjek sehat, mata tertutup |
| 5 | Subjek sehat, mata terbuka |

---

## 4. Pembagian Data

Pembagian data dilakukan secara **stratified split** agar proporsi kelas tetap terjaga pada setiap subset.

Hasil split yang tersimpan pada notebook:

- **Train**: 8050 sampel
- **Validation**: 1725 sampel
- **Test**: 1725 sampel

Karena distribusi seimbang, support pada classification report test menjadi:

- **345 sampel per kelas**

---

## 5. Alur Eksperimen Umum

Struktur blok pada semua notebook konsisten, yaitu:

1. **Import library**
2. **Konfigurasi eksperimen**
3. **Mencari file dataset di `/kaggle/input`**
4. **Membaca dataset**
5. **Membersihkan struktur kolom**
6. **Memisahkan fitur dan label**
7. **Melihat distribusi kelas**
8. **Encode label**
9. **Train / validation / test split**
10. **Normalisasi per sampel**
11. **Membentuk tensor sequence**
12. **Membuat pipeline `tf.data`**
13. **Membangun arsitektur model**
14. **Menyiapkan callback**
15. **Training**
16. **Visualisasi learning curve**
17. **Prediksi pada data test**
18. **Menghitung metrik evaluasi**
19. **Classification report**
20. **Confusion matrix**
21. **Menyimpan hasil ringkas ke JSON**

### 5.1 Preprocessing Utama
Preprocessing yang digunakan secara umum pada kelima notebook:

- kolom `Unnamed` dibuang
- seluruh fitur diubah ke numerik
- label di-encode ke indeks berurutan untuk `SparseCategoricalCrossentropy`
- normalisasi memakai **row-wise z-score**
- hasil normalisasi di-clip ke rentang **[-5, 5]**
- data diubah ke bentuk sequence: **`(jumlah_sampel, 178, 1)`**

### 5.2 Metrik Evaluasi
Semua notebook mencatat metrik berikut:

- Accuracy
- Balanced Accuracy
- Precision Macro
- Recall Macro
- F1 Macro
- Precision Weighted
- Recall Weighted
- F1 Weighted
- ROC-AUC OVR Macro
- ROC-AUC OVR Weighted
- Test Loss

---

## 6. Arsitektur Tiap Model

## 6.1 RNN
File: [`Epileptic_EEG_RNN.ipynb`](./notebooks/Epileptic_EEG_RNN.ipynb)

### Konfigurasi utama
- recurrent layer: **Bidirectional SimpleRNN**
- unit recurrent: **128 -> 64**
- dropout recurrent: **0.00**
- attention: **MultiHeadAttention**
- jumlah head: **4**
- key dimension: **32**
- attention dropout: **0.10**
- dense layer: **128**
- dense dropout: **0.30**
- optimizer: **Adam**
- learning rate: **1e-3**
- clipnorm: **1.0**
- batch size: **64**
- epoch maksimum: **40**

### Struktur umum
- input `(178, 1)`
- 2 blok bidirectional SimpleRNN
- self-attention
- residual add
- layer normalization
- global average pooling + global max pooling
- dense + dropout
- output softmax 5 kelas

---

## 6.2 LSTM
File: [`Epileptic_EEG_LSTM.ipynb`](./notebooks/Epileptic_EEG_LSTM.ipynb)

### Konfigurasi utama
- recurrent layer: **LSTM satu arah**
- unit recurrent: **128 -> 64**
- dropout recurrent block: **0.10**
- attention: **MultiHeadAttention**
- jumlah head: **4**
- key dimension: **32**
- attention dropout: **0.10**
- dense layer: **128**
- dense dropout: **0.30**
- optimizer: **Adam**
- learning rate: **8e-4**
- batch size: **64**
- epoch maksimum: **40**

### Struktur umum
- input `(178, 1)`
- 2 blok LSTM
- self-attention
- residual add
- layer normalization
- global average pooling + global max pooling
- dense + dropout
- output softmax 5 kelas

---

## 6.3 BiLSTM
File: [`Epileptic_EEG_BiLSTM.ipynb`](./notebooks/Epileptic_EEG_BiLSTM.ipynb)

### Konfigurasi utama
- recurrent layer: **Bidirectional LSTM**
- unit recurrent: **128 -> 64**
- dropout recurrent block: **0.05**
- attention: **MultiHeadAttention**
- jumlah head: **4**
- key dimension: **32**
- attention dropout: **0.10**
- dense layer: **128**
- dense dropout: **0.25**
- optimizer: **Adam**
- learning rate: **1e-3**
- clipnorm: **1.0**
- batch size: **64**
- epoch maksimum: **45**

### Struktur umum
- input `(178, 1)`
- 2 blok bidirectional LSTM
- self-attention
- residual add
- layer normalization
- global average pooling + global max pooling
- dense + dropout
- output softmax 5 kelas

---

## 6.4 GRU
File: [`Epileptic_EEG_GRU.ipynb`](./notebooks/Epileptic_EEG_GRU.ipynb)

### Konfigurasi utama
- recurrent layer: **GRU satu arah**
- unit recurrent: **192 -> 96**
- dropout recurrent block: **0.00**
- attention: **MultiHeadAttention**
- jumlah head: **4**
- key dimension: **32**
- attention dropout: **0.10**
- dense layer: **128**
- dense dropout: **0.25**
- optimizer: **Adam**
- learning rate: **1e-3**
- clipnorm: **1.0**
- batch size: **64**
- epoch maksimum: **45**

### Struktur umum
- input `(178, 1)`
- 2 blok GRU
- self-attention
- residual add
- layer normalization
- global average pooling + global max pooling
- dense + dropout
- output softmax 5 kelas

---

## 6.5 BiGRU
File: [`Epileptic_EEG_BiGRU.ipynb`](./notebooks/Epileptic_EEG_BiGRU.ipynb)

### Konfigurasi utama
- recurrent layer: **Bidirectional GRU**
- unit recurrent: **160 -> 96**
- dropout recurrent block: **0.00**
- attention: **MultiHeadAttention**
- jumlah head: **4**
- key dimension: **32**
- attention dropout: **0.10**
- dense layer: **128**
- dense dropout: **0.25**
- optimizer: **Adam**
- learning rate: **1e-3**
- clipnorm: **1.0**
- batch size: **64**
- epoch maksimum: **45**

### Struktur umum
- input `(178, 1)`
- 2 blok bidirectional GRU
- self-attention
- residual add
- layer normalization
- global average pooling + global max pooling
- dense + dropout
- output softmax 5 kelas

---

## 7. Hasil Evaluasi

### 7.1 Metrik Utama

| Model | Accuracy | Balanced Accuracy | Precision Macro | Recall Macro | F1 Macro | Precision Weighted | Recall Weighted | F1 Weighted | ROC-AUC OVR Macro | ROC-AUC OVR Weighted |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| RNN | 0.773333 | 0.773333 | 0.774573 | 0.773333 | 0.773065 | 0.774573 | 0.773333 | 0.773065 | 0.955288 | 0.955288 |
| LSTM | 0.733333 | 0.733333 | 0.736335 | 0.733333 | 0.731821 | 0.736335 | 0.733333 | 0.731821 | 0.944499 | 0.944499 |
| BiLSTM | 0.703188 | 0.703188 | 0.705219 | 0.703188 | 0.696228 | 0.705219 | 0.703188 | 0.696228 | 0.938053 | 0.938053 |
| GRU | 0.749565 | 0.749565 | 0.754940 | 0.749565 | 0.749114 | 0.754940 | 0.749565 | 0.749114 | 0.946260 | 0.946260 |
| BiGRU | 0.756522 | 0.756522 | 0.759330 | 0.756522 | 0.757085 | 0.759330 | 0.756522 | 0.757085 | 0.949573 | 0.949573 |

#### Ranking berdasarkan accuracy
1. **RNN** - 0.773333  
2. **BiGRU** - 0.756522  
3. **GRU** - 0.749565  
4. **LSTM** - 0.733333  
5. **BiLSTM** - 0.703188  

#### Ranking berdasarkan F1 Macro
1. **RNN** - 0.773065  
2. **BiGRU** - 0.757085  
3. **GRU** - 0.749114  
4. **LSTM** - 0.731821  
5. **BiLSTM** - 0.696228  

### 7.2 Ringkasan Classification Report

#### Pola umum yang konsisten
- **Kelas 1** menjadi kelas yang paling mudah diprediksi di hampir semua model.
- **Kelas 2** dan **kelas 3** cenderung lebih sulit dipisahkan dibanding kelas lain.
- **Kelas 4** dan **kelas 5** biasanya berada di kategori menengah sampai cukup baik.
- Pada model terbaik, macro score dan weighted score relatif berdekatan, menandakan distribusi kelas yang seimbang membantu evaluasi menjadi lebih adil.

---

## 8. Interpretasi Hasil

### 8.1 Interpretasi Umum
Berdasarkan hasil final:

- seluruh model berhasil menembus accuracy di atas **0.70**
- **RNN** justru menjadi model terbaik pada eksperimen ini
- **BiGRU** dan **GRU** berada sangat dekat di bawah RNN
- **LSTM** turun cukup jauh dibanding RNN
- **BiLSTM** menjadi model dengan hasil terendah pada rekap akhir

Hal ini menunjukkan bahwa model yang secara teori lebih kompleks **tidak otomatis** menghasilkan performa terbaik pada dataset ini. Kesesuaian arsitektur, tuning, dan generalisasi ternyata lebih penting daripada kompleksitas semata.

### 8.2 Makna Tiap Metrik
- **Accuracy** dan **Balanced Accuracy** bernilai sama karena data test seimbang.
- **F1 Macro** menjadi indikator penting karena menghitung kualitas rata-rata antar kelas.
- **ROC-AUC OVR Macro** yang semuanya tinggi, berada di sekitar **0.938–0.955**, menunjukkan bahwa seluruh model masih mampu memisahkan kelas dengan cukup baik, walaupun keputusan akhir softmax-nya tidak selalu setinggi AUC-nya.
- Gap terbesar antar model lebih jelas terlihat pada **accuracy** dan **F1 Macro** daripada pada **ROC-AUC**.

### 8.3 Implikasi Eksperimen
- dataset ini tidak otomatis menguntungkan model gated seperti LSTM/BiLSTM
- **RNN baseline yang kuat** dapat mengungguli model yang lebih kompleks
- **bidirectional** tidak selalu membawa peningkatan, karena:
  - BiLSTM < LSTM
  - BiGRU > GRU, tetapi kenaikannya tidak terlalu besar

---

## 9. Interpretasi per File

### 9.1 RNN
File ini menjadi hasil terbaik. Arsitektur **Bidirectional SimpleRNN + attention** berhasil memberi kombinasi yang paling efektif pada dataset ini. Nilai **accuracy 0.773333**, **F1 Macro 0.773065**, dan **ROC-AUC 0.955288** menunjukkan performa paling tinggi dan paling seimbang di antara lima model.

### 9.2 LSTM
LSTM satu arah masih memberikan hasil cukup baik, tetapi tertinggal dari RNN. Nilai **accuracy 0.733333** dan **F1 Macro 0.731821** menunjukkan bahwa gate memory pada LSTM tidak otomatis menjadi keunggulan pada eksperimen ini.

### 9.3 BiLSTM
BiLSTM menjadi model dengan hasil terendah pada rekap akhir. Nilai **accuracy 0.703188** dan **F1 Macro 0.696228** menunjukkan bahwa penambahan arah sequence dua sisi dan kompleksitas recurrent tidak berhasil meningkatkan generalisasi.

### 9.4 GRU
GRU berada di posisi tengah atas dengan **accuracy 0.749565** dan **F1 Macro 0.749114**. Hasil ini menunjukkan GRU cukup kompetitif dan sedikit lebih baik dari LSTM, tetapi tetap belum melampaui RNN.

### 9.5 BiGRU
BiGRU menjadi model terbaik kedua. Nilai **accuracy 0.756522** dan **F1 Macro 0.757085** menunjukkan bahwa pada keluarga GRU, versi bidirectional memberi peningkatan yang nyata walaupun belum menyalip RNN.

---

## 10. Perbandingan Antar Model

### 10.1 Perbandingan Arsitektur
| Model | Tipe Recurrent | Arah Sequence | Unit Utama | Attention | Dense |
| :-- | :-- | :-- | :-- | :-- | :-- |
| RNN | SimpleRNN | Bidirectional | 128 -> 64 | MultiHeadAttention | 128 |
| LSTM | LSTM | Unidirectional | 128 -> 64 | MultiHeadAttention | 128 |
| BiLSTM | LSTM | Bidirectional | 128 -> 64 | MultiHeadAttention | 128 |
| GRU | GRU | Unidirectional | 192 -> 96 | MultiHeadAttention | 128 |
| BiGRU | GRU | Bidirectional | 160 -> 96 | MultiHeadAttention | 128 |

### 10.2 Perbandingan Hasil
| Model | Accuracy | F1 Macro | ROC-AUC OVR Macro |
| :-- | --: | --: | --: |
| RNN | 0.773333 | 0.773065 | 0.955288 |
| LSTM | 0.733333 | 0.731821 | 0.944499 |
| BiLSTM | 0.703188 | 0.696228 | 0.938053 |
| GRU | 0.749565 | 0.749114 | 0.946260 |
| BiGRU | 0.756522 | 0.757085 | 0.949573 |

### 10.3 Analisis Perbandingan
- **RNN** unggul pada semua metrik utama.
- **BiGRU** menjadi kompromi terbaik kedua antara kompleksitas dan performa.
- **GRU** sedikit di bawah BiGRU, tetapi masih lebih baik dari LSTM.
- **LSTM** kalah dari GRU dan BiGRU pada eksperimen ini.
- **BiLSTM** menjadi model yang paling lemah, jadi penambahan bidirectional pada LSTM tidak efektif pada konfigurasi yang dipakai.

---

## 11. Kelebihan dan Kekurangan Project

### Kelebihan
- dataset seimbang, sehingga accuracy dan balanced accuracy dapat dibaca dengan adil
- alur eksperimen konsisten di semua notebook
- preprocessing dan metrik evaluasi seragam
- sequence input jelas: 178 titik EEG per sampel
- seluruh model sudah memakai attention sehingga ada mekanisme penekanan bagian sinyal yang relevan
- sudah ada lima variasi recurrent model sehingga perbandingan cukup lengkap

### Kekurangan
- hasil menunjukkan model yang lebih kompleks belum tentu lebih baik, sehingga tuning tiap model masih bisa dikembangkan lagi
- gap antar model ada, tetapi tidak ekstrem, kecuali BiLSTM yang turun paling jelas

---

## 12. Kesimpulan

Project ini berhasil membangun lima model sequence untuk klasifikasi EEG **5 kelas** pada dataset **Epileptic Seizure Recognition** dengan pipeline yang konsisten.

Kesimpulan utama dari hasil final adalah:

- **RNN** menjadi model terbaik dengan:
  - **Accuracy**: 0.773333
  - **F1 Macro**: 0.773065
  - **ROC-AUC OVR Macro**: 0.955288

- **BiGRU** menjadi model terbaik kedua dengan:
  - **Accuracy**: 0.756522
  - **F1 Macro**: 0.757085

- **GRU** berada di posisi ketiga:
  - **Accuracy**: 0.749565
  - **F1 Macro**: 0.749114

- **LSTM** berada di posisi keempat:
  - **Accuracy**: 0.733333
  - **F1 Macro**: 0.731821

- **BiLSTM** menjadi model dengan hasil terendah:
  - **Accuracy**: 0.703188
  - **F1 Macro**: 0.696228

Dengan demikian, pada dataset dan konfigurasi eksperimen ini, **RNN sederhana berbasis bidirectional recurrent block dan attention justru menjadi pendekatan yang paling efektif**, sedangkan peningkatan kompleksitas ke LSTM/BiLSTM tidak memberi jaminan hasil lebih tinggi.

---

## 13. File Project

- [`Epileptic_EEG_RNN.ipynb`](./notebooks/Epileptic_EEG_RNN.ipynb)
- [`Epileptic_EEG_LSTM.ipynb`](./notebooks/Epileptic_EEG_LSTM.ipynb)
- [`Epileptic_EEG_BiLSTM.ipynb`](./notebooks/Epileptic_EEG_BiLSTM.ipynb)
- [`Epileptic_EEG_GRU.ipynb`](./notebooks/Epileptic_EEG_GRU.ipynb)
- [`Epileptic_EEG_BiGRU.ipynb`](./notebooks/Epileptic_EEG_BiGRU.ipynb)

---

## 14. Referensi

### Dataset
- Kaggle: `Epileptic Seizure Recognition`
- UCI Machine Learning Repository
- Bonn EEG Dataset

### Pustaka
1. Andrzejak, R. G., Lehnertz, K., Rieke, C., Mormann, F., David, P., & Elger, C. E.  
   **Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state**.  
   *Physical Review E*, 2001.

2. Almustafa, Khaled Mohamad.  
   **Classification of epileptic seizure dataset using different machine learning algorithms**.  
   *Informatics in Medicine Unlocked*, 2020.