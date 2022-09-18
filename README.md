## <div>SIBI Language Estimation</div>
Pada repositori ini berisi proyek SIBI Language (Bahasa Isyarat) Estimation dengan menggunakan Mediapipe Library. Pada proyek ini menggunakan data koordinat (x dan y) dari hasil pembacaaan *Hand Landmark*. Kemudian pada proyek ini menggunakan metode spatiotemporal untuk melakukan ekstraksi fitur dari obyek yang akan diamati. Untuk proses *Learning* dan *Prediction* menggunakan metode *Support Vector Machine* (SVM).

Pada proyek ini membutuhkan beberapa library tambahan guna menunjang proyek ini. Library tambahan tersebut antara lain:
1.   Mediapipe
2.   OpenCV
3.   Scikit-learn
4.   Pandas
5.   Pickle

## <div>Dokumentasi</div>

Untuk menginstall Mediapipe Library kamu bisa kunjungi website [Mediapipe](https://google.github.io/mediapipe/) untuk pembacaan cara install dan dokumentasi lengkap tentang Mediapipe Library.

## <div>Akuisisi Data</div>

Pada bagian ini akan melakukan pengambilan data koordinat dari *hand landmark* dengan menggunakan metode spatiotemporal. Spatiotemporal adalah proses analisa yang melibatkan ruang dan waktu. Dengan spatiotemporal sistem dapat mengetahui perubahan pose dari obyek yang diamati pada tiap frame. Pada proyek ini akan mengambil data sebanyak 210 data. Data tersebut didapat dari perhitungan sebagai berikut:

  spatio_data = jumlah keypoint yang diambil x 2 x jumlah frame yang akan diambil

Pada proyek ini jumlah keypoint yang diambil sebanyak 21 titik. Nilai '2' diatas diambil dari jumlah koordinat yang akan dihitung yaitu X dan Y dan jumlah frame yang akan diambil sebanyak 5 frame. Untuk penjelasan mengena spatiotemporal biasa dilihat referensi dibawah ini. 

![hand_landmarks (1)](https://user-images.githubusercontent.com/51139989/189511491-d3043ec0-f71c-4837-ab6c-ce9195cd2139.png)

Spatiotemporal reference: [documentation](https://drive.google.com/file/d/18DpgE5vpjp3kihcZ48wzgN0cfSk5AUNo/view?usp=sharing)

## <div>Demo</div>

Untuk ingin melihat video demonstrasi dari proyek ini bisa dilihat [disini](https://youtu.be/v6xm07VC5A0)

## <div>Saran</div>

Pada proyek ini akan memprediksi bahasa isyarat yang sedang dilakukan oleh obyek. Untuk sekarang sistem ini mampu memprediksi bahasa isyarat dari huruf A, B, C, D, E, Hallo, dan Hai. Sistem ini akan terus diperbarui guna bisa memprediksi bahasa isyarat sesuai dari kamus yang dikeluarkan oleh kemendikbud tentang Sistem Isyarat Bahasa Indonesia / [SIBI](https://drive.google.com/file/d/18DpgE5vpjp3kihcZ48wzgN0cfSk5AUNo/view?usp=sharing)

