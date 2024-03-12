# Laporan Proyek Machine Learning - Sandy Susanto

## Domain Proyek

Domain proyek ini akan membahas mengenai permasalahan dalam bidang ekonomi dan bisnis. Fokus pada proyek ini adalah untuk membuat prediksi harga mobil bekas pakai berdasarkan fitur-fitur dan dimiliki oleh kendaraan tersebut.

<img src="https://image.cnbcfm.com/api/v1/image/106961034-16342989482021-10-15t115305z_757712464_rc2baq9ww2kh_rtrmadp_0_usa-economy.jpeg?v=1671462564" alt="Car Market" title="Car Market Illustration" width="100%">

Pembelian kendaraan baru merupakan salah investasi finansial yang signifikan bagi sebagian besar institusi maupun individu. Dalam konteks ini, nilai jual kembali kendaraan, yang mencerminkan sebagian dari *return of investment*, menjadi faktor penting dalam proses pengambilan keputusan pembelian. Oleh karena itu, baik konsumen individu maupun perusahaan memiliki kepentingan dalam mengidentifikasi atribut-atribut kendaraan yang dapat mempertahankan nilai jualnya di pasar primer ataupun sekunder [[1]](https://link.springer.com/article/10.1057/jors.2016.16).

Menurut data yang dianalisis, ada beberapa hal penting yang sangat mempengaruhi berapa harga mobil bekas nantinya, seperti siapa produsen yang membuat mobilnya, tipe mobil, seberapa jauh mobil itu sudah berjalan, berapa umurnya, riwayat servisnya, bagaimana kondisi mobil secara fisik, seberapa banyak mobil itu terjual di pasaran, bagaimana layanan setelah mobil itu dibeli, dan bagaimana cara pemilik sebelumnya mengendarainya [[2]](https://www.semanticscholar.org/paper/New-Model-for-Residual-Value-Prediction-of-the-Used-Shen-Wang/af9ec65507e1f156d6c1817f92caa9547e5ba61a).

Reputasi yang baik dari penjual dapat sedikit meningkatkan harga yang ditawarkan dalam lelang *online* untuk mobil bekas, terutama jika sudah ada tawaran yang masuk dan peluang untuk terjual. Namun, pengaruhnya tidak sebesar faktor lain seperti kejelasan status kepemilikan atau waktu penutupan lelang. Menariknya, meskipun mobil dipajang dengan menarik dan banyak foto, hal tersebut tidak terlalu berpengaruh terhadap harga akhir atau kesempatan mobil tersebut untuk terjual. Ini cukup mengejutkan karena biasanya presentasi yang baik dianggap penting dalam penjualan *online*, terutama untuk barang yang beragam dan bernilai tinggi seperti mobil [[3]](https://link.springer.com/article/10.1007/s11293-006-9045-7).

Semua hal ini saling berkaitan dan bersama-sama menentukan harga jual kembali mobil. Jika kita tidak memperhatikan bagaimana semua hal ini saling berhubungan dan hanya menghitung pengaruhnya satu per satu, maka prediksi kita tentang harga mobil bekas tidak akan akurat.

Proyek ini akan melibatkan beberapa tahap, mulai dari pengumpulan dan pembersihan data, eksplorasi data untuk memahami fitur-fitur yang paling berpengaruh terhadap harga, pembangunan model prediksi, hingga evaluasi dan penyempurnaan model. Dengan pendekatan yang sistematis dan pemanfaatan teknologi terkini, proyek prediksi harga mobil ini berpotensi memberikan kontribusi terhadap efisiensi dan transparansi pasar mobil bekas.

# Business Understanding

## Problem Statements

Dari konteks yang telah disampaikan sebelumnya, teridentifikasi dua pertanyaan utama yang akan dijawab melalui proyek ini:
1. Apa langkah-langkah yang diperlukan dalam mempersiapkan data sebelum diaplikasikan dalam pengembangan model *machine learning*?
2. Bagaimana proses pembuatan model *machine learning* yang dapat memprediksi harga jual mobil bekas?

## Goals

Dari permasalahan yang telah diuraikan, tujuan yang ingin dicapai melalui proyek ini adalah sebagai berikut:
1. Menjalankan proses persiapan data secara menyeluruh untuk memastikan data siap digunakan dalam model *machine learning*.
2. Mengembangkan model *machine learning* yang efektif untuk menganalisis dan memprediksi harga jual mobil bekas dengan tingkat *error* yang minimal.

## Solution Statements

Dari uraian sebelumnya, beberapa langkah strategis telah diidentifikasi untuk mencapai target proyek, antara lain:
1. Proses persiapan data akan meliputi teknik-teknik berikut:
   - Pembagian dataset menjadi dua bagian, yaitu set pelatihan dan set pengujian dengan proporsi 90% untuk pelatihan dan 10% untuk pengujian, yang akan digunakan dalam pengembangan model *machine learning*.
   - Standardisasi nilai pada fitur numerik untuk menghindari deviasi yang signifikan pada data.
2. Dalam fase pembuatan model *machine learning*, tiga model yang menggunakan algoritma yang berbeda akan diuji. Algoritma yang akan diaplikasikan meliputi Algoritma K-Nearest Neighbor, Algoritma Random Forest, dan Algoritma Adaptive Boosting. Setelah evaluasi kinerja masing-masing model, algoritma yang memberikan akurasi prediksi terbaik akan dipilih sebagai model utama.
   - **Algoritma K-Nearest Neighbor (KNN)**  
Algoritma KNN merupakan metode klasifikasi yang tidak bergantung pada parameter tertentu dan berada di bawah kategori pembelajaran dengan pengawasan. Algoritma ini memanfaatkan jarak antar titik data untuk menentukan klasifikasi atau prediksi kelompok dari sebuah titik data. Algoritma ini termasuk metode yang populer dan mudah digunakan dalam *machine learning* untuk klasifikasi dan regresi. Walaupun algoritma KNN bisa digunakan untuk regresi maupun klasifikasi, umumnya lebih sering digunakan untuk klasifikasi. Algoritma ini beroperasi berdasarkan prinsip bahwa titik-titik data yang mirip biasanya berdekatan [[4]](https://www.ibm.com/topics/knn).
     Cara kerja algoritma K-Nearest Neighbor adalah sebagai berikut: [[5]](https://geospasialis.com/k-nearest-neighbor/)
     - Tentukan jumlah ( K ), yaitu tetangga terdekat yang akan digunakan untuk klasifikasi
     - Hitunglah jarak dari data yang akan diklasifikasikan ke semua titik dalam *dataset.*
     - Urutkan titik-titik tersebut berdasarkan jarak dari yang terkecil hingga terbesar dan pilih ( K ) titik dengan jarak terkecil.
     - Identifikasi kelas yang paling sering muncul di antara ( K ) titik tersebut.
     - Klasifikasikan data baru ke dalam kelas yang paling dominan berdasarkan tetangga terdekatnya.
       
     <br>
     <img src="https://user-images.githubusercontent.com/64983961/188507827-0f729ab6-61a5-4dbc-9be2-afa424f6c294.png" alt="Ilustrasi Algoritma K-Nearest Neighbor" title="Ilustrasi Algoritma K-Nearest Neighbor">
     
     Perhitungan jarak ke tetangga terdekat dapat dilakukan dengan menggunakan metrik sebagai berikut:
     - *Euclidean distance*
       $$d(x,y)=\sqrt{\sum_{i=1}^n (x_i-y_i)^2}$$
     - *Manhattan distance*
       $$d(x,y)=\sum_{i=1}^n |x_i-y_i|$$
     - *Hamming distance*
       $$d(x,y)=\frac{1}{n}\sum_{n=1}^{n=n} |x_i-y_i|$$
     - *Minkowski distance*
       $$d(x,y)=\left(\sum_{i=1}^n |x_i-y_i|^p\right)^\frac{1}{p}$$
     
     Kelebihan dari algoritma K-Nearest Neighbor adalah: 
     - Kesederhanaan dan mudah dipahami
     - Mudah diterapkan
     - Berlaku untuk klasifikasi dan regresi
     - Dapat digunakan pada jumlah kelas yang beragam
     - Tidak memerlukan proses training
     - Penambahan data baru yang mudah
     - Parameter minimal
     - Hasil pemodelan non-linear, cocok untuk data dengan batasan tidak linear
     
     Sedangkan kelemahan dari algoritma K-Nearest Neighbor adalah: 
     - Penentuan nilai \( K \) yang optimal diperlukan
     - Biaya komputasi yang besar
     - Proses yang lambat untuk dataset besar
     - Performa menurun pada data berdimensi tinggi
     - Sensitivitas terhadap data *noisy*, data yang hilang, dan *outlier*
     
   - **Algoritma Random Forest**  
     *Random forest* memperluas metode *bagging* dengan menggabungkan teknik *bagging* dan pemilihan fitur secara acak, menciptakan kumpulan *decision tree* yang independen satu sama lain. Pemilihan fitur acak ini menciptakan subset fitur yang berbeda-beda, yang menjamin bahwa setiap *decision tree* memiliki sedikit kesamaan. Ini membedakan *random forest* dari *decision tree* biasa, yang biasanya mempertimbangkan semua fitur saat membagi data, sementara *random forest* hanya menggunakan sebagian dari fitur-fitur tersebut [[6]](https://www.ibm.com/topics/random-forest#:~:text=Random%20forest%20is%20a%20commonly,Decision%20trees).
     
     <img src="https://user-images.githubusercontent.com/64983961/188504775-b7e4aa9b-f1cd-41ef-8a70-a977db8f3d60.png" alt="Ilustrasi Algoritma Random Forest" title="Ilustrasi Algoritma Random Forest">
     
     Setelah dilakukan pelatihan, prediksi untuk sampel yang tidak terlihat ($x'$) dapat dibuat dengan menghitung rata-rata prediksi dari semua pohon setiap individu model pada $x'$ [[7]](https://en.wikipedia.org/wiki/Random_forest#Bagging 'Random Forest - Bagging').
     $$\hat{f}=\frac{1}{B}\sum_{b=1}^{B} f_b(x^{'})$$
     
   - **Algoritma Adaptive Boosting**  
     Algoritma Adaptive Boosting atau biasanya disingkat AdaBoost merupakan algoritma yang melakukan pelatihan model secara berurutan dan dengan proses iteratif atau berulang. Data latih (*training data*) akan mempunyai bobot atau *weight* yang sama, kemudian model akan melakukan pemeriksaan. Bobot yang lebih tinggi akan dimasukkan ke dalam model yang salah, sehingga akan lanjut ke tahap selanjutnya. Proses iteratif tersebut akan terus berlanjut hingga model mencapai tingkat akurasi yang diinginkan.
     
     <img src="https://user-images.githubusercontent.com/64983961/188507801-30224052-cac2-4e99-9c67-2aec18de8e59.png" alt="Ilustrasi Algoritma Adaptive Boosting" title="Ilustrasi Algoritma Adaptive Boosting">
     
     Algoritma AdaBoost mengacu kepada metode tertentu untuk melakukan pelatihan *classifier* yang di-*boosted*. Pengklasifikasian tersebut adalah pengklasifikasian dalam bentuk, [[7]](https://en.wikipedia.org/wiki/AdaBoost#Training 'AdaBoost - Training')
     $$F_T(x)=\sum_{t=q}^{T}f_t(x)$$
     di mana setiap $F_T$ adalah *learner* yang lemah yang mengambil objek $x$ sebagai input dan mengembalikan nilai yang menunjukkan kelas objek. Demikian juga pada pengklasifikasi $T$ merupakan nilai positif jika sampel berada dalam kelas positif, dan negatif jika sebaliknya.

## Data Understanding

<img src="https://user-images.githubusercontent.com/64983961/188505289-4725df5e-9e3a-48b9-b261-e538fd0c6fb9.png" alt="Electric Power Consumption Kaggle Dataset" title="Electric Power Consumption Kaggle Dataset" width="100%">

Data yang digunakan dalam proyek ini adalah *dataset* yang diambil dari Kaggle Dataset [Electric Power Consumption](https://www.kaggle.com/datasets/fedesoriano/electric-power-consumption 'Time series analysis of power consumption') dengan kategori *dataset*, yaitu *Energy* dan *Electricity*. Dalam *dataset* tersebut terdapat sebuah *file* atau berkas dengan nama `powerconsumption.csv` yang berekstensi (*file format*) `.csv` atau [comma-separated values](https://en.wikipedia.org/wiki/Comma-separated_values 'Comma-separated values') berukuran 4,33 MB.

Dari *dataset* tersebut, masih perlu dilakukan penyesuaian hingga *dataset* dapat benar-benar digunakan. Beberapa penyesuaian tersebut, yaitu
- Menghapus kolom yang tidak digunakan dalam model, yaitu kolom `GeneralDiffuseFlows`, dan kolom `DiffuseFlows`.
  ```python
   epower.drop('GeneralDiffuseFlows', inplace=True, axis=1)
   epower.drop('DiffuseFlows',        inplace=True, axis=1)
   ```
- Mengubah format atau tipe data pada kolom `Datetime` dari format `string` menjadi `datetime`.
  ```python
  epower.Datetime = pd.to_datetime(epower.Datetime)
  ```
- Melakukan penguraian atau pemisahan kolom `Datetime` menjadi `Year`, `Month`, `Day`, `Hour`, dan `Minute`, lalu menghapus atau membuang (*drop*) kolom `Datetime`.
  ```python
  epower['Year']   = epower['Datetime'].apply(lambda date: date.year)
  epower['Month']  = epower['Datetime'].apply(lambda date: date.month)
  epower['Day']    = epower['Datetime'].apply(lambda date: date.day)
  epower['Hour']   = epower['Datetime'].apply(lambda date: date.hour)
  epower['Minute'] = epower['Datetime'].apply(lambda date: date.minute)
  ```

Kemudian dilakukan proses *Exploratory Data Analysis* (EDA) sebagai investigasi awal untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data dengan menggunakan teknik statistik dan representasi grafis atau visualisasi.

1. **Deskripsi Variabel**  
   Berikut adalah informasi mengenai variabel-variabel yang terdapat pada *dataset* *Electric Power Consumption* adalah sebagai berikut,
   
   <img src="https://user-images.githubusercontent.com/64983961/188505396-dda2d93c-9266-4c80-bb67-6f7ae4e6e8aa.png" alt="Deskripsi Variabel" title="Deskripsi Variabel">
   
   Dari gambar di atas dapat dilihat bahwa terdapat 52.416 baris data dan 10 kolom atribut atau fitur. Di antaranya adalah enam (6) atribut/variabel dengan tipe data `float64 non-null` dan lima (5) atribut/variabel dengan tipe data `int64 non-null` yang merupakan hasil penguraian dari variabel `Datetime` yang sebelumnya memiliki tipe data `datetime64[ns]`. Berikut adalah keterangan untuk masing-masing variabel,
   - `Temperature` : Temperatur
   - `Humidity`    : Kelembaban
   - `WindSpeed`   : Kecepatan angin
   - `PowerConsumption_Zone1` : Konsumsi daya listrik di stasiun Quads, Tétouan, Maroko
   - `PowerConsumption_Zone2` : Konsumsi daya listrik di stasiun Smir, Tétouan, Maroko
   - `PowerConsumption_Zone3` : Konsumsi daya listrik di stasiun Boussafou, Tétouan, Maroko
   - `Year`   : Tahun
   - `Month`  : Bulan
   - `Day`    : Tanggal
   - `Hour`   : Jam
   - `Minute` : Menit
   
2. **Deskripsi Statistik**  
   
   <img src="https://user-images.githubusercontent.com/64983961/188506144-7b2f5f52-be07-47ef-96a5-c65dbba6452a.png" alt="Deskripsi Statistik" title="Deskripsi Statistik">
   
3. **Menangani Missing Value**  
   
   <img src="https://user-images.githubusercontent.com/64983961/188506196-0c2457b4-123c-4e13-8954-5edb04c0ed17.png" alt="Menangani Missing Value" title="Menangani Missing Value">
   
   Berdasarkan gambar tersebut, tidak terdapat *missing value*.
   
4. **Menangani Outliers**  
   *Outliers* merupakan sampel data yang nilainya berada sangat jauh dari cakupan umum data utama yang dapat merusak hasil analisis data. Berikut adalah visualisasi *boxplot* untuk melakukan pengecekan keberadaan *outliers*.
   
   <img src="https://user-images.githubusercontent.com/64983961/188506260-f27e7d3d-e16e-42e7-a31e-8812f2aca7ea.png" alt="Menangani Outliers - Sebelum" title="Menangani Outliers - Sebelum">
     
   Berdasarkan gambar tersebut, terdapat *outliers* pada fitur `Temperature`, `Humidity`, `PowerConsumption_Zone2`, dan `PowerConsumption_Zone3`. Sehingga dilakukan proses pembersihan *outliers* dengan metode IQR (*Inter Quartile Range*).
   
   $$IQR=Q_3-Q_1$$
   
   Kemudian membuat batas bawah dan batas atas untuk mencakup *outliers* dengan menggunakan,
   
   $BatasBawah=Q_1-1.5*IQR$
   
   $BatasAtas=Q_3-1.5*IQR$
   
   
   Setelah dilakukan pembersihan *outliers*, dilakukan kembali visualisasi *outliers* untuk melakukan pengecekan kembali sebagai berikut,
   
   <img src="https://user-images.githubusercontent.com/64983961/188506280-e40fe70d-804c-457e-a6f3-7a89d425950d.png" alt="Menangani Outliers - Sesudah" title="Menangani Outliers - Sesudah">
   
   Dari gambar di atas dapat dilihat bahwa *outliers* telah berkurang. Meskipun *outliers* masih terdapat pada fitur `Temperatur`, `Humidity`, `PowerConsumption_Zone2`, dan `PowerConsumption_Zone3`, tetapi masih dalam batas aman.
   
5. **Univariate Analysis**  
   Melakukan proses analisis data *univariate* pada fitur-fitur numerik. Proses analisis ini menggunakan bantuan visualisasi histogram untuk masing-masing fitur numerik.
   
   <img src="https://user-images.githubusercontent.com/64983961/188506395-dae2772e-f61a-4ce2-b6ad-26acaa99c319.png" alt="Univariate Analysis" title="Univariate Analysis">
   
   Dari data histogram di atas diperoleh informasi, yaitu:
   - Temperatur menunjukkan *zero-skewed* atau histogram simetris/normal.
   - Lebih dari 50% data kecepatan angin mendekati nilai 0, dan sisanya berada pada nilai 5.
   - Konsumsi daya pada stasiun Quads (Zona 1) sebagian besar berada pada rentang daya 21.000 hingga 40.000, dan paling banyak berada pada daya sekitar 22.500.
   - Konsumsi daya pada stasiun Smir (Zona 2) sebagian besar berada pada rentang daya 12.500 hingga 27.500, dan paling banyak berada pada daya sekitar 16.500.
   - Konsumsi daya pada stasiun Boussafou (Zona 3) sebagian besar berada pada rentang daya 9.000 hingga 17.500, dan rentang 24.000 hingga 26.000, serta paling banyak berada pada daya sekitar 14.000.
   - Data diambil pada tahun 2017.
   
6. **Multivariate Analysis**  
   Melakukan visualisasi distribusi data pada fitur-fitur numerik dari *dataframe* `epower`. Visualisasi dilakukan dengan bantuan *library* `seaborn` `pairplot` menggunakan parameter `diag_kind`, yaitu `kde`, untuk melihat perkiraan distribusi probabilitas antar fitur numerik.
   
   <img src="https://user-images.githubusercontent.com/64983961/188507899-65cd3a60-d19c-47d6-8d7d-c7b1a57364ea.png" alt="Multivariate Analysis" title="Multivariate Analysis">
   
7. **Correlation Matrix with Heatmap**  
   Melakukan pengecekan korelasi antar fitur numerik dengan menggunakan visualisasi diagram *heatmap* *correlation matrix*.
   
   <img src="https://user-images.githubusercontent.com/64983961/188507977-c0120633-e8c2-44f6-9bc6-1b59347ebf86.png" alt="Correlation Matrix with Heatmap" title="Correlation Matrix with Heatmap">
   
   Dapat dilihat pada diagram *heatmap* di atas memiliki *range* atau rentang angka dari 1.0 hingga 0.4 dengan keterangan sebagai berikut,
   - Jika semakin mendekati 1, maka korelasi antar fitur numerik semakin kuat bernilai positif.
   - Jika semakin mendekati 0, maka korelasi antar fitur numerik semakin rendah.
   - Jika semakin mendekati -1, maka korelasi antar fitur numerik semakin kuat bernilai negatif.
   
   Jika korelasi bernilai positif, berarti nilai kedua fitur numerik cenderung meningkat bersama-sama.  
   
   Jika korelasi bernilai negatif, berarti nilai salah satu fitur numerik cenderung meningkat ketika nilai fitur numerik yang lain menurun.

8. **Analisis Korelasi Antar Fitur**  
   - Fitur `PowerConsumption_Zone1` memiliki korelasi yang cukup kuat dengan fitur `Temperature`, `Humidity`, dan `Hour`.
   - Fitur `PowerConsumption_Zone2` memiliki korelasi yang cukup kuat dengan fitur `Temperature`, `Humidity`, `Month`, dan `Hour`.
   - Fitur `PowerConsumption_Zone3` memiliki korelasi yang cukup kuat dengan fitur `Temperature`, `Humidity`, `Month`, dan `Hour`.
   
   Sehingga, fitur `WindSpeed`, `Year`, `Day`, dan `Minute` memiliki korelasi yang paling rendah dengan fitur `PowerConsumption_Zone1`, `PowerConsumption_Zone2`, dan `PowerConsumption_Zone3`. Dengan begitu, dapat dilakukan *drop* (menghapus) fitur-fitur tersebut.
   
   <img src="https://user-images.githubusercontent.com/64983961/188507983-6b44443c-d576-4ab3-8dcf-f7b9cf22ad99.png" alt="Analisis Korelasi Antar Fitur" title="Analisis Korelasi Antar Fitur">

## Data Preparation

Pada tahap persiapan data atau *data preparation* dilakukan berdasarkan penjelasan yang sudah dipaparkan pada bagian [Solution Statements](#solution-statements "Solution Statements"). Tahap ini penting dilakukan untuk mempersiapkan data sehingga dapat digunakan untuk melatih model *machine learning* dengan baik. Berikut adalah dua tahapan data preparation yang dilakukan, yaitu,

1. **Split Data**  
   Pembagian data dilakukan untuk memisahkan data keseluruhan menjadi dua (2) bagian, yaitu data latih (*training data*) dan data uji (*testing data*) dengan perbandingan rasio sebesar 90 : 10 menggunakan `train_test_split`.
   
    ```python
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.1, random_state=123)
    ```
    
   Kemudian diperoleh hasil pembagian data masing-masing, yaitu sebagai berikut,
   
    ```python
    Total seluruh sampel : 50931
    Total data train     : 45837
    Total data test      : 5094
    ```

2. **Standarisasi pada Fitur Numerik**  
   Standarisasi fitur numerik menggunakan `StandardScaler` untuk mencegah terjadinya penyimpangan nilai data yang cukup besar. Proses standarisasi tersebut dilakukan dengan mengurangkan nilai rata-rata, lalu membaginya dengan standar deviasi atau simpangan baku untuk menggeser distribusi. Proses standarisasi akan menghasilkan distribusi dengan nilai rata-rata menjadi 0, dan nilai standar deviasi menjadi 1.
   
    ```python
    scaler = StandardScaler()
    scaler.fit(xTrain[numericalFeatures])
    xTrain[numericalFeatures]  = scaler.transform(xTrain.loc[:, numericalFeatures])
    ```
   
   <img src="https://user-images.githubusercontent.com/64983961/188508047-08b6a450-aa39-4b2f-8b40-ef86e5adc216.png" alt="Standarisasi pada Fitur Numerik" title="Standarisasi pada Fitur Numerik">

    ```python
    xTrain[numericalFeatures].describe().round(4)
    ```
   
   <img src="https://user-images.githubusercontent.com/64983961/188508061-75a22910-be6c-485a-a2da-e5364d75e311.png" alt="Deskripsi Statistik setelah Standarisasi" title="Deskripsi Statistik setelah Standarisasi">

## Modelling

Setelah dilakukannya tahap *data preparation*, selanjutnya adalah melakukan tahap persiapan model terlebih dahulu sebelum mengembangkan model menggunakan algoritma yang telah ditentukan.

Tahap persiapan *dataframe* untuk analisis model menggunakan parameter `index`, yaitu train_mse dan test_mse, serta parameter `columns` yang merupakan algoritma yang akan digunakan untuk melakukan prediksi, yaitu algoritma K-Nearest Neighbor (KNN), Random Forest, dan Adaptive Boosting (AdaBoost).

```python
models = pd.DataFrame(
    index   = ['train_mse', 'test_mse'],
    columns = ['KNN', 'RandomForest', 'Boosting']
)
```

Kemudian terapkan ketiga algoritma ke dalam model tersebut.

1. **K-Nearest Neighbor (KNN) Algorithm**  
   Pada algoritma K-Nearest Neighbor digunakan parameter `n_neighbors` dengan nilai k = 10 tetangga dan `metric` bawaan, yaitu Euclidean.
   
   ```python
   knn = KNeighborsRegressor(n_neighbors=10)
   ```
   
   Kemudian akan dilakukan analisis prediksi *error* menggunakan *Mean Squared Error* (MSE) pada data latih (*training data*) dan data uji (*testing data*)
   
2. **Random Forest Algorithm**  
   Pada algoritma K-Nearest Neighbor digunakan parameter `n_estimator` dengan jumlah 50 *trees* (pohon), `max_depth` dengan nilai kedalaman atau panjang pohon 16, `random_state` dengan nilai 55, dan `n_jobs` yang bernilai -1 (pekerjaan dilakukan secara paralel).
   
   ```python
   rf = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
   ```
   
   Kemudian akan dilakukan analisis prediksi *error* menggunakan *Mean Squared Error* (MSE) pada data latih (*training data*) dan data uji (*testing data*)
   
3. **Adaptive Boosting (AdaBoost) Algorithm**  
   Pada algoritma K-Nearest Neighbor digunakan parameter `learning_rate` dengan nilai bobot setiap *regressor* adalah 0.05, dan `random_state` dengan nilai 55.
   
   ```python
   boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
   ```
   
   Kemudian akan dilakukan analisis prediksi *error* menggunakan *Mean Squared Error* (MSE) pada data latih (*training data*) dan data uji (*testing data*)

Ketiga model yang telah dibangun di atas, akan dilakukan pengujian kinerja untuk masing-masing model yang menggunakan algoritma K-Nearest Neighbor, algoritma Random Forest, dan algoritma Adaptive Boosting. Dari ketiga model tersebut akan diperoleh satu (1) model dengan hasil prediksi yang paling baik dan tingkat *error* yang paling rendah.

## Evaluation

Pada tahap evaluasi model, akan dilakukan pengujian untuk melihat algoritma mana yang memberikan hasil prediksi paling baik dan dengan tingkat *error* yang paling rendah. Sebelumnya, akan dilakukan proses standarisasi atau *scaling* pada fitur numerik data uji (*testing data*) agar nilai rata-rata (*mean*) bernilai 0, dan varians bernilai 1.

```python
xTest.loc[:, numericalFeatures] = scaler.transform(xTest[numericalFeatures])
```

Kemudian evaluasi dari ketiga model, yaitu algoritma K-Nearest Neighbor, Random Forest, dan Adaptive Boosting (AdaBoost) untuk masing-masing data latih (*training data*) dan data uji (*testing data*) dengan melihat tingkat *error*-nya menggunakan *Mean Squared Error* (MSE),

$$MSE=\frac{1}{N}\sum_{i=1}^{N} (y_i-y\\_pred_i)^2$$

di mana, nilai $N$ adalah jumlah *dataset*, nilai $y_i$ merupakan nilai sebenarnya, dan $y\\_pred$ yaitu nilai prediksinya.

Penggunaan metode metrik *Mean Squared Error* (MSE) memiliki kelebihan, yaitu cukup sederhana dan mudah dipahami dalam melakukan perhitungan. Meskipun begitu, terdapat kelemahan pada metrik ini, yaitu hasil akurasi prediksi yang kecil karena tidak dapat membandingan hasil peramalan tersebut dengan kenyataannya. []

```python
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN', 'RF', 'Boosting'])
modelDict = {'KNN': knn, 'RF': rf, 'Boosting': boosting}
for name, model in modelDict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=yTrain, y_pred=model.predict(xTrain))/1e3
    mse.loc[name, 'test']  = mean_squared_error(y_true=yTest,  y_pred=model.predict(xTest))/1e3
```

<img src="https://user-images.githubusercontent.com/64983961/188511052-986610cd-7ef4-4f79-a7c1-eef777d3a4f8.png" alt="Evaluation" title="Evaluation">

Dari data tabel tersebut dapat divisualisasikan pada grafik batang berikut.

<img src="https://user-images.githubusercontent.com/64983961/188511209-7f53ee96-f76b-4252-b87c-5e27b0fed0fb.png" alt="Evaluation Graph" title="Evaluation Graph">

Dari visualisasi diagram di atas dapat disimpulkan bahwa,
1. Model dengan algoritma Random Forest memberikan nilai *error* yang paling kecil, yaitu sebesar 583.1 pada *training error*, dan 1542.6 pada *testing error*.
2. Model dengan algoritma K-Nearest Neighbor memiliki tingkat *error* yang sedang di antara dua algoritma lainnya.
3. Model dengan algoritma Adaptive Boosting mengalami *error* yang paling beser dengan nilai *training error* sebesar 7602.37, dan nilai *testing error* sebesar 7436.21.

Selanjutnya adalah pengujian prediksi model dengan menggunakan beberapa nilai konsumsi daya (*power consumption*) dari data uji (*testing data*)

<img src="https://user-images.githubusercontent.com/64983961/188511397-7664a384-d933-4962-9569-f42cdbdbcf69.png" alt="Testing Model" title="Testing Model">

Dapat dilihat prediksi pada model dengan algoritma Random Forest memberikan hasi yang paling mendekati dengan nilai `y_true` jika dibandingkan dengan algoritma model yang lainnya.

Nilai `y_true` sebesar **28507** dan nilai prediksi `Random Forest` sebesar **28308**.

Kesimpulannya adalah model yang digunakan untuk melakukan prediksi penggunaan daya listrik (*electric power consumption*) menghasilkan **tingkat *error* yang paling rendah** dengan menggunakan **algoritma Random Forest** pada model yang telah dibangun.

---

## Referensi

[1] Kihm, A., Vance, C. (2016). "The determinants of equity transmission between the new and used car markets: a hedonic analysis." *J Oper Res Soc* 67, 1250–1258 . https://doi.org/10.1057/jors.2016.16

[2] Shen Gongqi, Wang Yansong, & Zhu Qiang. (2011). "New Model for Residual Value Prediction of the Used Car Based on BP Neural Network and Nonlinear Curve Fit." *2011 Third International Conference on Measuring Technology and Mechatronics Automation.* doi:10.1109/icmtma.2011.455 

[3] Andrews, T., & Benzing, C. (2006). "The Determinants of Price in Internet Auctions of Used Cars." *Atlantic Economic Journal,* 35(1), 43–57. doi:10.1007/s11293-006-9045-7 

[4] *What is the K-nearest neighbors algorithm?*. IBM. https://www.ibm.com/topics/knn 

[5] Hussein, S. (2022, February 23). *Mengenal K-nearest neighbor: Algoritma Populer untuk machine learning.* GEOSPASIALIS. https://geospasialis.com/k-nearest-neighbor/ 

[6] *What is Random Forest?*. IBM. https://www.ibm.com/topics/random-forest#:~:text=Random%20forest%20is%20a%20commonly,Decision%20trees 

[7] "AdaBoost", Retrieved from: https://en.wikipedia.org/wiki/AdaBoost#Training

[8] S. R. P. Nur Hidayatika, and S. N. W.P, "USULAN PENGGUNAAN METODE FORECASTING UNTUK PERMINTAAN KOPI ROBUSTA PADA PT. XYZ," *Industrial Engineering Online Journal*, vol. 4, no. 3, 2016, Retrieved from: https://ejournal3.undip.ac.id/index.php/ieoj/article/view/9002

[9] A. Salam and A. E. Hibaoui, "Comparison of Machine Learning Algorithms for the Power Consumption Prediction : - Case Study of Tetouan city –," *2018 6th International Renewable and Sustainable Energy Conference (IRSEC)*, 2018, pp. 1-5, doi: 10.1109/IRSEC.2018.8703007.
