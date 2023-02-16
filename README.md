# Elektrik Fiyat ve Tuketiminin Spectrogram Incelemesi ve ARIMA Görsellerinin Oluşturulması

Veriler 2018 yılından 2022 yılı sonuna kadar saatlik formatta çekilmiştir. Saatlik elektrik fiyatları incelenmiş ve görselleri aşağıda sunulmuştur. Sol altta bulunan grafik saatlik değerlin çizgi grafikte görselidir. Sağ altta bulunan grafik ise verinin spectrogram görselidir.

 ![Figure 2023-02-16 130847](https://user-images.githubusercontent.com/58287201/219335674-87e0efd5-6d44-4b96-a4fb-e5d6f0b059de.png)
 ![Figure 2023-02-16 130855](https://user-images.githubusercontent.com/58287201/219335676-c05530f0-8167-4a2b-8a77-03a440a2def1.png)

Verilerin saatlik değerinin günlük olacak şekilde ortalaması alınmış ve aynı görsel tiplerindeki resimleri aşağağıda sunulmuştur. 

 ![Figure 2023-02-16 131423](https://user-images.githubusercontent.com/58287201/219336374-990b0ddc-d1da-4d68-b46f-0d27e5b0c495.png)
 ![Figure 2023-02-16 131427](https://user-images.githubusercontent.com/58287201/219336379-40db15df-4aa3-4533-b6f4-6b76e971f7a0.png)
 
Verilerin saatlik değerinin aylık olacak şekilde ortalaması alınmış ve aynı görsel tiplerindeki resimleri aşağağıda sunulmuştur. 
 
![Figure 2023-02-16 131524](https://user-images.githubusercontent.com/58287201/219336798-29d7500a-53e3-4382-adc3-e9bcd5de65a7.png)
![Figure 2023-02-16 131600](https://user-images.githubusercontent.com/58287201/219336803-9ab72d93-233e-4ec5-b677-a5d55e0c791a.png)

Fiyatlara ait gösterimler tamamlanmıştır. Saatlik veride desen daha belirgin görünmektedir.

Elektrik tüketim verisi içinde aynı görseller oluşturulmuş ve görselleri incelenmiştir. Taranan makalelerde elektrik tüketim verisinin; ekonomik durum (GSYH), nüfus, mevsimsel parametreler gibi bağımsız değişkenlere sahip olduğu görülmüştür. 
Bu verinin daha zaman serisi olduğu düşünülmektedir ve taranan makalelerde Time Series, ML ve YSA yöntemlerinin tercih edildiği görülmektedir. 

![Figure 2023-02-16 131717](https://user-images.githubusercontent.com/58287201/219337680-2029f1bc-ee7e-421e-ab6c-d719dc42b76d.png)
![Figure 2023-02-16 131738](https://user-images.githubusercontent.com/58287201/219337670-2744b92e-4869-4a8e-b4b2-f59ebdc8f67d.png)

Verilerin saatlik değerinin günlük olacak şekilde ortalaması alınmış ve aynı görsel tiplerindeki resimleri aşağağıda sunulmuştur. 

![Figure 2023-02-16 132233](https://user-images.githubusercontent.com/58287201/219338258-e7be9be2-120f-409d-9580-f345376852e9.png)
![Figure 2023-02-16 132236](https://user-images.githubusercontent.com/58287201/219338266-d81306f5-5e65-4d8a-b5cb-3d29f3d9151d.png)

Verilerin saatlik değerinin aylık olacak şekilde ortalaması alınmış ve aynı görsel tiplerindeki resimleri aşağağıda sunulmuştur. 

![Figure 2023-02-16 132318](https://user-images.githubusercontent.com/58287201/219338437-3b6b8c6b-333e-43a3-9de3-7c9b5b71f8be.png)
![Figure 2023-02-16 132322](https://user-images.githubusercontent.com/58287201/219338431-0c5a07cf-5ad6-44fc-9f98-a539ed7ea8b9.png)

Tüketim verisi incelendiğinde verilerin gün tipi olduğu görülmektedir. Resmi ve dini bayramlardaki tüketim alışkanlıklarının değişmesi veride anomali oluşturmaktadır. 

ARIMA Görsellerinin Sunulması: 
Otokorelasyon görselleri saatlik veriler için aşağıda sunulmuştur. 

![Figure 2023-02-16 134020](https://user-images.githubusercontent.com/58287201/219346793-7525197b-14ce-4e9a-bf30-d064f3292144.png)
![Figure 2023-02-16 134016](https://user-images.githubusercontent.com/58287201/219346797-0a5fe29a-cdff-4101-8ca5-886a4fbd2ecc.png)

ACF görselleri aşağıdaki resimlerde mevcuttur. /

![Figure 2023-02-16 140054](https://user-images.githubusercontent.com/58287201/219347452-7a733526-0632-4f09-948c-72566235b5eb.png)
![Figure 2023-02-16 140049](https://user-images.githubusercontent.com/58287201/219347453-71bc840d-826a-4ddf-b4b7-cbdfa07430d6.png)
![Figure 2023-02-16 140103](https://user-images.githubusercontent.com/58287201/219347445-38b69a0d-39af-4e23-80c2-d35e6678a9c2.png)
![Figure 2023-02-16 140059](https://user-images.githubusercontent.com/58287201/219347449-e5389266-9995-4db7-8b5b-9d6e33dc8807.png)
![Figure 2023-02-16 140111](https://user-images.githubusercontent.com/58287201/219347440-3a360270-c559-43f3-8ef0-71ae8ef255a7.png)
![Figure 2023-02-16 140107](https://user-images.githubusercontent.com/58287201/219347442-f73dc798-aadf-4e08-accd-e54c70dbb1b2.png)

PACF görselleri aşağıdaki resimlerde mevcuttur. /

![Figure 2023-02-16 140342](https://user-images.githubusercontent.com/58287201/219347984-7b15b110-a55d-4807-93c8-97cec08a1b3f.png)
![Figure 2023-02-16 140345](https://user-images.githubusercontent.com/58287201/219347980-fd5bed5e-367a-4d77-8663-18b338fe5588.png)
![Figure 2023-02-16 140334](https://user-images.githubusercontent.com/58287201/219347987-e0a2311e-b174-49eb-8f2e-e31f4caa17ae.png)
![Figure 2023-02-16 140338](https://user-images.githubusercontent.com/58287201/219347986-7c313e4c-4bdc-4c73-8583-0b686ef1bc5a.png)
![Figure 2023-02-16 140325](https://user-images.githubusercontent.com/58287201/219347996-be0fbe28-90c3-4f6a-b8cb-5a530a013336.png)
![Figure 2023-02-16 140329](https://user-images.githubusercontent.com/58287201/219347991-79315031-23ca-4c25-bd5f-986999873370.png)


