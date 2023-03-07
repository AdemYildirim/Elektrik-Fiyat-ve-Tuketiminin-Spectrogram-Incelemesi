# Elektrik Fiyat ve Tuketiminin Spectrogram Incelemesi ve ARIMA Görsellerinin Oluşturulması

Veriler 2018 yılından 2022 yılı sonuna kadar saatlik formatta çekilmiştir. Saatlik elektrik fiyatları incelenmiş ve görselleri aşağıda sunulmuştur. Sol altta bulunan grafik saatlik değerlin çizgi grafikte görselidir. Sağ altta bulunan grafik ise saatlik elektrik fiyatlarının logaritmaları alınmış ve görselleştirilmiştir. Verinin 10 tabanında logatitması alınmış ve negatif-tanımsız logaritma değerlerini yok etmek için saatlik fiyatlara 11 tam sayısı eklenmiş sonrasında 10 tabanında logaritması  alınmış son aşamada da 10 tabanında log(11), logartiması alınan kolondan çıkarılmıştır. Aşağıdaki formül seti uygulanmıştır.  

ptf["pricekukla"] = ptf2["price"] + 11

ptf["pricelog"] = np.log10(ptf2["pricekukla"])-np.log10(11)

 ![Figure 2023-02-16 130847](https://user-images.githubusercontent.com/58287201/219335674-87e0efd5-6d44-4b96-a4fb-e5d6f0b059de.png) 
 ![Figure 2023-03-08 013120](https://user-images.githubusercontent.com/58287201/223568905-06cfd701-b7c2-42e5-8f94-feded70c0285.png)

Saatlik elektrik tüketim verisi aşağıdaki görselde mevcuttur. Taranan makalelerde elektrik tüketim verisinin; ekonomik durum (GSYH), nüfus, mevsimsel parametreler gibi bağımsız değişkenlere sahip olduğu görülmüştür. Bu verinin daha zaman serisi olduğu düşünülmektedir ve taranan makalelerde Time Series, ML ve YSA yöntemlerinin tercih edildiği görülmektedir. 

![Figure 2023-02-16 131717](https://user-images.githubusercontent.com/58287201/219337680-2029f1bc-ee7e-421e-ab6c-d719dc42b76d.png)

Saatlik verilere ait birleştirlmiş spectrogram aşağıdaki görselde mevcuttur. Verideki renk geçişlerinin birbiri ile uyumlu olduğu görülmektedir. 

![Figure 2023-03-08 013843](https://user-images.githubusercontent.com/58287201/223570128-f3dda14f-58e9-45f7-b5f2-ce8be5e2f916.png)

Verilerin saatlik değerinin günlük olacak şekilde ortalaması alınmış ve aynı görsel tiplerindeki resimleri aşağağıda sunulmuştur. 

![Figure 2023-03-08 014628](https://user-images.githubusercontent.com/58287201/223571493-55d8faf7-bdbc-4118-9ff5-f2637a4c888c.png)
![Figure 2023-02-16 132233](https://user-images.githubusercontent.com/58287201/219338258-e7be9be2-120f-409d-9580-f345376852e9.png)

Günlük ortalama değerlerin spectrogramları aşağıdaki görselde sunulmuştur. 

![Figure 2023-03-08 015031](https://user-images.githubusercontent.com/58287201/223572048-11cece32-8e8b-486e-b95a-2af34c792db4.png)

Verilerin saatlik değerinin aylık olacak şekilde ortalaması alınmış ve aynı görsel tiplerindeki resimleri aşağağıda sunulmuştur. 

![Figure 2023-03-08 015232](https://user-images.githubusercontent.com/58287201/223572584-03dc8c2f-a9db-4589-ae59-64c5b03b2c51.png)
![Figure 2023-03-08 015223](https://user-images.githubusercontent.com/58287201/223572582-2a21b1ac-5cec-44ee-be31-03d6aabd2d07.png)

Aylık ortalama değerlerin spectrogramları aşağıdaki görselde sunulmuştur. 

![Figure 2023-03-08 015450](https://user-images.githubusercontent.com/58287201/223572788-292863c2-1b16-487d-9b04-3c4f5304386e.png)


ARIMA Görsellerinin Sunulması: 
Otokorelasyon görselleri saatlik veriler için aşağıda sunulmuştur. 

![Figure 2023-02-16 134020](https://user-images.githubusercontent.com/58287201/219346793-7525197b-14ce-4e9a-bf30-d064f3292144.png)
![Figure 2023-03-08 015947](https://user-images.githubusercontent.com/58287201/223573586-4acae87d-1333-4b34-ae08-619d21defff0.png)


ACF görselleri aşağıdaki resimlerde mevcuttur. /

![Figure 2023-02-16 140054](https://user-images.githubusercontent.com/58287201/219347452-7a733526-0632-4f09-948c-72566235b5eb.png)
![Figure 2023-03-08 020124](https://user-images.githubusercontent.com/58287201/223573952-797a11b7-adff-4eab-be78-8701b6553044.png)
![Figure 2023-02-16 140103](https://user-images.githubusercontent.com/58287201/219347445-38b69a0d-39af-4e23-80c2-d35e6678a9c2.png)
![Figure 2023-03-08 020239](https://user-images.githubusercontent.com/58287201/223573948-fc414d49-b95e-4779-94ec-f6dd84c54b0b.png)
![Figure 2023-02-16 140111](https://user-images.githubusercontent.com/58287201/219347440-3a360270-c559-43f3-8ef0-71ae8ef255a7.png)
![Figure 2023-03-08 020248](https://user-images.githubusercontent.com/58287201/223573940-fb621d6f-e0b6-48ba-988d-e7a73a9b678a.png)

PACF görselleri aşağıdaki resimlerde mevcuttur. /

![Figure 2023-02-16 140342](https://user-images.githubusercontent.com/58287201/219347984-7b15b110-a55d-4807-93c8-97cec08a1b3f.png)
![Figure 2023-03-08 020437](https://user-images.githubusercontent.com/58287201/223574309-2b6c5e1d-1a7a-4ff9-a795-01aaecddd759.png)
![Figure 2023-02-16 140334](https://user-images.githubusercontent.com/58287201/219347987-e0a2311e-b174-49eb-8f2e-e31f4caa17ae.png)
![Figure 2023-03-08 020452](https://user-images.githubusercontent.com/58287201/223574300-e05b6eb4-109f-41fe-939b-0af29c1bd4a1.png)
![Figure 2023-02-16 140325](https://user-images.githubusercontent.com/58287201/219347996-be0fbe28-90c3-4f6a-b8cb-5a530a013336.png)
![Figure 2023-03-08 020500](https://user-images.githubusercontent.com/58287201/223574282-d2b79e6c-44cf-4f34-af37-17c1d913d2be.png)


