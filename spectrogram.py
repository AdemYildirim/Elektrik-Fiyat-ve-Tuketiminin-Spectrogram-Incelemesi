# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:19:03 2023

@author: u56356
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seffaflik
from seffaflik.__ortak.__araclar import make_requests as __make_requests

Baslangic_Trh = "2018-01-01"  
Bitis_Trh = "2022-12-31"    

###verinin çekilmesi   

__first_part_url_market = "market/day-ahead-mcp"
particular_url = __first_part_url_market + "?startDate=" + Baslangic_Trh + "&endDate=" + Bitis_Trh
json = __make_requests(particular_url)
ptf = pd.DataFrame(json["body"]["dayAheadMCPList"])
ptf2 = ptf.reset_index() #saatlik veri bu tabloda
ptf2["price1"]   = ptf2["price"] + 1
ptf2["pricelog"] = np.log(ptf2["price1"])-np.log(1)

__second_part_url_market = "consumption/real-time-consumption"
particular_url = __second_part_url_market + "?startDate=" + Baslangic_Trh + "&endDate=" + Bitis_Trh
json = __make_requests(particular_url)
consumptions = pd.DataFrame(json["body"]["hourlyConsumptions"])
consumptions = consumptions.reset_index() #saatlik veri bu tabloda

#kolay pivotlansın diye yeni kolonlar eklendi 
ptf2["date-daily"]    = ptf2["date"].str[:10]
ptf2["date-mounth"]   = ptf2["date"].str[:7]
consumptions["date-daily"]   = consumptions["date"].str[:10]
consumptions["date-mounth"]   = consumptions["date"].str[:7]
#https://towardsdatascience.com/understanding-audio-data-fourier-transform-fft-spectrogram-and-speech-recognition-a4072d228520

#############################################################################################
############Spectrogram #####################################################################
#############################################################################################

############Saatlik Veriler #################################################################
#saatlik veriler 
plt.plot(ptf2["price"].values)
plt.title('Saatlik Fiyat')  
plt.show()
plt.plot(ptf2["pricelog"].values)
plt.title('Saatlik log(Fiyat)')  
plt.show()
plt.plot(consumptions["consumption"].values)
plt.title('Saatlik Tüketim')  
plt.show()

#saatlik spectrogramlar
fig = plt.figure()
gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
(ax1, ax2) = gs.subplots(sharex='col', sharey='row')
fig.suptitle('Saatlik Spectrogramlar \n(fiyat,tüketim)')
ax1.specgram(ptf2["pricelog"].values,Fs=len(ptf2["pricelog"])/24)
ax2.specgram(consumptions["consumption"].values,Fs=len(consumptions["consumption"])/24)

############Günlük Veriler ##################################################################
#günlük veriler
ptf3 = ptf2.pivot_table("pricelog","date-daily",aggfunc="mean")
plt.plot(ptf3["pricelog"].values)
plt.title('Günlük Ort Fiyat')  
plt.show()
consumptions2 = consumptions.pivot_table("consumption","date-daily",aggfunc="mean")
plt.plot(consumptions2["consumption"].values)
plt.title('Günlük Ort Tüketim')  
plt.show()

#günlük spectrogramlar
fig = plt.figure()
gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
(ax1, ax2) = gs.subplots(sharex='col', sharey='row')
fig.suptitle('Günlük Spectrogramlar \n(fiyat,tüketim)')
ax1.specgram(ptf3["pricelog"].values,Fs=1)
ax2.specgram(consumptions2["consumption"].values,Fs=1)

############Aylık Veriler ###################################################################
#Aylık Veriler
ptf4 = ptf2.pivot_table("pricelog","date-mounth",aggfunc="mean")
plt.plot(ptf4["pricelog"].values)
plt.title('Aylık Ort Fiyat')  
plt.show()
consumptions3 = consumptions.pivot_table("consumption","date-mounth",aggfunc="mean")
plt.plot(consumptions3["consumption"].values)
plt.title('Aylık Ort Tüketim')  
plt.show()

#aylık spectrogramlar
fig = plt.figure()
gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
(ax1, ax2) = gs.subplots(sharex='col', sharey='row')
fig.suptitle('Aylık Spectrogramlar \n(fiyat,tüketim)')
ax1.specgram(ptf4["pricelog"].values,Fs=1)
ax2.specgram(consumptions3["consumption"].values,Fs=1)


#############################################################################################
############ arıma      #####################################################################
#############################################################################################
#https://github.com/skar94376/Time_Series_Analysis__Basics/blob/main/TimeSeries.ipynb
#https://www.google.com/search?q=ar%C4%B1ma+acf+ve+pacf+uygulamaas%C4%B1+&biw=1171&bih=649&tbm=vid&sxsrf=AJOqlzV4j-ifqDfsZvREn_Vnv6Xiiv324Q%3A1675258319352&ei=z2naY8CSFZGCxc8PoIG8sAo&ved=0ahUKEwiAtqDIt_T8AhURQfEDHaAAD6YQ4dUDCA0&uact=5&oq=ar%C4%B1ma+acf+ve+pacf+uygulamaas%C4%B1+&gs_lcp=Cg1nd3Mtd2l6LXZpZGVvEAMyBwghEKABEAoyBwghEKABEAoyBwghEKABEAoyBwghEKABEAo6BAgjECc6CAghEBYQHhAdOgUIIRCgAVCkBViLJGC2JWgAcAB4AIABsQGIAZgPkgEEMC4xNJgBAKABAcABAQ&sclient=gws-wiz-video#fpstate=ive&vld=cid:c3ac6b3d,vid:36-10VVroPk
#https://dosya.kmu.edu.tr/sbe/userfiles/file/tezler/isletme/ebrukaya.pdf


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(ptf2["pricelog"])
plt.show()

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels as sm

###################### acf ler 
sm.graphics.tsaplots.plot_acf(ptf2["pricelog"],lags=7*24)
plt.title('saatlik fiyat acf')  
plt.show()
sm.graphics.tsaplots.plot_acf(consumptions["consumption"],lags=24*7)
plt.title('saatlik tüketim acf')  
plt.show()
sm.graphics.tsaplots.plot_acf(ptf3["pricelog"],lags=7)
plt.title('günlük fiyat acf')  
plt.show()
sm.graphics.tsaplots.plot_acf(consumptions2["consumption"],lags=100)
plt.title('günlük tüketim acf')  
plt.show()
sm.graphics.tsaplots.plot_acf(ptf4["pricelog"],lags=12)
plt.title('aylık fiyat acf')  
plt.show()
sm.graphics.tsaplots.plot_acf(consumptions3["consumption"],lags=12)
plt.title('aylık tüketim acf')  
plt.show()

###################### pacf ler 
sm.graphics.tsaplots.plot_pacf(ptf2["pricelog"],lags=7*24)
plt.title('saatlik fiyat pacf')  
plt.show()
sm.graphics.tsaplots.plot_pacf(consumptions["consumption"],lags=24*7)
plt.title('saatlik tüketim pacf')  
plt.show()
sm.graphics.tsaplots.plot_pacf(ptf3["pricelog"],lags=7)
plt.title('günlük fiyat pacf')  
plt.show()
sm.graphics.tsaplots.plot_pacf(consumptions2["consumption"],lags=100)
plt.title('günlük tüketim pacf')  
plt.show()
sm.graphics.tsaplots.plot_pacf(ptf4["pricelog"],lags=12)
plt.title('aylık fiyat pacf')  
plt.show()
sm.graphics.tsaplots.plot_pacf(consumptions3["consumption"],lags=12)
plt.title('aylık tüketim pacf')  
plt.show()

