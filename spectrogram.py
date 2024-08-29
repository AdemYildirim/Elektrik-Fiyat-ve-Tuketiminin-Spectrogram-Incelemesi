# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:19:03 2023

@author: ademyildirim
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seffaflik
from seffaflik.__ortak.__araclar import make_requests as __make_requests
import seaborn as sns
from matplotlib import pyplot
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split



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

#saatlik spectrogramlar for image process
fig = plt.figure()
gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
(ax1, ax2) = gs.subplots(sharex='col', sharey='row')
ax1.axis('off')
ax2.axis('off')
ax1.specgram(ptf2["pricelog"].values,Fs=len(ptf2["pricelog"])/24)
ax2.specgram(consumptions["consumption"].values,Fs=len(consumptions["consumption"])/24)


############Günlük Veriler ##################################################################
#günlük veriler
ptf3 = ptf2.pivot_table(["pricelog","price"], "date-daily",aggfunc="mean")
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

#günlük spectrogramlar for image process
fig = plt.figure()
gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
(ax1, ax2) = gs.subplots(sharex='col', sharey='row')
ax1.axis('off')
ax2.axis('off')
ax1.specgram(ptf3["pricelog"].values,Fs=1)
ax2.specgram(consumptions2["consumption"].values,Fs=1)

############Aylık Veriler ###################################################################
#Aylık Veriler
ptf4 = ptf2.pivot_table(["pricelog","price"],"date-mounth",aggfunc="mean")
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

#aylık spectrogramlar for image process
fig = plt.figure()
gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
(ax1, ax2) = gs.subplots(sharex='col', sharey='row')
ax1.axis('off')  
ax2.axis('off')
ax1.specgram(ptf4["pricelog"].values,Fs=1)
ax2.specgram(consumptions3["consumption"].values,Fs=1)


#############################################################################################
############ arıma      #####################################################################
#############################################################################################
#https://github.com/skar94376/Time_Series_Analysis__Basics/blob/main/TimeSeries.ipynb
#https://www.google.com/search?q=ar%C4%B1ma+acf+ve+pacf+uygulamaas%C4%B1+&biw=1171&bih=649&tbm=vid&sxsrf=AJOqlzV4j-ifqDfsZvREn_Vnv6Xiiv324Q%3A1675258319352&ei=z2naY8CSFZGCxc8PoIG8sAo&ved=0ahUKEwiAtqDIt_T8AhURQfEDHaAAD6YQ4dUDCA0&uact=5&oq=ar%C4%B1ma+acf+ve+pacf+uygulamaas%C4%B1+&gs_lcp=Cg1nd3Mtd2l6LXZpZGVvEAMyBwghEKABEAoyBwghEKABEAoyBwghEKABEAoyBwghEKABEAo6BAgjECc6CAghEBYQHhAdOgUIIRCgAVCkBViLJGC2JWgAcAB4AIABsQGIAZgPkgEEMC4xNJgBAKABAcABAQ&sclient=gws-wiz-video#fpstate=ive&vld=cid:c3ac6b3d,vid:36-10VVroPk
#https://dosya.kmu.edu.tr/sbe/userfiles/file/tezler/isletme/ebrukaya.pdf

###Bu kısım arıma için kullanılacak
'''
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

'''

###image filtering
#image convert to number 
#https://www.geeksforgeeks.org/how-to-convert-images-to-numpy-array/

#pip install Pillow
#data image proses ile rakamlara dönüştürüldü
####################################################
####################################################
#hourly
####################################################
####################################################

from PIL import Image
from numpy import asarray
img = Image.open(r"C:\Users\u56356\OneDrive - Statkraft AS\Desktop\hourly-price-consumption.png")
numpydata = asarray(img)
matrix1 = numpydata.all(axis=(2) , out=None)
a = numpydata[:,:,1]
d = pd.DataFrame(a).astype('int')
d1 = d.iloc[6:-7,7:-6]                                      #kenarlıklar yok edildi
y , x = np.shape(d)                                         #x y ölçüleri edinildi

#data left ve right olarak bülündü ve kolonların çarpımı alındı 
al = d1.iloc[:,0:round(x/2)-1]
almax = al.max().max()
al = al/almax
al = al.to_numpy()
aleft =np.prod(al, axis=(0))
aleft1 =np.sum(al, axis=(0))
aleft1log = np.log(aleft1)

ar = d1.iloc[:,-(round(x/2)-1):]            
armax = ar.max().max()
ar = ar/almax
ar = ar.to_numpy()
aright =np.prod(ar, axis=(0))
aright1 =np.sum(ar, axis=(0))
aright1log = np.log(aright1)

corhourly     = np.corrcoef(ar[:,0],al[:,-1])               #spectrogramın kesiştiği yerin korelasyonu 
corhourlyprod = np.corrcoef(aright,aleft)                   #satırların çarpımının korelasyonu
corhourlysum  = np.corrcoef(aright1,aleft1)                 #satırların toplamının korelasyonu

plt.plot(ar[:,0])                                           #spectrogramın kesiştiği yerin grafiği 
plt.plot(al[:,-1])
plt.title("houly spectrogram columns")
plt.show()

####################################################
####################################################
#daily
####################################################
####################################################

img = Image.open(r"C:\Users\u56356\OneDrive - Statkraft AS\Desktop\daily-price-consumption.png")
numpydata = asarray(img)
matrix1 = numpydata.all(axis=(2) , out=None)
a = numpydata[:,:,1]
d = pd.DataFrame(a).astype('int')
d1 = d.iloc[6:-7,7:-6]                                      #kenarlıklar yok edildi
y , x = np.shape(d)                                         #x y ölçüleri edinildi

#data left ve right olarak bülündü ve kolonların çarpımı alındı 
al = d1.iloc[:,0:round(x/2)-1]
almax = al.max().max()
al = al/almax
al = al.to_numpy()
aleft =np.prod(al, axis=(0))
aleft1 =np.sum(al, axis=(0))
aleft1log = np.log(aleft1)

ar = d1.iloc[:,-(round(x/2)-1):]            
armax = ar.max().max()
ar = ar/almax
ar = ar.to_numpy()
aright =np.prod(ar, axis=(0))
aright1 =np.sum(ar, axis=(0))
aright1log = np.log(aright1)

cordaily    = np.corrcoef(ar[:,0],al[:,-1])               #spectrogramın kesiştiği yerin korelasyonu 
cordailyprod = np.corrcoef(aright,aleft)                   #satırların çarpımının korelasyonu
cordailysum  = np.corrcoef(aright1,aleft1)                 #satırların toplamının korelasyonu

plt.plot(ar[:,0])                                           #spectrogramın kesiştiği yerin grafiği 
plt.plot(al[:,-1])
plt.title("Daily spectrogram columns")
plt.show()

####################################################
####################################################
#monthly
####################################################
####################################################

img = Image.open(r"C:\Users\u56356\OneDrive - Statkraft AS\Desktop\monthly-price-consumption.png")
numpydata = asarray(img)
matrix1 = numpydata.all(axis=(2) , out=None)
a = numpydata[:,:,1]
d = pd.DataFrame(a).astype('int')
d1 = d.iloc[6:-7,7:-6]                                      #kenarlıklar yok edildi
y , x = np.shape(d)                                         #x y ölçüleri edinildi

#data left ve right olarak bülündü ve kolonların çarpımı alındı 
al = d1.iloc[:,0:round(x/2)-1]
almax = al.max().max()
al = al/almax
al = al.to_numpy()
aleft =np.prod(al, axis=(0))
aleft1 =np.sum(al, axis=(0))
aleft1log = np.log(aleft1)

ar = d1.iloc[:,-(round(x/2)-1):]            
armax = ar.max().max()
ar = ar/almax
ar = ar.to_numpy()
aright =np.prod(ar, axis=(0))
aright1 =np.sum(ar, axis=(0))
aright1log = np.log(aright1)

cormonthly   = np.corrcoef(ar[:,0],al[:,-1])                 #spectrogramın kesiştiği yerin korelasyonu 
cormonthlyprod = np.corrcoef(aright,aleft)                   #satırların çarpımının korelasyonu
cormonthlysum  = np.corrcoef(aright1,aleft1)                 #satırların toplamının korelasyonu

plt.plot(ar[:,0])                                            #spectrogramın kesiştiği yerin grafiği 
plt.plot(al[:,-1])
plt.title("Monthly spectrogram columns")
plt.show()

####################################################
####################################################
#modeling
####################################################
####################################################

#pip install ployly
#pip install xgboost
#pip install lightgbm 
#conda install -c conda-forge lightgbm
#pip install catboost

###############################################################
#################### Hourly Forecasting ########################
###############################################################

x_train, x_test, y_train, y_test = train_test_split(consumptions["consumption"], 
                                                    ptf2["price"], 
                                                    test_size=0.1, 
                                                    random_state=42)
x_train = x_train.to_frame()
y_train = y_train.to_frame()

x_test = x_test.to_frame()
x_test = x_test.reset_index()
x_test = x_test.drop(['index'], axis=1)

y_test = y_test.to_frame()
y_test = y_test.reset_index()
y_test = y_test.drop(['index'], axis=1)
y_test = y_test +0.1

#ÇOKLU DOĞRUSAL REGRESYON - 
from sklearn.linear_model import LinearRegression
regressory = LinearRegression()
regressory. fit(x_train,y_train)
sonucregycdr = regressory .predict(x_test) 
sonucregylin = pd.DataFrame(sonucregycdr , columns=['sonucregylin']) 


#ÇOKLU POLİNOM REGRESYON - 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(x_train.values)
pol_reg = LinearRegression()
polinom = pol_reg.fit(X_poly, y_train.values)
sonucregypol0 = polinom.predict(poly_reg.fit_transform(x_test.values))
sonucregypol = pd.DataFrame(sonucregypol0 , columns=['sonucregypol']) 

#Decission Tree
from sklearn.tree import DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(random_state=0)
DTR_reg = dec_tree_reg.fit(x_train.values,y_train.values)
sonucregydtr0 =DTR_reg.predict(x_test.values)
sonucregydtr = pd.DataFrame(sonucregydtr0 , columns=['sonucregydtr']) 

#Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_regression = RandomForestRegressor(n_estimators= 200,
                                      random_state=0)
#gscvreg = GridSearchCV(y_train, x_train.values)
RFR_reg = rf_regression.fit(x_train.values,y_train.values)
sonucregyrfr0 = RFR_reg.predict(x_test.values)
sonucregyrfr = pd.DataFrame(sonucregyrfr0 , columns=['sonucregyrfr']) 

#adaboost 
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression
adaboost_regr = AdaBoostRegressor(random_state=0, n_estimators=100)
adaboostfit = adaboost_regr.fit(x_train.values,y_train.values)
sonucregyada0 =adaboostfit.predict(x_test.values)
sonucregyada = pd.DataFrame(sonucregyada0 , columns=['sonucregyada']) 

#xgboost
import xgboost as xgb
DM_train = xgb.DMatrix(data = x_train.values , label = y_train.values)
from xgboost import XGBRegressor
xgb = XGBRegressor().fit(x_train.values ,y_train.values)
sonucregyxgb0 = xgb.predict(x_test.values)
sonucregyxgb     = pd.DataFrame(sonucregyxgb0 , columns=['sonucregyxgb']).astype(float)

#lightgbm
from lightgbm import LGBMRegressor
lgbm = LGBMRegressor()
lgbm_model    = lgbm.fit(x_train.values ,y_train.values                         )
sonucregylgb0 = lgbm_model.predict(x_test.values)
sonucregylgb  = pd.DataFrame(sonucregylgb0 , columns=['sonucregylgb']) 

#catboost
from catboost import CatBoostRegressor
catb = CatBoostRegressor()
catb_model       = catb.fit(x_train.values ,y_train.values)
sonucregycat0    = catb_model.predict(x_test.values)
sonucregycat     = pd.DataFrame(sonucregycat0 , columns=['sonucregycat']) 

#SVR Regression
from sklearn.svm import SVR
support_reg_poly =SVR(kernel ='poly' , degree = 2)
SVR_poly = support_reg_poly.fit(x_train.values,y_train.values)
sonucregysvr0 =SVR_poly.predict(x_test.values)
sonucregysvr = pd.DataFrame(sonucregysvr0 , columns=['sonucregysvr']) 

def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))
    return mape

mapelin  = MAPE(y_test.values, regressory.predict(x_test))
mapepol  = MAPE(y_test.values, polinom.predict(poly_reg.fit_transform(x_test.values)))
mapedtr  = MAPE(y_test.values, DTR_reg.predict(x_test.values))
maperfr  = MAPE(y_test.values, RFR_reg.predict(x_test.values))
mapeada  = MAPE(y_test.values, adaboostfit.predict(x_test.values))
mapexgb  = MAPE(y_test.values, xgb.predict(x_test.values))
mapelgb  = MAPE(y_test.values, lgbm_model.predict(x_test.values))
mapecat  = MAPE(y_test.values, catb_model.predict(x_test.values))
mapesvr  = MAPE(y_test.values, SVR_poly.predict(x_test.values))

sonuclarhmape =[mapelin,
                mapepol,
                mapedtr,
                maperfr,
                mapeada,
                mapexgb,
                mapelgb,
                mapecat,
                mapesvr]
sonuclarhmape = pd.DataFrame(sonuclarhmape , index=None ,columns=["mape"]).round(2)
columnsname  =['lin','pol','dtr','rfr','ada','xgb','lgb','cat','svr']
columnsname  = pd.DataFrame(columnsname , index=None , columns=["method_name"])
sonuclarhmape = pd.concat([columnsname,sonuclarhmape],axis=1)               
 
###https://www.codecademy.com/article/seaborn-design-ii
ax= sns.barplot(x = 'method_name',
                y = 'mape',
                data = sonuclarhmape,
                palette = "GnBu_d")
ax.bar_label(ax.containers[0], fmt='%g')
plt.title("saatlik mape")
plt.show()


sonuclarh = pd.concat([y_test,
                       sonucregylin,
                       sonucregypol,
                       sonucregydtr,
                       sonucregyrfr,
                       sonucregyada,
                       sonucregyxgb,
                       sonucregylgb,
                       sonucregycat,
                       sonucregysvr],axis=1)

plt.plot(sonuclarh)
plt.title("saatlik tahmin değerleri")
plt.show()

sonuclarhc = sonuclarh.corr()
mask = np.triu(np.ones_like(sonuclarhc, 
                            dtype=bool))
sns.heatmap(sonuclarhc,
            cmap=sns.cubehelix_palette(as_cmap=True),
            mask = mask,
            vmin=0,
            vmax=1, 
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5},
            annot=True,
            annot_kws={"size":6}).set_title('houly corelation matrix')
plt.show()

###############################################################
#################### Daily Forecasting ########################
###############################################################


x_train, x_test, y_train, y_test = train_test_split(consumptions2["consumption"], 
                                                    ptf3["price"], 
                                                    test_size=0.1, 
                                                    random_state=42)
x_train = x_train.to_frame()
y_train = y_train.to_frame()

x_test = x_test.to_frame()
x_test = x_test.reset_index()
x_test = x_test.drop(['date-daily'], axis=1)

y_test = y_test.to_frame()
y_test = y_test.reset_index()
y_test = y_test.drop(['date-daily'], axis=1)

#ÇOKLU DOĞRUSAL REGRESYON - 
from sklearn.linear_model import LinearRegression
regressory = LinearRegression()
regressory. fit(x_train,y_train)
sonucregycdr = regressory .predict(x_test) 
sonucregylin = pd.DataFrame(sonucregycdr , columns=['sonucregylin']) 


#ÇOKLU POLİNOM REGRESYON - 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(x_train.values)
pol_reg = LinearRegression()
polinom = pol_reg.fit(X_poly, y_train.values)
sonucregypol0 = polinom.predict(poly_reg.fit_transform(x_test.values))
sonucregypol = pd.DataFrame(sonucregypol0 , columns=['sonucregypol']) 

#Decission Tree
from sklearn.tree import DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(random_state=0)
DTR_reg = dec_tree_reg.fit(x_train.values,y_train.values)
sonucregydtr0 =DTR_reg.predict(x_test.values)
sonucregydtr = pd.DataFrame(sonucregydtr0 , columns=['sonucregydtr']) 

#Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_regression = RandomForestRegressor(n_estimators= 200,
                                      random_state=0)
#gscvreg = GridSearchCV(y_train, x_train.values)
RFR_reg = rf_regression.fit(x_train.values,y_train.values)
sonucregyrfr0 = RFR_reg.predict(x_test.values)
sonucregyrfr = pd.DataFrame(sonucregyrfr0 , columns=['sonucregyrfr']) 

#adaboost 
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression
adaboost_regr = AdaBoostRegressor(random_state=0, n_estimators=100)
adaboostfit = adaboost_regr.fit(x_train.values,y_train.values)
sonucregyada0 =adaboostfit.predict(x_test.values)
sonucregyada = pd.DataFrame(sonucregyada0 , columns=['sonucregyada']) 

#xgboost
import xgboost as xgb
DM_train = xgb.DMatrix(data = x_train.values , label = y_train.values)
from xgboost import XGBRegressor
xgb = XGBRegressor().fit(x_train.values ,y_train.values)
sonucregyxgb0 = xgb.predict(x_test.values)
sonucregyxgb     = pd.DataFrame(sonucregyxgb0 , columns=['sonucregyxgb']).astype(float)

#lightgbm
from lightgbm import LGBMRegressor
lgbm = LGBMRegressor()
lgbm_model    = lgbm.fit(x_train.values ,y_train.values                         )
sonucregylgb0 = lgbm_model.predict(x_test.values)
sonucregylgb  = pd.DataFrame(sonucregylgb0 , columns=['sonucregylgb']) 

#catboost
from catboost import CatBoostRegressor
catb = CatBoostRegressor()
catb_model       = catb.fit(x_train.values ,y_train.values)
sonucregycat0    = catb_model.predict(x_test.values)
sonucregycat     = pd.DataFrame(sonucregycat0 , columns=['sonucregycat']) 

#SVR Regression
from sklearn.svm import SVR
support_reg_poly =SVR(kernel ='poly' , degree = 2)
SVR_poly = support_reg_poly.fit(x_train.values,y_train.values)
sonucregysvr0 =SVR_poly.predict(x_test.values)
sonucregysvr = pd.DataFrame(sonucregysvr0 , columns=['sonucregysvr']) 

def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))
    return mape

mapelin  = MAPE(y_test.values, regressory.predict(x_test))
mapepol  = MAPE(y_test.values, polinom.predict(poly_reg.fit_transform(x_test.values)))
mapedtr  = MAPE(y_test.values, DTR_reg.predict(x_test.values))
maperfr  = MAPE(y_test.values, RFR_reg.predict(x_test.values))
mapeada  = MAPE(y_test.values, adaboostfit.predict(x_test.values))
mapexgb  = MAPE(y_test.values, xgb.predict(x_test.values))
mapelgb  = MAPE(y_test.values, lgbm_model.predict(x_test.values))
mapecat  = MAPE(y_test.values, catb_model.predict(x_test.values))
mapesvr  = MAPE(y_test.values, SVR_poly.predict(x_test.values))

sonuclardmape =[mapelin,
                mapepol,
                mapedtr,
                maperfr,
                mapeada,
                mapexgb,
                mapelgb,
                mapecat,
                mapesvr]
sonuclardmape = pd.DataFrame(sonuclardmape , index=None ,columns=["mape"]).round(2)
columnsname  =['lin','pol','dtr','rfr','ada','xgb','lgb','cat','svr']
columnsname  = pd.DataFrame(columnsname , index=None , columns=["method_name"])
sonuclardmape = pd.concat([columnsname,sonuclardmape],axis=1)               
 
###https://www.codecademy.com/article/seaborn-design-ii
axm= sns.barplot(x = 'method_name',
                y = 'mape',
                data = sonuclardmape,
                palette = "GnBu_d")
axm.bar_label(ax.containers[0], fmt='%g')
plt.title("günlük mape")
plt.show()

sonuclard = pd.concat([y_test,
                       sonucregylin,
                       sonucregypol,
                       sonucregydtr,
                       sonucregyrfr,
                       sonucregyada,
                       sonucregyxgb,
                       sonucregylgb,
                       sonucregycat,
                       sonucregysvr],axis=1)

plt.plot(sonuclard)
plt.title("günlük tahmin değerleri")
plt.show()

sonuclardc = sonuclard.corr()
mask = np.triu(np.ones_like(sonuclardc, 
                            dtype=bool))
sns.heatmap(sonuclardc,
            cmap=sns.cubehelix_palette(as_cmap=True),
            mask = mask,
            vmin=0,
            vmax=1, 
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5},
            annot=True,
            annot_kws={"size":6}).set_title('daily corelation matrix')
plt.show()

###############################################################
#################### Month Forecasting ########################
###############################################################


x_train, x_test, y_train, y_test = train_test_split(consumptions3["consumption"], 
                                                    ptf4["price"], 
                                                    test_size=0.1, 
                                                    random_state=42)
x_train = x_train.to_frame()
y_train = y_train.to_frame()

x_test = x_test.to_frame()
x_test = x_test.reset_index()
x_test = x_test.drop(['date-mounth'], axis=1)

y_test = y_test.to_frame()
y_test = y_test.reset_index()
y_test = y_test.drop(['date-mounth'], axis=1)

#ÇOKLU DOĞRUSAL REGRESYON - 
from sklearn.linear_model import LinearRegression
regressory = LinearRegression()
regressory. fit(x_train,y_train)
sonucregycdr = regressory .predict(x_test) 
sonucregylin = pd.DataFrame(sonucregycdr , columns=['sonucregylin']) 


#ÇOKLU POLİNOM REGRESYON - 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(x_train.values)
pol_reg = LinearRegression()
polinom = pol_reg.fit(X_poly, y_train.values)
sonucregypol0 = polinom.predict(poly_reg.fit_transform(x_test.values))
sonucregypol = pd.DataFrame(sonucregypol0 , columns=['sonucregypol']) 

#Decission Tree
from sklearn.tree import DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(random_state=0)
DTR_reg = dec_tree_reg.fit(x_train.values,y_train.values)
sonucregydtr0 =DTR_reg.predict(x_test.values)
sonucregydtr = pd.DataFrame(sonucregydtr0 , columns=['sonucregydtr']) 

#Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_regression = RandomForestRegressor(n_estimators= 200,
                                      random_state=0)
#gscvreg = GridSearchCV(y_train, x_train.values)
RFR_reg = rf_regression.fit(x_train.values,y_train.values)
sonucregyrfr0 = RFR_reg.predict(x_test.values)
sonucregyrfr = pd.DataFrame(sonucregyrfr0 , columns=['sonucregyrfr']) 

#adaboost 
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression
adaboost_regr = AdaBoostRegressor(random_state=0, n_estimators=100)
adaboostfit = adaboost_regr.fit(x_train.values,y_train.values)
sonucregyada0 =adaboostfit.predict(x_test.values)
sonucregyada = pd.DataFrame(sonucregyada0 , columns=['sonucregyada']) 

#xgboost
import xgboost as xgb
DM_train = xgb.DMatrix(data = x_train.values , label = y_train.values)
from xgboost import XGBRegressor
xgb = XGBRegressor().fit(x_train.values ,y_train.values)
sonucregyxgb0 = xgb.predict(x_test.values)
sonucregyxgb     = pd.DataFrame(sonucregyxgb0 , columns=['sonucregyxgb']).astype(float)

#lightgbm
from lightgbm import LGBMRegressor
lgbm = LGBMRegressor()
lgbm_model    = lgbm.fit(x_train.values ,y_train.values                         )
sonucregylgb0 = lgbm_model.predict(x_test.values)
sonucregylgb  = pd.DataFrame(sonucregylgb0 , columns=['sonucregylgb']) 

#catboost
from catboost import CatBoostRegressor
catb = CatBoostRegressor()
catb_model       = catb.fit(x_train.values ,y_train.values)
sonucregycat0    = catb_model.predict(x_test.values)
sonucregycat     = pd.DataFrame(sonucregycat0 , columns=['sonucregycat']) 

#SVR Regression
from sklearn.svm import SVR
support_reg_poly =SVR(kernel ='poly' , degree = 2)
SVR_poly = support_reg_poly.fit(x_train.values,y_train.values)
sonucregysvr0 =SVR_poly.predict(x_test.values)
sonucregysvr = pd.DataFrame(sonucregysvr0 , columns=['sonucregysvr']) 

def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))
    return mape

mapelin  = MAPE(y_test.values, regressory.predict(x_test))
mapepol  = MAPE(y_test.values, polinom.predict(poly_reg.fit_transform(x_test.values)))
mapedtr  = MAPE(y_test.values, DTR_reg.predict(x_test.values))
maperfr  = MAPE(y_test.values, RFR_reg.predict(x_test.values))
mapeada  = MAPE(y_test.values, adaboostfit.predict(x_test.values))
mapexgb  = MAPE(y_test.values, xgb.predict(x_test.values))
mapelgb  = MAPE(y_test.values, lgbm_model.predict(x_test.values))
mapecat  = MAPE(y_test.values, catb_model.predict(x_test.values))
mapesvr  = MAPE(y_test.values, SVR_poly.predict(x_test.values))

sonuclarmmape =[mapelin,
                mapepol,
                mapedtr,
                maperfr,
                mapeada,
                mapexgb,
                mapelgb,
                mapecat,
                mapesvr]
sonuclarmmape = pd.DataFrame(sonuclarmmape , index=None ,columns=["mape"]).round(2)
columnsname  =['lin','pol','dtr','rfr','ada','xgb','lgb','cat','svr']
columnsname  = pd.DataFrame(columnsname , index=None , columns=["method_name"])
sonuclarmmape = pd.concat([columnsname,sonuclarmmape],axis=1)               
 
###https://www.codecademy.com/article/seaborn-design-ii
axm= sns.barplot(x = 'method_name',
                y = 'mape',
                data = sonuclarmmape,
                palette = "GnBu_d")
axm.bar_label(ax.containers[0], fmt='%g')
plt.title("aylık mape")
plt.show()

sonuclarm = pd.concat([y_test,
                       sonucregylin,
                       sonucregypol,
                       sonucregydtr,
                       sonucregyrfr,
                       sonucregyada,
                       sonucregyxgb,
                       sonucregylgb,
                       sonucregycat,
                       sonucregysvr],axis=1)

plt.plot(sonuclarm)
plt.title("aylık tahmin değerleri")
plt.show()

sonuclarmc = sonuclarm.corr()
mask = np.triu(np.ones_like(sonuclarmc, 
                            dtype=bool))
sns.heatmap(sonuclarmc,
            cmap=sns.cubehelix_palette(as_cmap=True),
            mask = mask,
            vmin=0,
            vmax=1, 
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5},
            annot=True,
            annot_kws={"size":6}).set_title('mounth corelation matrix')
plt.show()
