## PYTHON EĞİTİMİ BİTİRME PROJESİ - BİROL GÜNGÖR ##

## MAKİNE ÖĞRENMESİ UYGULAMASI ##

## STREAMLIT KULLANILARAK GİRİLEN KONFİGURASYONA GÖRE MAKİNE ÖĞRENMESİ İLE LAPTOP FİYAT TAHMİNİ ##

## * VERİ: https://www.kaggle.com/muhammetvarl/laptop-pr ##

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.neighbors import KNeighborsRegressor

## Veri okunuyor.
laptop=pd.read_csv("laptop_price.csv",encoding='latin-1')

## RAM değeri int'e çevrilip yeni bir sütuna ekleniyor.
laptop['RAM_int'] = laptop['Ram'].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(int)

## En yüksek tahmin skorunu bulabilmek için tüm skorların tutulacağı liste oluşturuluyor.
scorelist=[]

## En yüksek skorlu modelin kontrolü maksatlı model adı ve skorunu tutacak olan kütüphane oluşturuluyor.
scores={}

################################# S T R E A M L I T #################################
## Tab yazısı hazırlanıyor.
st.set_page_config(page_title="Birol'un Python Projesi",page_icon="💥")

st.header(":male-teacher: **PYTHON EĞİTİMİ PROJESİ**")

## Proje konusu yazdırılıyor.
st.markdown("<style> .font {font-size:25px ; fontStyle='bold'; font-family: 'Cooper Black'; color: #000000;}</style> ", unsafe_allow_html=True)
st.markdown('<p class="font">-----Laptop Fiyat Tahmini-----</p>', unsafe_allow_html=True)

## Dip Not hazırlanıyor.
footer="""<style>
.footer {
position: fixed;
style:bold;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by Birol Güngör</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

## Kullanıcı girdileri alınıyor.
st.sidebar.subheader(":point_down: Seçimlerinizi yapınız")

company=st.sidebar.selectbox("Company",laptop.Company.unique())

urunler=laptop.loc[laptop["Company"]==company]
urunsecim=list(urunler["Product"].unique())

product=st.sidebar.selectbox("Product",urunsecim)
gpu=st.sidebar.selectbox("Graphics Processing Unit",laptop.Gpu.unique())
cpu=st.sidebar.selectbox("CPU",laptop.Cpu.unique())
ram=st.sidebar.selectbox("RAM",laptop.Ram.unique())
ram=ram.rstrip("GB")
ram=int(ram)
size=st.sidebar.selectbox("Screen Size (Inches)",laptop.Inches.unique())

################################# P R E P R O C E S S I N G #################################
laptop= laptop.drop(columns=["laptop_ID","TypeName","Weight","Ram"],axis=1)
laptop= laptop.append({"Company":company,"Product":product,"Inches":size,"Gpu":gpu,"Cpu":cpu,"RAM_int":ram,"Price_euros":0},ignore_index=True)
laptop["ScreenResolution"]=laptop["ScreenResolution"].fillna("")
laptop["Memory"]=laptop["Memory"].fillna("")
laptop["OpSys"]=laptop["OpSys"].fillna("")

## Dummy değişkenler oluşturuluyor.
laptopnew = pd.get_dummies(laptop, columns=["Company", "Product", "ScreenResolution","Cpu","Memory","Gpu","OpSys"], drop_first=True)

y = laptopnew[["Price_euros"]]
x = laptopnew.drop("Price_euros", axis=1)

## Train/Test
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,random_state=26)

## Normalizer
norm = Normalizer()
normx = norm.fit_transform(x)

## Hesaplamanın başlatılabilmesi için basılması gereken buton oluşturuluyor.
hesapla=st.sidebar.button("Hesapla")
if hesapla:

################################### R E G R E S Y O N L A R ###################################
## Tüm regresyonlarda en uygun parametrelerin seçilmesine özen gösterilmiştir!!!

##################### L I N E A R #####################
      reg = LinearRegression()

      model_lr = reg.fit(x, y)
      score_lr = model_lr.score(x,y)
      scorelist.append(score_lr)
      scores['model_lr']=score_lr

      ## Train & Test
      model_lr_trte = reg.fit(x_train, y_train)
      score_lr_trte = model_lr_trte.score(x_test, y_test)
      scorelist.append(score_lr_trte)
      scores['model_lr_trte']=score_lr_trte

      ## Normalized
      model_lr_norm = reg.fit(normx, y)
      score_lr_norm = model_lr_norm.score(normx, y)
      scorelist.append(score_lr_norm)
      scores['model_lr_norm']=score_lr_norm

##################### R I D G E #####################
      ridge = Ridge(alpha=0.1)

      model_rd = ridge.fit(x, y)
      score_rd = model_rd.score(x, y)
      scorelist.append(score_rd)
      scores['model_rd']=score_rd

      ## Train & Test
      model_rd_trte = ridge.fit(x_train, y_train)
      score_rd_trte = model_rd_trte.score(x_test, y_test)
      scorelist.append(score_rd_trte)
      scores['model_rd_trte']=score_rd_trte

      ## Normalized
      model_rd_norm = ridge.fit(normx, y)
      score_rd_norm = model_rd_norm.score(normx, y)
      scorelist.append(score_rd_norm)
      scores['model_rd_norm']=score_rd_norm

##################### L A S S O #####################
      lasso = Lasso()

      model_ls = lasso.fit(x, y)
      score_ls = model_ls.score(x, y)
      scorelist.append(score_ls)
      scores['model_ls']=score_ls

      ## Train & Test
      model_ls_trte = lasso.fit(x_train, y_train)
      score_ls_trte = model_ls_trte.score(x_test, y_test)
      scorelist.append(score_ls_trte)
      scores['model_ls_trte']=score_ls_trte

      ## Normalized
      model_ls_norm= lasso.fit(normx, y)
      score_ls_norm = model_ls_norm.score(normx, y)
      scorelist.append(score_ls_norm)
      scores['model_ls_norm']=score_ls_norm

##################### E L A S T I C N E T #####################
      elastic = ElasticNet()

      model_el = elastic.fit(x, y)
      score_el = model_el.score(x, y)
      scorelist.append(score_el)
      scores['model_el']=score_el

      ## Train & Test
      model_el_trte = elastic.fit(x_train, y_train)
      score_el_trte = model_el_trte.score(x_test, y_test)
      scorelist.append(score_el_trte)
      scores['model_el_trte']=score_el_trte

      ## Normalized
      model_el_norm = elastic.fit(normx, y)
      score_el_norm = model_el_norm.score(normx, y)
      scorelist.append(score_el_norm)
      scores['model_el_norm']=score_el_norm

##################### D E C I S I O N  T R E E #####################
      tree=DecisionTreeRegressor()

      model_dt=tree.fit(x,y)
      score_dt=model_dt.score(x,y)
      scorelist.append(score_dt)
      scores['model_dt']=score_dt

      ## Train & Test
      model_dt_trte= tree.fit(x_train,y_train)
      score_dt_trte=model_dt_trte.score(x_test,y_test)
      scorelist.append(score_dt_trte)
      scores['model_dt_trte']=score_dt_trte

      ## Normalized
      model_dt_norm=tree.fit(normx,y)
      score_dt_norm=model_dt_norm.score(normx,y)
      scorelist.append(score_dt_norm)
      scores['model_dt_norm']=score_dt_norm

##################### R A N D O M  F O R E S T #####################
      rf=RandomForestRegressor (n_estimators=100,random_state=42)

      model_rf=rf.fit(x,y)
      score_rf=model_rf.score(x,y)
      scorelist.append(score_rf)
      scores['model_rf']=score_rf

      ## Train & Test
      model_rf_trte=rf.fit(x_train,y_train)
      score_rf_trte=model_rf_trte.score(x_test,y_test)
      scorelist.append(score_rf_trte)
      scores['model_rf_trte']=score_rf_trte

      ## MinMaxScaler
      n_x=MinMaxScaler().fit_transform(x)
      nx_train,nx_test,ny_train,ny_test=train_test_split(n_x,y,train_size=0.80,random_state=42)
      nrf=RandomForestRegressor(n_estimators=200,n_jobs=-1,random_state=42)
      model_rf_mm=nrf.fit(nx_train,ny_train)
      score_rf_mm=model_rf_mm.score(nx_test,ny_test)
      scorelist.append(score_rf_mm)
      scores['model_rf_mm']=score_rf_mm

      ## StandardScaler
      s_x=StandardScaler().fit_transform(x)
      sx_train,sx_test,sy_train,sy_test=train_test_split(s_x,y,train_size=0.80,random_state=42)
      srf=RandomForestRegressor(n_estimators=200,n_jobs=-1,random_state=42)
      model_rf_ss=srf.fit(sx_train,sy_train)
      score_rf_ss=model_rf_ss.score(sx_test,sy_test)
      scorelist.append(score_rf_ss)
      scores['model_rf_ss']=score_rf_ss

##################### K N E I G H B O R S #####################
      komsu=KNeighborsRegressor(n_neighbors=3)

      model_kn=komsu.fit(x,y)
      score_kn=model_kn.score(x,y)
      scorelist.append(score_kn)
      scores['model_kn']=score_kn

      ## Train & Test
      model_kn_trte=komsu.fit(x_train,y_train)
      score_kn_trte=model_kn_trte.score(x_test,y_test)
      scorelist.append(score_kn_trte)
      scores['model_kn_trte']=score_kn_trte

##################### T A H M İ N  M O D E L İ N İ N  B U L U N M A S I #####################
## En yüksek skora sahip model seçiliyor.
      maxsc=max(scorelist)
      if maxsc==score_lr:
         model=model_lr
      elif maxsc==score_lr_trte:
           model= model_lr_trte
      elif maxsc==score_lr_norm:
           model= model_lr_norm
      elif maxsc==score_rd:
           model= model_rd
      elif maxsc==score_rd_trte:
           model= model_rd_trte
      elif maxsc==score_rd_norm:
           model= model_rd_norm
      elif maxsc==score_ls:
           model= model_ls
      elif maxsc==score_ls_trte:
           model= model_ls_trte
      elif maxsc==score_ls_norm:
           model= model_ls_norm
      elif maxsc==score_el:
           model= model_el
      elif maxsc==score_el_trte:
           model= model_el_trte
      elif maxsc==score_el_norm:
           model= model_el_norm
      elif maxsc==score_dt:
           model= model_dt
      elif maxsc==score_dt_trte:
           model= model_dt_trte
      elif maxsc==score_dt_norm:
           model= model_dt_norm
      elif maxsc==score_rf:
           model= model_rf
      elif maxsc==score_rf_trte:
           model= model_rf_trte
      elif maxsc==score_rf_mm:
           model= model_rf_mm
      elif maxsc==score_rf_ss:
           model= model_rf_ss
      elif maxsc==score_kn:
           model= model_kn
      else:
           model= model_kn_trte

##################### P R E D I C T I O N #####################
## En yüksek skora sahip model kullanılarak tahmin yapılıyor.
      tahmin=model.predict([laptopnew.drop(columns="Price_euros").loc[len(laptopnew)-1]])

################################# S T R E A M L I T  S O N U Ç L A R #################################
      st.write("**Marka:**",product)
      st.write("**GPU :**", gpu)
      st.write("**CPU :**", cpu)
      st.write("**RAM :**", str(ram),"GB")
      st.write("**Ekran Boyutu:**", str(size),("inches"))
      st.write("-------------------------------")
      st.write("**TAHMİNİ FİYAT**: ",str(tahmin.item()), "Euro")
      st.write("Tahmin doğruluğu: %",str((maxsc*100).item()))


