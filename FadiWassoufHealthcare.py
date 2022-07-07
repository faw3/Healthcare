#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#Importing Libraries

import pandas as pd
import numpy as np
import streamlit as st
import hydralit_components as hc 
import plotly.figure_factory as ff
import plotly.express as px
import raceplotly as rs
from raceplotly.plots import barplot
import plotly.graph_objects as go
from echarts import Echart, Legend, Bar, Axis
from pyecharts.charts import Bar
from pyecharts import options as opts
from streamlit_echarts import st_echarts
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# In[2]:

st.set_page_config(layout="wide",page_title=None)
df = pd.read_csv("C:\\Users\\USER\\Downloads\\finallifeexp.csv")

#Imputing Null Values

df["Adult_Mortality"] = df["Adult_Mortality"].fillna(df["Adult_Mortality"].mean())
df["Alcohol"] = df["Alcohol"].fillna(df["Alcohol"].mean())
df["Hepatitis_B"] = df["Hepatitis_B"].fillna(df["Hepatitis_B"].mean())
df["Measles"] = df["Measles"].fillna(df["Measles"].mean())
df["Polio"] = df["Polio"].fillna(df["Polio"].mean())
df["BMI"] = df["BMI"].fillna(df["BMI"].mean())
df["Total_expenditure"] = df["Total_expenditure"].fillna(df["Total_expenditure"].mean())
df["Diphtheria"] = df["Diphtheria"].fillna(df["Diphtheria"].mean())
df["Income_composition_of resources"] = df["Income_composition_of resources"].fillna(df["Income_composition_of resources"].mean())
df["Schooling"] = df["Schooling"].fillna(df["Schooling"].mean())
df["GDP"] = df["GDP"].fillna(df["GDP"].mean())
df["Infant_Mortality_Rate"]= df["Infant_Mortality_Rate"].fillna(df["Infant_Mortality_Rate"].mean())
df["thinness _10-19_years"] = df["thinness _10-19_years"].fillna(df["thinness _10-19_years"].mean())
df["thinness _5-9_years"] = df["thinness _5-9_years"].fillna(df["thinness _5-9_years"].mean())
df["Population"] = df["Population"].fillna(df["Population"].mean()) 








#Creating Navigation bar
menu_data = [{"label":"Overview"},{'label':'Vaccinations'},{'label':'Mortality'},{'label':'Monetary'},{'label':'Life Expectancy Prediction'}]
menu_id = hc.nav_bar(menu_definition=menu_data, sticky_mode='sticky')
if menu_id =="Overview":
  st.header("Life Expectancy Prediction")
  st.subheader("Global Overview")
  col1,col2 = st.columns([6,3])


       
 #Select box for country status

  with st.sidebar:
        hh2 = ["All","Developed","Developing"]
        jj2 = st.selectbox("Status",hh2,0)
        if jj2 == "All":
          df=df[df["Status"]==df["Status"]]
        if jj2=="Developed":
         df = df[df["Status"]==jj2]
        if jj2 == "Developing":
         df=df[df["Status"]==jj2]
        

    
        
  

  
  #Creating radio buttons for factors within a select box of the factors' categories
  
  with col2:
      df_1 = df[["GDP","Total_expenditure","Income_composition_of resources"]]
      df_2 = df[["HIV/AIDS","thinness _10-19_years","thinness _5-9_years","Measles","Under_five","Infant_Mortality_Rate","Adult_Mortality"]]
      df_3 = df[["Population","Life_expectancy","BMI","Alcohol","Schooling"]]
      df_4 = df[["Hepatitis_B","Polio","Diphtheria"]]
      rr = st.selectbox("Select Factor",["Monetory Factors","Diseases","General","Vaccinations"])
      if rr =="Monetory Factors":

       
      #Conditional y variable for the map below, where what the map shows depends on the value of tt
       factors= st.radio("Monetory Factors",df_1.columns)
       if factors == "GDP":
        tt = df_1["GDP"]
       if factors == "Total_expenditure":
        tt = df_1["Total_expenditure"]
       if factors == "Income_composition_of resources":
        tt = df_1["Income_composition_of resources"]
       
      if rr =="General":
        factors= st.radio("General Factors",df_3.columns)
       
        if factors == "Population":
         tt = df_3["Population"]
        if factors == "Life_expectancy":
         tt = df_3["Life_expectancy"]  
        if factors == "BMI":
         tt = df_3["BMI"]
        if factors == "Alcohol":
         tt = df_3["Alcohol"] 
        if factors == "Schooling":
         tt = df_3["Schooling"] 
    
     
    

  with col2:
     if rr =="Diseases":
      factors= st.radio("Mortality Rates/1000 and Diseases",df_2.columns)
       

      
      if factors == "Measles":
        tt = df_2["Measles"] 
      
      if factors == "HIV/AIDS":
        tt = df_2["HIV/AIDS"]
      
      if factors == "thinness _5-9_years":
        tt = df_2["thinness _5-9_years"]
      if factors == "thinness _10-19_years":
        tt = df_2["thinness _10-19_years"]
      if factors == "Under_five":
        tt = df_2["Under_five"]
      if factors == "Adult_Mortality":
        tt = df_2["Adult_Mortality"]
      if factors == "Infant_Mortality_Rate":
        tt = df_2["Infant_Mortality_Rate"]

     if rr =="Vaccinations":
      factors= st.radio("Vaccination among 1 years-old in % ",df_4.columns)

      
      if factors == "Hepatitis_B":
        tt = df_4["Hepatitis_B"]

      
      if factors == "Polio":
        tt = df_4["Polio"]
      
      if factors == "Diphtheria":
        tt = df_4["Diphtheria"]
     
  
# Creating plotly map
  with col1:  
   data = [dict(
   type='choropleth',

   autocolorscale = False,
   locations = df['Country'],
   z = tt.astype(float),
   locationmode = 'country names',

   hoverinfo = "location+z",
   marker = dict(
   line = dict (
   color = 'rgb(255,255,255)',
   width = 2
)
),
   colorbar = dict(
   title = 'Factor Intensity'
)
)]

   layout = dict(
   

   geo = dict(
   scope='world',

   showlakes = True,
   lakecolor = 'rgb(255, 255, 255)',
   
)
)

   fig = dict(data=data, layout=layout)
   st.plotly_chart(fig)

  col5,col6,col7,col8 = st.columns([3,3,3,3])
  with col5:
    st.markdown("Average Vaccination Rates")
    theme_vacc = {'bgcolor':'#EFF8F7','content_color':'navy','progress_color':'navy'}
    
    #Info Cards for vaccinations
     
    hc.info_card(title="Hepatitis B",bar_value=df["Hepatitis_B"].mean(),content=round(df["Hepatitis_B"].mean()),theme_override=theme_vacc)
    hc.info_card(title="Polio",bar_value=df["Polio"].mean(),content=round(df["Polio"].mean()),theme_override=theme_vacc)
    hc.info_card(title="Diphtheria ",bar_value=df["Diphtheria"].mean(),content=round(df["Diphtheria"].mean()),theme_override=theme_vacc)

  with col6:
    st.markdown("Average Mortality Rates")
    theme_deaths = {'bgcolor':'#FFF0F0','content_color':'darkred','progress_color':'darkred'}

    #info cards for mortalities    
    hc.info_card(title="Infants Mortality",bar_value=df["Infant_Mortality_Rate"].mean(),content=round(df["Infant_Mortality_Rate"].mean()),theme_override=theme_deaths)
    hc.info_card(title="Adult Mortality",bar_value=df["Adult_Mortality"].mean(),content=round(df["Adult_Mortality"].mean()),theme_override=theme_deaths)
    hc.info_card(title="Under 5 Mortality",bar_value=df["Under_five"].mean(),content=round(df["Under_five"].mean()),theme_override=theme_deaths)
    
  with col7:
    st.markdown("Average Monetary Values")
    theme_mon = {'bgcolor':'#EFF8F7','content_color':'darkgreen','progress_color':'darkgreen'}
    #info cards for economic factors
    income = df["Income_composition_of resources"].mean()*100
    hc.info_card(title="GDP",bar_value=df["GDP"].mean(),content=round(df["GDP"].mean()),sentiment="good")
    hc.info_card(title="Health Exp.",bar_value=df["Total_expenditure"].mean(),content=round(df["Total_expenditure"].mean()),sentiment="good")
    hc.info_card(title="Household Exp.",bar_value=income,content="{:.1f}".format(income),sentiment="good")
        
  with col8:
    st.markdown("Average Social Rates")
    theme_gen = {'bgcolor':'#F9F9F9','content_color':'yellow','progress_color':'yellow'}
    #Info card for general factors
    hc.info_card(title="Life Expectancy",bar_value=df["Life_expectancy"].mean(),content=round(df["Life_expectancy"].mean()),sentiment="neutral")
    hc.info_card(title="Population",bar_value=df["Population"].mean(),content=round(df["Population"].mean()),sentiment="neutral")
    hc.info_card(title="School Years",bar_value=df["Schooling"].mean(),content=round(df["Schooling"].mean()),sentiment="neutral")
    
        
    
    

    
    
    



      

       
  
    

#Mortality Page
if menu_id=="Mortality":
    #Country selectbox
    df["Total_deaths"] = df["Adult_Mortality"] + df["Infant_Mortality_Rate"] + df["Under_five"]
    with st.sidebar:
        Countries_tolist= df["Country"].unique().tolist()
        s1 = st.selectbox("Choose Country",Countries_tolist,0)
        df = df[df["Country"]==s1]

    st.subheader("Mortalities Per 1000")
    
  
    col1,col2,col3 = st.columns([3,3,3])
    
    with col1:
      #mortality info cards
        theme = {'bgcolor':'#FFFFFF','content_color':'darkred','progress_color':'darkred'}
        hc.info_card(title="Infants Mortality",bar_value=df["Infant_Mortality_Rate"].mean(),content=round(df["Infant_Mortality_Rate"].mean()),theme_override=theme)
    with col2:
        hc.info_card(title="Adults Mortality",bar_value=df["Adult_Mortality"].mean(),content=round(df["Adult_Mortality"].mean()),theme_override=theme)
        
    with col3:
        hc.info_card(title="Under Five Mortality",bar_value=df["Under_five"].mean(),content=round(df["Under_five"].mean()),theme_override=theme)


    #Total Deaths bargraph
    option = {
    "title": {
    "text": 'Total Deaths by Years'
    },
  "xAxis": {
    "type": 'category',
    "splitLine":{'show':False},
    "data": df["Year"].tolist(),
    'color':"red",
    },
  "yAxis": {
    "type": 'value',
    "splitLine":{'show':False},
  },
  "series": [
    {
      "data": df["Total_deaths"].tolist(),
       'color':"darkred",
      "type": 'bar'
      
    }
  ]
}

    st_echarts(options=option)

    col7,col8 = st.columns([12,0.5])
    with col7:
      #Thinness line graph
      option7 = {
  "title": {
    "text": 'Thinness Age Category'
  },
  "tooltip": {
    "trigger": 'axis'
  },
  "legend": {
    "data": ['5-10','10-19']
  },
  "grid": {
    "left": '3%',
    "right": '4%',
    "bottom": '3%',
    "containLabel": True
  },
  "toolbox": {
    "feature": {
      "saveAsImage": {}
    }
  },
  "xAxis": {
    "type": 'category',
    "splitLine":{'show':False},
    "boundaryGap": False,
    "data": df["Year"].tolist(),
    
  },
  "yAxis": {
    "type": 'value',
    "splitLine":{'show':False},
  },
  "series": [
    {
      "name": '5-10',
      "type": 'line',
      "stack": 'Total',
      "data": df["thinness _5-9_years"].tolist(),
      'color':"darkred"
    },
    {
      "name": '10-19',
      "type": 'line',
      "stack": 'Total',
      "data": df["thinness _10-19_years"].tolist(),
      'color':'lightred'
    }

  ]
    }
      st_echarts(options=option7)

    
      

      


     
    
#Monetary page

if menu_id == "Monetary":
    st.subheader("Global Metrics")
    col4,col5,col6= st.columns([3,3,3])
    
    
    #Economic metrics
    income1 = df["Income_composition_of resources"].mean()*100
    col4.metric(label = "Average GDP in 10,000$",value=round(df.GDP.mean()),delta = "1.5")
    col5.metric(label = "Average Govermental Expenditure on Health",value=round(df["Total_expenditure"].mean()),delta = "-1.5")   
    col6.metric(label = "Household Exp. On Health",value="{:.2f}".format(income1),delta = "-1.5") 
    
    col7,col8 = st.columns([6,3])
    df["money"] = df["Total_expenditure"]*-1
    
    #Country selectbox

    with st.sidebar:
        Countries_list = df["Country"].unique().tolist()
        s2 = st.selectbox("Choose Country",Countries_list,0)
        df = df[df["Country"]==s2]
    with col7:
      #GDP filled linechart
      fig9 = px.area(df,x="Year",y="GDP",color_discrete_sequence =['green']*len(df),line_group="Country")
      fig9.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
       plot_bgcolor='rgba(0,0,0,0)')
      st.plotly_chart(fig9)

    col9,col10=st.columns([5,5])
    
    with col9:
      #Health expenditure out of total

      st.subheader("Govermental Expenditure on Health%")
      option10 = {
  "tooltip": {
    "trigger":'axis',
    "axisPointer": {
      "type": 'shadow'
    }
  },
  "legend": {
    "data": ['Profit', 'Expenses']
  },
  "grid": {
    "left": '3%',
    "right": '4%',
    "bottom": '3%',
    "containLabel": True
  },
  "xAxis": [
    {
      "type": 'value'
    }
  ],
  "yAxis": [
    {
      "type": 'category',
      
      "axisTick": {
        "show": False
      },
      "data": df["Year"].tolist()
    }
  ],
  "series": [
    {
      "name": 'Income',
      'color':'green',
      "type": 'bar',
      "stack": 'Total',
      "label": {
        "show": True
        
      },
      "emphasis": {
        "focus": 'series'
      },
      "data": [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
    },
    {
      "name": 'Health Exp.',
      'color':'red',
      "type": 'bar',
      "stack": 'Total',
      "label": {
        "show": True,
        "position": 'left'
      },
      "emphasis": {
        "focus": 'series'
      },
      "data": df["money"].tolist()
    }
  ]
}
      st_echarts(options=option10)

    with col10:
      #Progress of individual health exp yearly scatter plot
     st.subheader("Household Exp. On Health%")
     fig10 = px.scatter(df,x="Year",y="Income_composition_of resources",color_discrete_sequence =['green']*len(df),
     labels={"Income_composition_of resources":"Individual Exp on Health"})

     fig10.update_layout(
      paper_bgcolor='rgba(0,0,0,0)',
       plot_bgcolor='rgba(0,0,0,0)'
       
      )

     
     st.plotly_chart(fig10)


      
    with col8:
      #Life expectancy info card
        hc.info_card(title="Life_expectancy",bar_value=df["Life_expectancy"].mean(),content=round(df["Life_expectancy"].mean()),sentiment="good")
        #Colored table
        def highlight_conversion(s):
          return ['background-color: lightgreen']*len(s) if s.Total_expenditure >df["Total_expenditure"].mean() else ['background-color:red']*len(s)
        st.markdown("Health Expenditure Average")
        st.dataframe(df.style.apply(highlight_conversion, axis=1))
  
        

#Vaccination page
if menu_id =="Vaccinations":

  st.subheader("Vaccination Rate%")
    
  
  
 
  col9,col10,col11=st.columns([3,0.5,3])
  #Country selectbox
  with st.sidebar:
        Countries_tolist1 = df["Country"].unique().tolist()
        cc2= st.selectbox("Choose Country",Countries_tolist1,0)
        df = df[df["Country"]==cc2]
  with col9:
    #Vaccination progress bar
    theme1 = {'bgcolor':'#D3D3D3','progress_color':'navy','content_color':'white'}
    hc.progress_bar(df["Hepatitis_B"].mean(),"Hepatitis B",override_theme=theme1)
    hc.progress_bar(df["Polio"].mean(),"Polio",override_theme=theme1)
    hc.progress_bar(df["Diphtheria"].mean(),"Diphtheria", override_theme=theme1)
  with col10:
    st.markdown(round(df["Hepatitis_B"].mean())) 
    st.markdown(round(df["Polio"].mean()))
    st.markdown(round(df["Diphtheria"].mean()))
  with col11:
    theme3 = {'bgcolor':'#FAF9F6',"content_color":"navy","progress_color":'navy'}
    hc.info_card(title="Life Expectancy",bar_value=df["Life_expectancy"].mean(),content=round(df["Life_expectancy"].mean()),theme_override=theme3)
  st.markdown("Population Ove Time")  
  #Population bargraph
  option = {
  
  "xAxis": {
    "title": {
    "text": 'Population Over Time'
    },
    "type": 'category',
    "data": df["Year"].tolist()
  },
  "yAxis": {
    "type": 'value'
  },
  "series": [
    {
      "data": df["Population"].tolist(),
      "type": 'bar',
      'color':'navy'
    }
  ]
}
  st_echarts(options=option)

  col12,col13 = st.columns([5,5])
  with col12:
    #diseases linegraph
    df["HIV/AIDS"] = df["HIV/AIDS"]*df["Population"]/1000
    option2 = {
  "title": {
    "text": 'Diseases'
  },
  "tooltip": {
    "trigger": 'axis'
  },
  "legend": {
    "data": ['Measles','HIV/AIDS']
  },
  "grid": {
    "left": '3%',
    "right": '4%',
    "bottom": '3%',
    "containLabel": True
  },
  "toolbox": {
    "feature": {
      "saveAsImage": {}
    }
  },
  "xAxis": {
    "type": 'category',
    "splitLine":{'show':False},
    "boundaryGap": False,
    "data": df["Year"].tolist()
  },
  "yAxis": {
    "type": 'value',
    "splitLine":{'show':False},
  },
  "series": [
    {
      "name": 'Measles',
      "type": 'line',
      "stack": 'Total',
      "data": df["Measles"].tolist()
    },
    {
      "name": 'HIV/AIDS',
      "type": 'line',
      "stack": 'Total',
      "data": df["HIV/AIDS"].tolist()
    }

  ]
    }
    st_echarts(options=option2)
    with col13:
      #SCHOOL AND HEALTH EXP
      option3 = {
        "title": {
    "text": 'Years of School vs Exp. on Health'
        },
  "xAxis": {
    "type": 'category',
    "splitLine":{'show':False},
    "data": df["Schooling"].tolist()
  },
  "yAxis": {
    "type": 'value',
    "splitLine":{'show':False},
  },
  "series": [
    {
      "data": df["Income_composition_of resources"].tolist(),
      "type": 'line',
      "smooth": True
    }
  ]
}
      st_echarts(options=option3)

###importing models
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Normalizer
nn = Normalizer()
df["Status"] = pd.Categorical(df["Status"])

rf = RandomForestRegressor()

#Life expectancy page
if menu_id == "Life Expectancy Prediction":
  
  
  col17,col18,col19,col20,col21= st.columns([2,2,2,2,2])
  col22,col23 = st.columns([3,6])
  col24,col25 = st.columns([3,3])

  with col17:
    ## 15 sliders for machine learning prediction
    mes_inpt=st.slider(
   "Measles", min_value=0.1, max_value=1.1,step=0.1,value=0.1)
    df=df[df["Measles"]<(mes_inpt)]

    hepatisis_inpt=st.slider(
   "Hepatitis B Vaccination%", min_value=0, max_value=100,step=1,value=60)
    df=df[df["Hepatitis_B"]<(hepatisis_inpt)]


 

    adult_inpt=st.slider(
    "Adults", min_value=20 ,max_value=800,step=10,value=200)
    df=df[df["Adult_Mortality"]<(adult_inpt)]

  with col18:
    tt_inpt=st.slider(
    "Avg Expenditure", min_value=0, max_value=1100,step=10,value=500)
    df=df[df["percentage_expenditure"]<(tt_inpt)]


    gdp_inpt=st.slider(
   "GDP", min_value=6000, max_value=100000000,step=100000,value=3000000)
    df=df[df["GDP"]<(gdp_inpt)]

    
    Population_inpt=st.slider(
   "Population", min_value=82000, max_value=60000000,step=5000,value=200000)
    df=df[df["Population"]<(Population_inpt)]
  
  with col19:
    Polio_inpt=st.slider(
   "Polio", min_value=50, max_value=100,step=1,value=80)
    df=df[df["Polio"]<(Polio_inpt)]


    Infant_inpt=st.slider(
  "Infants Mortality Rate", min_value=0, max_value=120,step=1,value=80)
    df=df[df["Infant_Mortality_Rate"]<(Infant_inpt)]



    tt_inpt=st.slider(
  "Total Expenditure", min_value=0, max_value=12,step=1,value=5)
    df=df[df["Total_expenditure"]<(Population_inpt)]

  with col20:
    Alc_inpt=st.slider(
"Alcohol Consumption in Ltrs", min_value=5.5, max_value=16.1,step=1.5,value=7.5)
    df=df[df["Alcohol"]<(Alc_inpt)]


    Diph_inpt=st.slider(
"Diphtheria", min_value=50, max_value=100,step=1,value=80)
    df=df[df["Diphtheria"]<(Diph_inpt)]

    HIV_inpt=st.slider(
"HIV/AIDS", min_value=0.1, max_value=2.1,step=0.1,value=0.5)
    df=df[df["HIV/AIDS"]<(HIV_inpt)]


  with col21:
    thin10_inpt=st.slider(
"Thinness", min_value=0.6, max_value=16.1,step=0.5,value=8.5)
    df=df[df["thinness _10-19_years"]<(thin10_inpt)]


    sc_inpt=st.slider(
"School Years", min_value=12, max_value=21,step=1,value=15)
    df=df[df["Schooling"]<(sc_inpt)]


    BMI_inpt=st.slider(
"Average BMI", min_value=10, max_value=45,step=5,value=15)
    df=df[df["BMI"]<(BMI_inpt)]

  features_with_outliers = ['Life_expectancy','Adult_Mortality','Infant_Mortality_Rate','Hepatitis_B','Polio','Total_expenditure','HIV/AIDS','Income_composition_of resources','thinness _10-19_years','Diphtheria','BMI','Measles',"percentage_expenditure",'Alcohol','Under_five','Population']
  for feature in features_with_outliers:
    q1 = df[feature].quantile(0.25)#the median of the lower half 
    q3 = df[feature].quantile(0.75)#the median of the upper half 
    IQR = q3-q1#interquartile range
    lower_limit = q1 - (IQR*1.5)
    upper_limit = q3 + (IQR*1.5)
    df.loc[df[feature]<lower_limit,feature] = lower_limit
    df.loc[df[feature]>upper_limit,feature] = upper_limit


  stan= StandardScaler()
  #splitting x and y
  X = df.drop(["Life_expectancy","Year","Under_five","thinness _5-9_years","Status","Income_composition_of resources","Country"],axis=1)
  y = df["Life_expectancy"]
  #splitting X and y into train and test
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=111)
#Normalizing data
  XX_train = nn.fit_transform(X_train)
  XX_test = nn.fit_transform(X_test)
#Tring linear regression
  lr = LinearRegression()
  lr.fit(XX_train,y_train)
  ll = lr.predict(XX_test)
  from sklearn.metrics import mean_squared_error
  lin_mse = mean_squared_error(y_test ,ll)
  lin_rmse = np.sqrt(lin_mse)
    




#fitting the random forest model
  rf.fit(XX_train,y_train)
  ft =rf.predict(XX_test)
  from sklearn.metrics import mean_squared_error
  rand_mse = mean_squared_error(y_test ,ft)
  rand_rmse = np.sqrt(rand_mse)
  
#Prediction button
  with col22:
    if st.button ("Expected Life Span"):
     st.subheader(ft.mean())
     st.markdown("""Please note that the factors used to determine Life Expectancy are limited to the Country Scale Factors.
As an individual, you have an extremely high control level over your life expectancy based on your lifestyle:""")


   
#Factors increasing life exp
  with col24:
    st.subheader("Increases Life Expectancy")
    st.markdown("""Exercising 3 times a week      +7""")
    st.markdown("Having/maintaining a healthy diet  +10")
    st.markdown("Socialize +5")
#Factors decreasing life exp
  with col25:
    st.subheader("Deacreases Life Expectancy")
    st.markdown("Smoking  -10")
    st.markdown("Stressing too much  -2.8")
    st.markdown("Little/Too much sleep -11")




 

  
  


  



   

      


  


  

  
       
     
    
    
  



  










   
 

        
       

        
        

      

      
      


