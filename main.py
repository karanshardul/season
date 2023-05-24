import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from ast import literal_eval
from streamlit_lottie import st_lottie

model = pickle.load(open('model.pkl','rb'))
location_name = pickle.load(open('location_name.pkl','rb'))
final_rating = pickle.load(open('final_rating.pkl','rb'))
pivot_table = pickle.load(open('pivot_table.pkl','rb'))
data = pickle.load(open('data.pkl','rb'))
seas = final_rating['Season'].values
season = list(set(seas))
lll = final_rating['Location'].values
locations = list(set(lll))

# st.markdown(hide_menu,unsafe_allow_html=True)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_101rlWCIdG.json")







st.set_page_config(page_title="JetSetWiz",page_icon=":desert_island:",layout="wide")

with st.container():
    l , r,e,t,y,u,i,o, = st.columns(8)
    with l:
        st.subheader("JetSetWiz.com")
    with r:
        st_lottie(lottie_coding,height=50,key="coding")
        st.write(" ")
with st.container():

    st.write("       ")
    st.write("       ")
    st.write("----")


with st.container():
    st.subheader("Season-Wise recommendation")
    st.write(" ")
    one_col , two_col, third_col, fourth_col = st.columns(4)
    with one_col:
        selected_season = st.selectbox('Select season?',season)
        def season(selected_season):
            data['Season'] = data['Season'].str.lower()
            filtered_data = data.loc[data['Season'] == selected_season.lower()]
            alll = filtered_data[['Location']]
            return alll

        df = season(selected_season)
    with two_col:
        
        cities = st.selectbox('Select/Type City?',df)
    with fourth_col:
        st.write(" ")


with st.container():
    st.subheader(" ")


with st.container():
    st.subheader("")


with st.container():
    st.subheader(":green[Other places u might like to visit...]")

url = "itinerary_arunachal.html"

def recommend_loc(location_name):
    loc_id = np.where(pivot_table.index == location_name)[0][0]
    distance, suggestion = model.kneighbors(pivot_table.iloc[loc_id,:].values.reshape(1,-1),n_neighbors=6)

    for i in range(len(suggestion)):
        loc = pivot_table.index[suggestion[i]]
        for j in loc:
            if j != location_name:
                    with st.container():
                        st.subheader(j)
                        # st.markdown(f"[Locate..]({url})")
                        st.write(" ")

recommend_loc(cities)



