### import libraries
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import altair as alt

### Create a title
st.title("Codon Frecuency Classification Project")

### DATA LOADING

### A. define function to load data
def load_data(path):
    #import data
    codon_df_clean = pd.read_csv(path)
    return codon_df_clean

codon_df_clean = load_data("codon_df_clean.csv")

### DATA ANALYSIS & VISUALIZATION

### B. Add filter on side bar after initial bar chart constructed


st.bar_chart(data=codon_df_clean, x='Kingdom', order=codon_df_clean['Kingdom'].value_counts().index, color='blue')