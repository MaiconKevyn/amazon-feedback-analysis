import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta, date
import matplotlib.pyplot as plt
import dashboard_helpers as dh


st.set_page_config(
    page_title="Feedback System Dashboard",
    page_icon="https://img.icons8.com/?size=160&id=vzcEz9itFD3W&format=png",
    layout="wide",
    initial_sidebar_state="auto",
    )

#Data

df = pd.read_csv("topics_classified.csv")
df['reviewTime'] = pd.to_datetime(df['reviewTime'], format='%Y-%m-%d')

max_date = df['reviewTime'].max()
min_date = df['reviewTime'].min()


col1, col2 = st.columns([1, 1])
with col1:
    start_date = st.date_input("Select from-date", min_date, min_value=min_date, max_value=max_date)
with col2:
    end_date = st.date_input("Select to-date", max_date, min_value=min_date, max_value=max_date)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

df_filtered = df[(df['reviewTime'] >= start_date) & (df['reviewTime'] <= end_date)]


col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
with col1:
    placeholder_reporting = st.empty()
with col2:
    placeholder_avg_rating = st.empty()
with col3:
    placeholder_avg_time = st.empty()
with col4:
    placeholder_max_time = st.empty()
with col5:
    placeholder_min_time = st.empty()

##SENTIMENT COUNT
sentiment_counts = df['sentiment'].value_counts()
sentiment_labels = sentiment_counts.index.tolist()
sentiment_values = sentiment_counts.values.tolist()

col1, col2 = st.columns([1, 1])

with col1:
    st.plotly_chart(dh.overall_sentiment(df_filtered))
with col2:
    st.plotly_chart(dh.sentiment_overtime(df_filtered))


col1, col2 = st.columns([1, 1])

with col1:
    st.plotly_chart(dh.topic_distributions(df_filtered))
with col2:
    st.plotly_chart(dh.plot_sentiment_distribution_by_topic(df_filtered))


col1, col2 = st.columns([1, 1])

with col1:
    st.plotly_chart(dh.plot_topic_overtime(df_filtered))


total_reportings = len(df_filtered) 
placeholder_reporting.metric("Total Reportings", total_reportings)

avg_rating = df_filtered['overall'].mean().round(2)
placeholder_avg_rating.metric("Average rating", avg_rating)


