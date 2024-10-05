import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

def sentiment_overtime(df):
# Prepara dados para o gráfico de Sentimento ao Longo do Tempo
    df_grouped = df.groupby(['Date', 'GPT-4']).size().unstack(fill_value=0)
    daily_totals = df_grouped.sum(axis=1)
    normalized_sentiments = df_grouped.divide(daily_totals, axis=0)

    # Cria o gráfico com Plotly para Sentimento ao Longo do Tempo
    fig_time_series = go.Figure()
    colors = {'positive': '#6BCD53', 'neutral': 'orange', 'negative': '#FF6347'}

    for sentiment, color in colors.items():
        if sentiment in normalized_sentiments:
            fig_time_series.add_trace(go.Scatter(
                x=normalized_sentiments.index,
                y=normalized_sentiments[sentiment],
                mode='lines',
                name=sentiment,
                line=dict(color=color, width=2)
            ))

    fig_time_series.update_layout(
        width=1000,
        height=400,
        title={
            'text': 'Normalized Sentiment Analysis Over Time',
            'x': 0.5,  # Centraliza o título
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family="Arial, sans-serif",
                size=24,
                color="black"
            ),
        },    
        xaxis_title='Date',
        yaxis_title='Proportion',
        xaxis=dict(
            title_font=dict(size=18, color='black'),  # Tamanho e cor do título do eixo X
            tickfont=dict(size=16, color='black'),  # Tamanho e cor dos ticks do eixo X
            tickcolor='black'  # Cor dos ticks
        ),
        yaxis=dict(
            title_font=dict(size=18, color='black'),  # Tamanho e cor do título do eixo Y
            tickfont=dict(size=16, color='black'),  # Tamanho e cor dos ticks do eixo Y
            tickcolor='black'  # Cor dos ticks
        ),
        legend_title='Sentiment',
        legend=dict(
            title_font=dict(size=14, color='black'),  # Tamanho e cor do título da legenda
            font=dict(size=12, color='black')  # Tamanho e cor dos itens da legenda
        ),
        template='plotly_white',
        plot_bgcolor='rgba(255, 255, 255, 1)',  # Fundo totalmente branco
        paper_bgcolor='rgba(255, 255, 255, 1)'  # Fundo externo também branco
    )
    return fig_time_series