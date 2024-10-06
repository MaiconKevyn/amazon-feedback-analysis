import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

def sentiment_overtime(df):
    #df_grouped = df.groupby(['reviewTime', 'sentiment']).size().unstack(fill_value=0)
    df['reviewTime'] = pd.to_datetime(df['reviewTime'], format='%Y-%m-%d')
    df_grouped = df.groupby([df['reviewTime'].dt.to_period('M'), 'sentiment']).size().unstack(fill_value=0)
    df_grouped.index = df_grouped.index.to_timestamp()

    daily_totals = df_grouped.sum(axis=1)
    normalized_sentiments = df_grouped.divide(daily_totals, axis=0)

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


def overall_sentiment(df):
    colors = {'positive': '#6BCD53', 'neutral': 'orange', 'negative': '#FF6347'}
    sentiment_labels = ['positive', 'neutral', 'negative']

    sentiment_counts = df['sentiment'].value_counts()
    sentiment_labels = sentiment_counts.index.tolist()
    sentiment_values = sentiment_counts.values.tolist()
    # Sorting the colors according to the sentiment labels for consistent application in both charts.
    sorted_colors = [colors[label] for label in sentiment_labels]

    # Creating a layout with two subplots: one for the bar chart and one for the pie chart.
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'bar'}, {'type':'domain'}]])

    # Adding a bar chart to the first subplot, showing sentiment counts with the text labels inside each bar.
    # The bars are colored according to the sentiment, and the text is positioned for clear visibility.
    fig.add_trace(
        go.Bar(
            x=sentiment_labels, 
            y=sentiment_values, 
            marker_color=sorted_colors, 
            text=sentiment_values, 
            textposition='inside', 
            showlegend=False,
            hoverinfo='y',
            textfont=dict(size=18)
        ),
        row=1, col=1
    )

    # Adding a pie chart to the second subplot, representing the sentiment distribution as percentages.
    # The pie chart has a hole in the center (donut chart), and the colors match those used in the bar chart.
    fig.add_trace(
        go.Pie(
            labels=sentiment_labels, 
            values=sentiment_values, 
            hole=.4, 
            marker=dict(colors=sorted_colors), 
            hoverinfo='label+percent',
            textfont=dict(size=18)
        ),
        row=1, col=2
    )

    # Updating the layout to include a title, annotations, and overall chart dimensions.
    # The title is centered, and annotations describe the two charts for clarity.
    fig.update_layout(
        title_text='Sentiment Distributions',
        
        title_x=0.25,  # Center the main title
        title_font=dict(family="Arial, sans-serif",
                        size=24,
                        color="black"),  # Increase the font size of the title
        annotations=[
            dict(text='Sentiment Count', x=0.11, y=1.15, xref='paper', yref='paper', showarrow=False, font=dict(size=20)),
            dict(text='Sentiment Percentage', x=0.92, y=1.15, xref='paper', yref='paper', showarrow=False, font=dict(size=20))
        ],
        width=1000,  # Set the width of the figure
        height=400   # Set the height of the figure
    )

    # Display the final combined visualization with the bar chart and pie chart side by side.
    return fig


def topic_distributions(df_filtered):
    real_topic_counts = df_filtered['deberta_topics'].value_counts().sort_values(ascending=True)
    topic_labels = real_topic_counts.index.tolist()
    topic_values = real_topic_counts.values.tolist()

    # Cores personalizadas para cada tópico
    topic_colors = {
        'Product functionality': '#89CFF0',  # light blue
        'Price': '#FFA500',  # orange
        'Delivery Service': '#FF6347',  # red
        'Compatibility with devices': '#3CB371',  # medium sea green
        'Quality': '#FFD700'  # gold
    }
    sorted_topic_colors = [topic_colors[label] for label in topic_labels]

    # Cria o gráfico de barras horizontal para a contagem de tópicos
    fig_topics = go.Figure(go.Bar(
        y=topic_labels,  # Coloca os rótulos no eixo Y
        x=topic_values,  # Coloca os valores no eixo X
        orientation='h',  # Define orientação horizontal
        marker_color=sorted_topic_colors,
        text=topic_values,
        textposition='outside',
        textfont=dict(
            color='black',  # Cor do texto nas barras
            size=14  # Tamanho do texto nas barras
        ),
        hoverinfo='x'
    ))

    fig_topics.update_layout(
        title={
            'text': 'Distribution of Classified Topics',
            'x': 0.5,  # Centraliza o título
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family="Arial, sans-serif",
                size=24,
                color="black"  # Cor da fonte do título
            ),
        },
        yaxis_title='Topic',
        xaxis_title='Frequency',
        xaxis=dict(
            title_font=dict(size=18, color='black'),  # Tamanho e cor do título do eixo X
            tickfont=dict(size=15, color='black')  # Tamanho e cor dos ticks do eixo X
        ),
        yaxis=dict(
            title_font=dict(size=18, color='black'),  # Tamanho e cor do título do eixo Y
            tickfont=dict(size=15, color='black'),  # Tamanho e cor dos ticks do eixo Y
            gridcolor='#e6e8eb',  # Cor das linhas de grade
            gridwidth=0.5,
        ),
        width=1500,
        height=500,
        plot_bgcolor='rgba(255, 255, 255, 1)',  # Fundo totalmente branco
        paper_bgcolor='rgba(255, 255, 255, 1)'  # Fundo externo também branco
    )

    return fig_topics


def plot_sentiment_distribution_by_topic(df_filtered):
    topic_sentiment_count = df_filtered.groupby(['deberta_topics', 'sentiment']).size().unstack(fill_value=0)
    topic_percentages = topic_sentiment_count.div(topic_sentiment_count.sum(axis=1), axis=0) * 100
    
    colors = {"negative": "#FF6347", "neutral": "orange", "positive": "#6BCD53"}
    sentiments_order = ['negative', 'neutral', 'positive']  

    fig = go.Figure()

    for i, sentiment in enumerate(sentiments_order):
        if i == 0:
            base = None
        else:
            base = topic_percentages[sentiments_order[:i]].sum(axis=1)

        fig.add_trace(go.Bar(
            name=sentiment,
            x=topic_percentages.index,
            y=topic_percentages[sentiment],
            marker_color=colors[sentiment],
            base=base,
            offsetgroup=0,
            hoverinfo='y+name'
        ))

    fig.update_layout(
        barmode='relative',
        title={
            'text': 'Sentiment Distribution by Topic in Percentage',
            'x': 0.5,  # Centraliza o título
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family="Arial, sans-serif",  # Família da fonte
                size=24,  # Tamanho da fonte do título
                color="black"  # Cor da fonte do título
            ),
        },
        xaxis_title="Topics",
        yaxis_title="Percentage",
        xaxis=dict(
            categoryorder='total descending',
            title_font=dict(size=18, color='black'),  # Tamanho e cor do título do eixo X
            tickfont=dict(size=16, color='black'),
            tickangle=0  
        ),
        yaxis=dict(
            title_font=dict(size=18, color='black'),  # Tamanho e cor do título do eixo Y
            tickfont=dict(size=16, color='black'),
            gridcolor='#e6e8eb',  # Cor das linhas de grade
            gridwidth=0.5,  # Tamanho e cor dos ticks do eixo Y
        ),
        legend_title_text='Sentiment',
        legend=dict(
            title_font=dict(size=14, color='black'),  # Tamanho e cor do título da legenda
            font=dict(size=12, color='black')  # Tamanho e cor dos itens da legenda
        ),
        width=1000,
        height=500,
        plot_bgcolor='rgba(255, 255, 255, 1)',  # Fundo totalmente branco do gráfico
        paper_bgcolor='rgba(255, 255, 255, 1)'  # Fundo externo também branco
    )

    return fig


def plot_topic_overtime(df_original):
        if 'reviewTime' in df_original.columns:
            df_original['reviewTime'] = pd.to_datetime(df_original['reviewTime'])
            df_original.set_index('reviewTime', inplace=True)

        daily_topic_counts = df_original.groupby([df_original.index.date, 'deberta_topics']).size().unstack(fill_value=0)

        # Ordenando colunas conforme as preferências para empilhamento visual
        sorted_columns = ['Compatibility with devices', 'Price', 'Product functionality', 'Quality', 'Delivery Service']
        daily_topic_counts = daily_topic_counts.reindex(sorted_columns, axis=1, fill_value=0)

        # Cores atribuídas por ordem definida
        topic_colors = {
            'Compatibility with devices': 'red',
            'Price': 'blue',
            'Product functionality': 'purple',
            'Quality': 'orange',
            'Delivery Service': 'green'
        }
        colors = [topic_colors[topic] for topic in sorted_columns]

        # Criando o gráfico Plotly
        fig_topic_overtime = go.Figure()
        for topic, color in zip(sorted_columns, colors):
            fig_topic_overtime.add_trace(go.Scatter(
                x=daily_topic_counts.index,
                y=daily_topic_counts[topic],
                stackgroup='one',  # define stacking
                name=topic,
                mode='lines',
                line=dict(width=0.5, color=color),
                fill='tonexty',  # fill area between traces
                hoverinfo='x+y'
            ))

        # Configurando layout do gráfico
        fig_topic_overtime.update_layout(
            title={
                'text': 'Topic Distribuition Over Time',
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(
                    family="Arial, sans-serif",
                    size=24,
                    color="black"
                ),
            },
            xaxis_title="Day",
            yaxis_title="Number of Mentions",
            legend_title='Topics',
            xaxis=dict(
                title_font=dict(size=18, color='black'),  # Título do eixo X
                tickmode='auto',
                nticks=10,
                tickfont=dict(size=16, color='black')  # Cor e tamanho dos ticks do eixo X
            ),
            yaxis=dict(
                title_font=dict(size=18, color='black'),  # Título do eixo Y
                tickfont=dict(size=16, color='black'),
                gridcolor='#e6e8eb',  # Cor das linhas de grade
                gridwidth=0.5,  # Cor e tamanho dos ticks do eixo Y
            ),
            legend=dict(
                title_font=dict(color='black', size = 18),  # Título da legenda
                font=dict(color='black')  # Itens da legenda
            ),
            hovermode='x unified',
            plot_bgcolor='rgba(255, 255, 255, 1)',  # Fundo totalmente branco do gráfico
            paper_bgcolor='rgba(255, 255, 255, 1)'  # Fundo externo também branco
        )

        return fig_topic_overtime