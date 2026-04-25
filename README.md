# Amazon Feedback Analysis

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.3-150458.svg)](https://pandas.pydata.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.45.1-FFD21E.svg)](https://huggingface.co/docs/transformers/index)
[![Plotly](https://img.shields.io/badge/Plotly-5.24.1-3F4F75.svg)](https://plotly.com/python/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991.svg)](https://openai.com/)
[![Streamlit Dashboard](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)

> A zero-shot feedback intelligence project that classifies Amazon product reviews by sentiment and topic, then exposes the results through an interactive Streamlit dashboard for time-based analysis.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the Project](#running-the-project)
- [Dataset](#dataset)
- [Modeling Approach](#modeling-approach)
- [Dashboard and Results](#dashboard-and-results)
- [Key Findings](#key-findings)
- [Limitations and Next Steps](#limitations-and-next-steps)
- [Acknowledgments](#acknowledgments)

## Overview
**Amazon Feedback Analysis** analyzes customer reviews from an Amazon product reviews dataset to identify both **sentiment** and **business-relevant topics** in user feedback. The workflow combines prompt-based sentiment classification with zero-shot topic classification and consolidates the output into a dashboard focused on operational insight.

### Key Capabilities
- **Sentiment Classification**: Labels each review as positive, neutral, or negative
- **Zero-Shot Topic Detection**: Assigns reviews to topics without task-specific fine-tuning
- **Time-Based Analysis**: Filters feedback by date range to inspect trend shifts over time
- **Interactive Dashboard**: Presents sentiment, topic, and mixed-topic sentiment views in Streamlit
- **Precomputed Analysis Assets**: Ships with processed CSV outputs for immediate exploration
- **Exploratory Notebook Workflow**: Keeps the end-to-end analysis reproducible in `intro.ipynb`

## Features
- **Dual-stage classification pipeline**: sentiment first, topic classification second
- **LLM-assisted sentiment analysis**: OpenAI GPT-4o used for zero-shot sentiment labeling
- **Transformer-based topic classification**: DeBERTa zero-shot model used for business topic assignment
- **Topic-level sentiment breakdown**: Highlights which areas concentrate more negative feedback
- **Temporal plots**: Tracks both sentiment and topic distribution across time
- **Review metrics**: Dashboard surfaces total reportings and average rating for filtered windows

## Architecture
![Workflow Overview](https://github.com/user-attachments/assets/0416d8de-bd33-4dad-b45a-8fe0f00c8b2e)

The project flow is:
1. Load Amazon reviews from CSV
2. Clean and prepare review text
3. Classify sentiment with OpenAI GPT-4o
4. Classify topics with `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`
5. Store processed outputs in CSV files
6. Serve the final analysis in a Streamlit dashboard backed by Plotly charts

## Technology Stack
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language Model** | OpenAI GPT-4o | Zero-shot sentiment classification |
| **Topic Classifier** | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` | Zero-shot topic labeling |
| **Data Processing** | Pandas 2.2.3, NumPy 1.26.4 | Data cleaning, grouping, and aggregation |
| **Visualization** | Plotly 5.24.1, Matplotlib 3.9.2 | Charts and exploratory analysis |
| **Dashboard** | Streamlit | Interactive analytics interface |
| **Notebook Workflow** | Jupyter Notebook | Reproducible experimentation and model execution |
| **API Integration** | OpenAI Python SDK 1.51.0 | Model access for sentiment classification |

## Project Structure
```bash
amazon-feedback-analysis/
├── data/
│   ├── amazon_reviews.csv          # Raw review dataset
│   ├── sentiment_classified.csv    # Reviews with sentiment labels
│   └── topics_classified.csv       # Reviews with sentiment + topic labels
├── dashboard.py                    # Streamlit entry point
├── dashboard_helpers.py            # Plotly chart builders
├── intro.ipynb                     # End-to-end exploratory workflow
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Getting Started
### Prerequisites
- **Python**: 3.11 or higher recommended
- **pip**: for dependency installation
- **OpenAI API key**: only required if you want to rerun the sentiment classification workflow from the notebook
- **Jupyter Notebook**: optional, for reproducing the modeling steps in `intro.ipynb`

### Installation
1. **Clone the repository**
```bash
git clone https://github.com/MaiconKevyn/amazon-feedback-analysis.git
cd amazon-feedback-analysis
```

2. **Create and activate a virtual environment**
```bash
python -m venv .venv

# On macOS/Linux
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install streamlit
```

The repository already includes the processed CSV outputs, so you can explore the dashboard immediately after installation.

### Configuration
If you want to reproduce the sentiment analysis step from the notebook, create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

This is not required to run the dashboard with the precomputed files in `data/`.

### Running the Project
1. **Launch the dashboard**
```bash
streamlit run dashboard.py
```

2. **Open the app**

Streamlit will expose the dashboard locally, typically at [http://localhost:8501](http://localhost:8501).

3. **Optional: reproduce the analysis workflow**

Open `intro.ipynb` in Jupyter and execute the notebook to inspect preprocessing, sentiment classification, topic classification, and chart generation.

## Dataset
The project uses the [Amazon reviews dataset on Kaggle](https://www.kaggle.com/datasets/tarkkaanko/amazon). In this repository, the analysis is centered on three fields from the source data:

- **`overall`**: product rating from 1 to 5
- **`reviewText`**: free-text user review
- **`reviewTime`**: date when the review was submitted

### Dataset Snapshot
- **Raw reviews**: 4,915
- **Classified reviews**: 4,914
- **Time span**: 2012-01-09 to 2014-12-07
- **Average rating**: 4.588
- **Average review length**: 50.45 words

These fields drive the core use case: detect sentiment, assign a topic, and then analyze how both dimensions evolve over time.

## Modeling Approach
### 1. Sentiment Analysis
The first stage uses **OpenAI GPT-4o** to classify each processed review into one of three sentiment labels:
- **Positive**
- **Neutral**
- **Negative**

This stage is implemented in the notebook workflow and persisted to `data/sentiment_classified.csv`.

### 2. Topic Classification
The second stage uses Hugging Face Transformers with [`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) to assign a single best-fit topic per review.

The current topic set is:
- **Product functionality**
- **Compatibility with devices**
- **Quality**
- **Price**
- **Delivery Service**

The final enriched output is stored in `data/topics_classified.csv`.

### 3. Aggregation and Visualization
After both labels are assigned, the project aggregates feedback to answer questions such as:
- Which topics receive the most mentions?
- Which topics carry the highest share of negative sentiment?
- How do sentiment and topic volumes change over time?
- What is the rating profile inside a selected date window?

## Dashboard and Results
### Dashboard Draft
The dashboard allows users to select a start and end date and instantly recalculate the visual summaries from the filtered review set.

![Dashboard Draft](https://github.com/user-attachments/assets/5a197fb5-7864-4349-a4c6-6e533872e81e)

### Included Views
- **Overall sentiment distribution**: bar and donut views of sentiment counts
- **Sentiment over time**: normalized sentiment trend lines by month
- **Topic distribution**: horizontal bar chart of topic frequency
- **Topic sentiment mix**: percentage distribution of sentiment inside each topic
- **Topic over time**: stacked temporal view of topic mentions

### Example Charts
**Overall sentiment distribution**

<img width="995" alt="Overall sentiment distribution" src="https://github.com/user-attachments/assets/275e9af2-7f32-49b5-814d-ff1da23ced41">

**Sentiment over time**

<img width="994" alt="Sentiment over time" src="https://github.com/user-attachments/assets/bba82c66-f512-450a-8bae-597eb6a48d1c">

**Topic distribution**

<img width="1200" alt="Topic distribution" src="https://github.com/user-attachments/assets/5c4a4104-06bf-4180-928c-18a0fa2293b0">

**Topic over time**

<img width="1577" alt="Topic over time" src="https://github.com/user-attachments/assets/10683bb1-1b95-4f83-a9f1-f53efe3285c0">

**Sentiment by topic**

<img width="996" alt="Sentiment by topic" src="https://github.com/user-attachments/assets/66c30f25-a689-40ff-9ff8-40e8d9a95d79">

## Key Findings
Based on the processed data included in this repository:

- **Sentiment distribution**: 3,665 positive, 751 neutral, 498 negative reviews
- **Most frequent topic**: `Product functionality` with 3,046 mentions
- **Second largest topic**: `Compatibility with devices` with 1,021 mentions
- **Smaller but meaningful clusters**: `Quality` (441), `Price` (299), `Delivery Service` (107)
- **Rating skew**: the dataset is strongly concentrated in 5-star reviews, which helps explain the dominant positive sentiment profile

The strongest practical value of the project is not just counting sentiment, but surfacing **which product themes deserve investigation first**.

## Limitations and Next Steps
- **Topic taxonomy is fixed**: a broader or more domain-specific topic list could improve classification quality
- **Single-label topic assignment**: some reviews may legitimately belong to multiple topics
- **Notebook-centric pipeline**: turning the notebook workflow into scripts would improve reproducibility
- **Dependency gap**: the dashboard depends on Streamlit, but the current `requirements.txt` does not list it yet
- **Model cost and latency**: rerunning sentiment classification with OpenAI scales with dataset size
- **OOD handling**: future iterations could reject reviews that do not fit any of the predefined topics

## License
This project is licensed under the **MIT License**.

See [LICENSE](LICENSE) for the full text.

Copyright (c) 2026 Maicon Kevyn


