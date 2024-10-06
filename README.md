# Feedback Analysis with Zero-Shot Learning
This project focuses on analyzing customer reviews from an **Amazon product reviews dataset**. The main goal is to classify both the **sentiments** and **topics** from user feedback to better understand which topics are most critical to the application’s performance. The deliverable is an interactive **Streamlit dashboard** that presents all the generated results, including sentiment distribution and topic analysis, allowing for a clear visualization of the insights derived from the data.

## Workflow Overview
![image](https://github.com/user-attachments/assets/0416d8de-bd33-4dad-b45a-8fe0f00c8b2e)

## Dataset
The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/tarkkaanko/amazon). This dataset addresses some of the most important problems in e-commerce, such as the correct calculation of product ratings after sales and the accurate ordering of product comments. Solving these problems helps improve customer satisfaction, ensures product prominence for sellers, and provides a seamless shopping experience for buyers. Additionally, it helps prevent misleading comments from impacting customer decisions and causing financial or customer losses. By addressing these issues, e-commerce sites and sellers can increase sales, while customers enjoy a smoother purchasing journey.

In this project, we specifically use the following columns from the dataset:
- **overall**: the product rating (ranging from 1 to 5).
- **reviewText**: the textual feedback from customers.
- **reviewTime**: the date when the feedback was submitted.

These columns are crucial for performing sentiment and topic classification, as well as for filtering data by date in the dashboard to analyze trends over time.

## Installation

Before running the project, ensure you have all the required dependencies installed. These are listed in the `requirements.txt` file. To install them, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository

2. Install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt

## Approach

1. **Sentiment Analysis**:  
   Initially, I applied **zero-shot learning** to classify the sentiment of the comments. For this step, I used OpenAI's **GPT-4o** model to categorize the reviews into sentiments such as:
   - Positive
   - Neutral
   - Negative

2. **Topic Classification**:  
   After classifying the sentiments, I proceeded to classify the topics within the comments. To diversify the approach, I used a different **LLM model** fine-tuned for zero-shot classification: [**MoritzLaurer/deberta-v3-large-zeroshot-v2.0**](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) from Hugging Face. This model helped identify topics such as:
   - Product Functionality
   - Price
   - Quality
   - Delivery Service
    - Compatibility with devices

3. **Aggregation and Analysis**:  
   Once both sentiment and topic classifications were completed, I grouped the data by topic to analyze the **sentiment distribution per topic**. This allowed me to understand which topics were the most critical and negatively impacted the user experience.

## Technologies Used

- **OpenAI GPT-4o** for sentiment analysis.
- **Hugging Face Transformers**: "MoritzLaurer/deberta-v3-large-zeroshot-v2.0" for topic classification.
- **Pandas** for data manipulation and aggregation.
- **Streamlit** for dashboard creation (optional for visualization).

## Results

The project successfully identified key topics that significantly impacted the application based on customer feedback sentiment. By combining zero-shot learning and prompt engineering for both sentiment and topic classification, this approach provided deep insights into areas that require improvement to enhance the overall user experience.


### Dashboard Draft
In the **Streamlit dashboard**, users can interactively filter the data by selecting a specific date range. This functionality allows for more granular analysis, enabling the exploration of trends and patterns within user feedback over custom time periods. By adjusting the date range, users can focus on feedback from specific periods, making it easier to identify changes in sentiment or topic relevance over time.

![image](https://github.com/user-attachments/assets/5a197fb5-7864-4349-a4c6-6e533872e81e)

### Sentiment Analysis
**Overall sentiment distribution**
<img width="995" alt="image" src="https://github.com/user-attachments/assets/275e9af2-7f32-49b5-814d-ff1da23ced41">

**Sentiment overtime**
<img width="994" alt="image" src="https://github.com/user-attachments/assets/bba82c66-f512-450a-8bae-597eb6a48d1c">


### Topic Classification
**Topics distributions**
<img width="1200" alt="image" src="https://github.com/user-attachments/assets/5c4a4104-06bf-4180-928c-18a0fa2293b0">

**Topics Overtime**
<img width="1577" alt="image" src="https://github.com/user-attachments/assets/10683bb1-1b95-4f83-a9f1-f53efe3285c0">

### Sentiment by Topic 
<img width="996" alt="image" src="https://github.com/user-attachments/assets/66c30f25-a689-40ff-9ff8-40e8d9a95d79">

