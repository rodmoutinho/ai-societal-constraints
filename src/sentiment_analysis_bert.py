import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

# Load the sentiment analysis pipeline using a pre-trained BERT model
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def analyze_sentiment_bert(text):
    """Analyze sentiment using a BERT model and return label & confidence score."""
    if pd.isna(text) or text.strip() == "":
        return "neutral", 0.0  # Handle empty text cases
    result = sentiment_analyzer(text[:512])[0]  # BERT has a max token limit of 512
    return result["label"], result["score"]

# File paths
input_file = "C:\\Users\\isabe\\Documents\\Phd\\Text Mining\\data\\filtered_youtube_comments_with_ids.csv"
output_file = "C:\\Users\\isabe\\Documents\\Phd\\Text Mining\\data\\youtube_comments_with_BERT_sentiments.csv"

# Load data
df = pd.read_csv(input_file)

# Apply sentiment analysis with a progress bar using tqdm
tqdm.pandas(desc="Processing Comments")

df["comment_sentiment"], df["comment_confidence"] = zip(*df["comment"].progress_apply(lambda x: analyze_sentiment_bert(str(x))))
df["description_sentiment"], df["description_confidence"] = zip(*df["description"].progress_apply(lambda x: analyze_sentiment_bert(str(x))))

# Save results
df.to_csv(output_file, index=False, encoding='utf-8')

print(f"✅ Sentiment analysis completed! Results saved to: {output_file}")
