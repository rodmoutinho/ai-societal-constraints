import pandas as pd
import re
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import emoji
import logging
import nltk
import os

# Download NLTK resources
nltk.data.path.append('C:/Users/isabe/nltk_data')
nltk.download('punkt', download_dir='C:/Users/isabe/nltk_data')
nltk.download('wordnet', download_dir='C:/Users/isabe/nltk_data')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Text cleaning functions
def truncate_text(text, word_limit):
    words = text.split()
    return ' '.join(words[:word_limit])

def clean_text(text):
    try:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
    except Exception as e:
        logging.error(f"Error during text cleaning: {e}")
        return text

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def convert_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))

def process_chunk(chunk, start_date=None, end_date=None):
    logging.info(f"Processing chunk with {len(chunk)} rows")
    
    # Ensure datetime parsing
    chunk['comment_date'] = pd.to_datetime(chunk['comment_date'], errors='coerce')
    chunk['video_published_date'] = pd.to_datetime(chunk['video_published_date'], errors='coerce')

    # Filter by date range
    if start_date and end_date:
        before = len(chunk)
        chunk = chunk[(chunk['comment_date'] >= start_date) & (chunk['comment_date'] <= end_date)]
        logging.info(f"Filtered by date range: {before} -> {len(chunk)} rows")

    # Preserve original comment and description
    chunk['original_comment'] = chunk['comment']
    chunk['original_description'] = chunk['description']

    # Text preprocessing
    chunk['comment'] = chunk['comment'].fillna('')
    chunk['description'] = chunk['description'].fillna('')

    chunk.loc[:, 'comment'] = chunk['comment'].apply(lambda x: truncate_text(str(x), 100))
    chunk.loc[:, 'description'] = chunk['description'].apply(lambda x: truncate_text(str(x), 100))

    chunk.loc[:, 'comment'] = chunk['comment'].apply(convert_emojis)
    chunk.loc[:, 'description'] = chunk['description'].apply(convert_emojis)

    chunk.loc[:, 'comment'] = chunk['comment'].apply(clean_text)
    chunk.loc[:, 'description'] = chunk['description'].apply(clean_text)

    # Filter non-English content
    before = len(chunk)
    chunk = chunk[chunk['comment'].apply(is_english) & chunk['description'].apply(is_english)]
    logging.info(f"Filtered non-English comments: {before} -> {len(chunk)} rows")

    # Remove videos with zero views
    before = len(chunk)
    chunk = chunk[chunk['view_count'] > 0]
    logging.info(f"Filtered videos with >0 views: {before} -> {len(chunk)} rows")

    # Remove duplicates
    before = len(chunk)
    chunk = chunk.drop_duplicates(subset=['comment'])
    logging.info(f"Removed duplicate comments: {before} -> {len(chunk)} rows")

    # Include shorter comments (optional)
    before = len(chunk)
    chunk = chunk[chunk['comment'].str.split().str.len() >= 5]  # Allow shorter comments
    logging.info(f"Filtered short comments: {before} -> {len(chunk)} rows")

    # Add unique ID for each comment
    chunk['comment_id'] = range(1, len(chunk) + 1)

    return chunk

# File paths and parameters
input_file = "C:\\Users\\isabe\\Documents\\Phd\\Text Mining\\data\\youtube_comments_with_likes.csv"
output_file = "C:\\Users\\isabe\\Documents\\Phd\\Text Mining\\data\\filtered_youtube_comments_with_ids.csv"
chunk_size = 10000
processed_chunks = []

# Date range
start_date = '2010-01-01'
end_date = '2024-12-31'

# Process dataset
logging.info("Starting dataset processing")
for chunk in pd.read_csv(input_file, chunksize=chunk_size):
    processed_chunk = process_chunk(chunk, start_date=start_date, end_date=end_date)
    processed_chunks.append(processed_chunk)

# Combine processed chunks
df_filtered = pd.concat(processed_chunks, ignore_index=True)

# Save results
number_of_videos = len(df_filtered['video_id'].unique())
number_of_comments = len(df_filtered)
logging.info(f"Number of videos with more than 0 views: {number_of_videos}")
logging.info(f"Number of unique comments after filtering: {number_of_comments}")

df_filtered.to_csv(output_file, index=False, encoding='utf-8')
logging.info(f"Filtered comments saved to '{output_file}'")
logging.info(f"Preview of filtered data:\n{df_filtered.head()}")
