from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering
from bertopic import BERTopic
import pandas as pd
import numpy as np
from collections import Counter

# File paths
input_file = "C:\\Users\\isabe\\Documents\\Phd\\Text Mining\\data\\youtube_comments_with_BERT_sentiments.csv"
output_words_file = "C:\\Users\\isabe\\Documents\\Phd\\Text Mining\\data\\hybrid_word2vec_bertopic_topics.csv"
output_topics_file = "C:\\Users\\isabe\\Documents\\Phd\\Text Mining\\data\\youtube_comments_with_hybrid_topics.csv"

# Load dataset
df = pd.read_csv(input_file)

# Prepare text data
comments = df["comment"].dropna().tolist()
tokenized_comments = [comment.split() for comment in comments]

# Train Word2Vec model
print("\n🔄 Training Word2Vec model...")
word2vec_model = Word2Vec(sentences=tokenized_comments, vector_size=100, window=5, min_count=5, workers=4)

# Get word embeddings
word_vectors = word2vec_model.wv
word_list = word_vectors.index_to_key
word_embeddings = np.array([word_vectors[word] for word in word_list])

# Perform hierarchical clustering
print("\n🔄 Performing Hierarchical Clustering...")
n_clusters = 5  # Define number of main topics
cluster_model = AgglomerativeClustering(n_clusters=n_clusters)
clusters = cluster_model.fit_predict(word_embeddings)

# Map words to topics
word_topics = {word: clusters[idx] for idx, word in enumerate(word_list)}

def assign_comment_topic(comment):
    words = comment.split()
    topic_counts = Counter([word_topics[word] for word in words if word in word_topics])
    return topic_counts.most_common(1)[0][0] if topic_counts else -1

# Assign topics to comments using Word2Vec + Clustering
df["word2vec_topic"] = df["comment"].apply(assign_comment_topic)

# Extract the most important words per topic
def get_top_words(word_topics, top_n=10):
    """Selects the top N most relevant words for each topic."""
    topic_words = {}
    for word, topic in word_topics.items():
        if topic not in topic_words:
            topic_words[topic] = []
        topic_words[topic].append(word)
    
    # Keep only the top N words per topic
    for topic in topic_words:
        topic_words[topic] = topic_words[topic][:top_n]
    
    return topic_words

# Get top words per topic
top_topic_words = get_top_words(word_topics)

# Train BERTopic model
print("\n🔄 Training BERTopic model...")
topic_model = BERTopic()
bertopic_topics, probs = topic_model.fit_transform(comments)

# Assign BERTopic topics to comments
df["bertopic_topic"] = bertopic_topics

# Save topics and subtopics
print("\n💾 Saving Hybrid Word2Vec + BERTopic topics to CSV...")
topic_data = pd.DataFrame([(topic, ', '.join(words)) for topic, words in top_topic_words.items()], columns=["Topic", "Top_Words"])
topic_data.to_csv(output_words_file, index=False)

# Save comments with topic assignments
df.to_csv(output_topics_file, index=False, encoding="utf-8")

print(f"\n✅ Hybrid Topic Modeling (Word2Vec + BERTopic) completed! Results saved to:\n- Topics/Subtopics Words: {output_words_file}\n- Comments with Topics: {output_topics_file}")
