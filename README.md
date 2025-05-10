# AI Societal Constraints: YouTube Public Discourse Analysis

This repository contains the full code pipeline used for the project **"AI Societal Constraints in Public Discourse"**, focused on analyzing how users perceive artificial intelligence through large-scale YouTube comment analysis using Natural Language Processing (NLP), sentiment classification, and hybrid topic modeling.

> 🧠 This work is part of a PhD research project and was accepted for presentation at AIRSI 2025.

---

## 🔍 Objective

The project explores **how AI is framed in public discourse**, focusing on three key conceptual constraints:

- **Agency Transference**
- **Parametric Reductionism**
- **Regulated Expression**

We use a large corpus of user-generated YouTube comments to detect sentiment patterns and thematic clusters using modern NLP techniques.

---

## 🧪 Pipeline Overview

### 1. 🔴 **YouTube Data Collection**

- Search and collect metadata for videos on "artificial intelligence"
- Extract top-level comments, likes, dates, and geolocation (if available)

📁 File: `src/data_collection.py`

---

### 2. 🧼 **Text Preprocessing & Filtering**

- Clean and lemmatize text
- Convert emojis to text
- Detect and retain only English content
- Remove short or duplicate comments

📁 File: `src/preprocess_filter.py`

---

### 3. 💬 **Sentiment Analysis**

- Use pre-trained multilingual BERT model (`nlptown/bert-base-multilingual-uncased-sentiment`)
- Analyze both **comment text** and **video descriptions**
- Output sentiment label and confidence score

📁 File: `src/sentiment_analysis.py`

---

### 4. 🧠 **Hybrid Topic Modeling**

- Use **Word2Vec + Agglomerative Clustering** to assign semantic groupings
- Use **BERTopic** for probabilistic topic extraction
- Merge both into hybrid topic clusters

📁 File: `src/topic_modeling.py`

---

## 📊 Technologies Used

- Python
- YouTube Data API v3
- Hugging Face Transformers
- BERTopic, Gensim, scikit-learn
- NLTK, langdetect, tqdm, pandas, emoji

---


> 🔒 **Note**: Due to GitHub’s file size limits, large `.csv` files have been excluded from this repository. You can request access to the dataset or download it via a shared academic data repository (link coming soon).

---

## 📖 Citation

If you use this codebase in your work, please cite:

Moutinho, R., Rohden, S. F., & Pinto, D. C. (2025).
AI Societal Constraints in Public Discourse: Insights from Advanced NLP and Topic Modeling.
AIRSI 2025 Conference Proceedings.


---

## 📬 Contact

- **Rodrigo Moutinho**  
  PhD Candidate, NOVA IMS 
  20220362@novaims.unl.pt 
  [rodmoutinho.github.io](https://github.com/rodmoutinho)

---

## ✅ License

MIT License

