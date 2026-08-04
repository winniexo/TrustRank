# TrustRank

An Information Retrieval (IR) and Machine Learning based search engine that retrieves Amazon products and evaluates the trustworthiness of their reviews. TrustRank combines classical IR techniques with a Logistic Regression classifier to identify suspicious reviews and improve the ranking of search results.

---

## Features

- 🔍 Product search using TF-IDF and Cosine Similarity
- 📂 Fast retrieval using an Inverted Index
- 🤖 Fake review detection using Logistic Regression
- ⭐ Trust score based product re-ranking
- 🎚️ Rating-based filtering
- 💻 Interactive Streamlit web interface

---

## Tech Stack

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Joblib

---

## Project Structure

```text
TrustRank/
│
├── app.py
├── .gitignore
│
├── assets/
│   ├── __init__.py
│   ├── components.py
│   └── style.css
│
├── data/
│   ├── amazon.csv
│   ├── labeled_reviews.csv
│   └── labeled_reviews_text.csv
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
└── src/
    ├── __init__.py
    ├── classifier.py
    ├── features.py
    ├── filter.py
    ├── labeler.py
    ├── preprocessing.py
    ├── retrieval.py
    ├── search.py
    └── trust.py
```

---

## Information Retrieval Techniques

The retrieval engine is built using classical Information Retrieval techniques, including:

- Text Preprocessing
- Inverted Index Construction
- TF-IDF Vectorization
- Cosine Similarity
- Document Ranking
- Product Re-ranking

---

## Machine Learning

TrustRank uses a Logistic Regression classifier to classify reviews as **Genuine** or **Suspicious**.

The classifier is trained on labeled review data and uses extracted review features to predict the credibility of reviews. These predictions are then incorporated into the search ranking process, allowing products with more trustworthy reviews to appear higher in the results.

---

## Dataset

The project uses an Amazon product review dataset containing:

- Product Name
- Category
- Product Description
- Product Rating
- Review Title
- Review Content
- Product Links
- Product Images

Additional labeled datasets are used to train the Logistic Regression classifier.

---

## Installation

Clone the repository.

```bash
git clone https://github.com/winniexo/TrustRank.git
```

Navigate to the project directory.

```bash
cd TrustRank
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

Run the Streamlit application.

```bash
streamlit run app.py
```

---

## How It Works

1. The user enters a product search query.
2. The query is preprocessed.
3. An Inverted Index retrieves relevant candidate products.
4. TF-IDF vectorization and Cosine Similarity compute relevance scores.
5. Review features are extracted.
6. The Logistic Regression classifier predicts whether reviews are genuine or suspicious.
7. A trust score is generated and combined with the retrieval score.
8. Products are re-ranked according to both relevance and review credibility.
9. The final ranked products are displayed through the Streamlit interface.

---

## Future Enhancements

- Deep learning based review classification
- Sentiment analysis integration
- Personalized search recommendations
- Explainable AI for trust prediction
- Deployment on Streamlit Community Cloud

---
