from pathlib import Path
import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "amazon.csv"

TEXT_COLUMNS = [
    "product_name",
    "category",
    "about_product",
    "review_title",
    "review_content",
]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    words = text.split()

    words = [
        word
        for word in words
        if word not in ENGLISH_STOP_WORDS
    ]
    
    text = " ".join(words)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess():
   
   CSV_PATH = BASE_DIR / "data" /"amazon.csv"
   df = pd.read_csv(CSV_PATH)
   
   df = df.fillna("")
   df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

   df = df.dropna(subset=["rating"])

   df["document"] = (
    df["product_name"] + " " +
    df["product_name"] + " " +
    df["product_name"] + " " +

    df["category"] + " " +
    df["category"] + " " +

    df["about_product"] + " " +
    df["review_title"] + " " +
    df["review_content"]
   )
    

   
   df = df.reset_index(drop=True)
   
   return df

