from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import clean_text

class retrievalengine:

    def __init__(self,df):
        self.df = df

        self.vector = TfidfVectorizer(ngram_range=(1,2), min_df=2)
        self.matrix = self.vector.fit_transform(df["document"])

    def rank(self,query,cand):
        query = clean_text(query)

        qvector =  self.vector.transform([query])
        candvec = self.matrix[cand]

        score = cosine_similarity(qvector,candvec).flatten()

        order = score.argsort()[::-1][:10]

        ranked_ids = [cand[i] for i in order]

        result = self.df.iloc[ranked_ids].copy()

        result["similarity_score"] = score[order]

        return result
    
    
