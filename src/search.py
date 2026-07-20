from .preprocessing import preprocess
from .filter import invertedindex
from .retrieval import retrievalengine

df = preprocess()
index= invertedindex(df)
engine = retrievalengine(df)

def search(query):
    cand = index.search(query)

    if not cand: 
        return df.iloc[0:0].copy()
     
    result = engine.rank(query,cand)

    return result

