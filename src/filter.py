from collections import defaultdict
from .preprocessing import clean_text

class invertedindex:

    def __init__(self,df):
        self.df = df
        self.index = self.invindex()
    
    
    def invindex(self):
        invind = defaultdict(set)

        for docid,doc in enumerate(self.df["document"]):
            words = clean_text(doc).split()

            for w in words:
                invind[w].add(docid) 
        
        return dict(invind)
    


    def search(self,query):
        query = clean_text(query)
        words = query.split()

        matches = defaultdict(int)

        for word in words:

            for docid in self.index.get(word, set()):
                matches[docid] += 1

        cand = sorted(
        matches,
        key=matches.get,
        reverse=True
)


        return cand
        
