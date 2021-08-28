from typing import List

import spacy
from spacy.lang.en import English

from joblib import Parallel, delayed


class Sentencizer:
    """Sentencizer compatible with multiprocessing.  
    References: 
        https://prrao87.github.io/blog/spacy/nlp/performance/2020/05/02/spacy-multiprocess.html#Option-2:-Use-nlp.pipe
        https://joblib.readthedcs.io/en/latest/generated/joblib.Parallel.html
    """
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
    

    def __call__(self, articles:List[str], n_jobs:int=5, chunk_size:int=5000):
        """
        Args:
            articles (List[str]): Wikipedia article data.
            n_jobs (int): how many jobs will you run simultaneously?
            chunk_size (int): how many articles per process?
        
        Returns:
            List[List[str]]: sentenceized articles
        """
        if len(articles) < chunk_size:
            chunk_size = len(articles)
        print(f"Estimated number of tasks: {int(len(articles) / chunk_size)}")
        return self.parallel_processor(articles, n_jobs, chunk_size)

    
    def flatten_list(self, lists:List[List[List[str]]]):
        """Flatten a list of lists to a combined list."""
        return [item for sublist in lists for item in sublist]
    

    def chunker(self, iterable:List[str], total_length:int, chunk_size:int):
        """Chunking raw data."""
        return (iterable[pos: pos + chunk_size] for pos in range(0, total_length, chunk_size))
    

    def split_article_into_sentences(self, articles:List[str]):
        """Split each article into sentences using spaCy."""
        ret = []
        for article in self.nlp.pipe(articles, batch_size=50):
            ret.append([sent.string.strip() 
                        for sent in list(article.sents)])
        return ret


    def parallel_processor(self, articles:List[str], n_jobs:int=5, chunk_size:int=1000):
        executor = Parallel(n_jobs=n_jobs, verbose=10, backend='multiprocessing', prefer="processes")
        do = delayed(self.split_article_into_sentences)
        tasks = (do(chunk) for chunk in self.chunker(articles, len(articles), chunk_size=chunk_size))
        ret = executor(tasks)
        return self.flatten_list(ret)
        