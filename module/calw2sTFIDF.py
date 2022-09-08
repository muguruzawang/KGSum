import sys 
sys.path.append("..") 
import os
import json
from tqdm import tqdm
import pdb

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from nltk.corpus import stopwords
import pickle
import numpy as np
from textrank import Textrank

def GetType(path):
    filename = path.split("/")[-1]
    return filename.split(".")[0]

def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res

def get_tfidf_embedding(text,stopwords):
    """
    
    :param text: list, sent_number * word
    :return: 
        vectorizer: 
            vocabulary_: word2id
            get_feature_names(): id2word
        tfidf: array [sent_number, max_word_number]
    """
    vectorizer = CountVectorizer(lowercase=True,stop_words = stopwords)
    word_count = vectorizer.fit_transform(text)
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(word_count)
    tfidf_weight = tfidf.toarray()
    return vectorizer, tfidf_transformer
    
def compress_array(a, id2word):
    """
    
    :param a: matrix, [N, M], N is document number, M is word number
    :param id2word: word id to word
    :return: 
    """
    d = {}
    for i in range(len(a)):
        d[i] = {}
        for j in range(len(a[i])):
            if a[i][j] != 0:
                d[i][id2word[j]] = a[i][j]
    return d
    

def main():
    types = ['train', 'val', 'test']
    path = r'/data/run01/scv0028/wpc/survey_generation/HeterEnReTSumGraph_ABS/datasets/multi_xscience'
    path_ent = r'/data/run01/scv0028/wpc/survey_generation/HeterEnReTSumGraph_ABS/cache/multi_xscience'


    textrank = Textrank()
    stopword = stopwords.words('english')
    for _type in types:
        path_json = os.path.join(path,_type+'.label.jsonl')
        path_jsonent = os.path.join(path_ent,_type+'.ent_type_relation.jsonl')

        with open(path_json,'r',encoding='utf-8') as f, open(path_jsonent,'r',encoding='utf-8') as g:
            lines = f.readlines()
            linesent = g.readlines()
            for line,lineent in tqdm(zip(lines, linesent)):
                linedict =  json.loads(line)
                text = linedict['text']
                corpus = []
                text = sum(text, [])
                    
                cntvector, tfidf_transformer = get_tfidf_embedding(text, stopword)
    
                linedict =  json.loads(lineent)
                entities = linedict['entities']
                entity_corpus = []
                for key in entities:
                    for x in entities[key]:
                        entity_corpus.append(x)

                remove_index = []
                entity_tfidf = tfidf_transformer.transform(cntvector.transform(entity_corpus)).toarray()
                for i in range(len(entity_corpus)):
                    if np.dot(entity_tfidf[i],entity_tfidf[i]) == 0:
                        remove_index.append(i)

                entity_corpus = np.delete(np.array(entity_corpus),remove_index)
                entity_tfidf = np.delete(entity_tfidf,remove_index, axis=0)
                scores = textrank(entity_corpus,entity_tfidf)
if __name__ == '__main__':
    main()
        
