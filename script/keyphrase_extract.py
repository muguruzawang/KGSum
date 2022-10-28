# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:02:51 2020
@author: Muguruza
"""

from rake import Rake
import pandas as pd
from functools import reduce
import operator
from  Clustering_experiment import get_entities, get_entity_index,get_entity_embedding
import os
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import Scibert_for_Clustering
import torch
from nltk.tokenize import sent_tokenize
import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
#from nlp-rake import rake
stoppath = '../RAKE-tutorial/data/stoplists/SmartStoplist.txt'


#r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.

def get_fulltext_entities(path):
    df = pd.read_json(path,lines=True)
    contents = {}
    entity_dic = []
    for doc,words,ners in zip(df.doc_id,df.words,df.ner):
        dic = {}
        dic['Task'] = []
        dic['Method'] = []
        dic['Material'] = []
        dic['Metric'] = []
        sentences_id = {}
        for ner in ners:
            phrase = [words[index] for index in range(int(ner[0]),int(ner[1]))]
            
            phrase = ' '.join(phrase)
            
            if ner[2] not in dic:
                dic[ner[2]] = []
                dic[ner[2]].append(phrase)
            else:
                dic[ner[2]].append(phrase)
        entity_dic.append(dic)
    
    return entity_dic

def get_entity_corresponding_sentence(sentences,entity):
    sentence_list = sent_tokenize(sentences)
    for sentence in sentence_list:
        if sentence.find(entity+' ') != -1:
            return sentence
    return ' '

def Lemma_word(sentence):
    tokens = wordpunct_tokenize(sentence)
    tagged_sent = nltk.pos_tag(tokens)

    wnl = WordNetLemmatizer()
    word_list = []
    for tag in tagged_sent:
        if tag[1].startswith('J'):
            wordnet_pos = wordnet.ADJ
        elif tag[1].startswith('V'):
            wordnet_pos = wordnet.VERB
        elif tag[1].startswith('N'):
            wordnet_pos = wordnet.NOUN
        elif tag[1].startswith('R'):
            wordnet_pos = wordnet.ADV
        else:
            wordnet_pos = wordnet.NOUN
        word_list.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
    return word_list

def get_content(path):
    df = pd.read_json(path,lines=True)
    df = df.sort_values("doc_key")
    
    contents = []
    for doc,sentences,ners in zip(df.doc_key,df.sentences,df.predicted_ner):
        sentences = reduce(operator.add,sentences)
        sentences = ' '.join(sentences)
        
        contents.append(sentences)
    return contents

def get_most_important_entity_RAKE(entity_dic,contents,entity_type):

    path_w = r'./RAKE_score.txt'
    f = open(path_w,'w',encoding = 'utf-8')
    for index ,(dic,content) in enumerate(zip(entity_dic,contents)):
        print(dic)
        
        sentence_list = sent_tokenize(content)
        entities = dic[entity_type]
        r = Rake()
        r.extract_keywords_from_text(content)
        
        if len(entities) != 0:
            scores = {}
            
            if len(sentence_list) == 1:
                for entity in entities:
                    score = 0
                    lemma_ents = Lemma_word(entity)
                    for ent in lemma_ents:
                        try:
                            score += r.get_word_score(ent)
                        except:
                            score += 0
                    scores[entity] = score
                    f.write(str(index)+'\n')
                    f.write(str(entity)+ '  '+ str(score)+'   ')
                entity_dic[index][entity_type] = sorted(scores.items(),key = lambda item: item[1],reverse = True)[0][0]
            else:
                for entity in entities:
                    if sentence_list[0].find(entity) != -1:
                        score = 200
                    elif sentence_list[1].find(entity) != -1:
                        score = 100
                    else:
                        score = 0
                    lemma_ents = Lemma_word(entity)
                    for ent in lemma_ents:
                        try:
                            score += r.get_word_score(ent)
                        except:
                            score += 0
                    scores[entity] = score
                    f.write(str(index)+'\n')
                    f.write(str(entity)+ '  '+ str(score)+'   ')
                entity_dic[index][entity_type] = sorted(scores.items(),key = lambda item: item[1],reverse = True)[0][0]
            
        f.write('\n')

    f.close()

    return entity_dic

def get_most_important_entity_TextrankEntity(entity_dic,entity_dic_full,entity_type):
    path_w = r'./TextrankEntity_score2.txt'
    path_temp = r'./temp_sentence.txt'
    LANGUAGE = "english"
    f = open(path_w,'w',encoding = 'utf-8')
    
    for index ,(dic_abs, dic_full) in enumerate(zip(entity_dic,entity_dic_full)):
        entities_abs = dic_abs[entity_type]
        entities_full = dic_full[entity_type]
        
        sentences = entities_abs + entities_full
                    
        with open(path_temp,'w',encoding = 'utf-8') as g:
            for i in sentences:
                g.write(i+'.\n')
        
        parser = PlaintextParser.from_file(path_temp, Tokenizer("english"))
        stemmer = Stemmer('en')

        summarizer = Summarizer(stemmer)
        summarizer.stop_words = get_stop_words('english')
        
        SENTENCES_COUNT = len(sentences)
        ranked_sentences = summarizer(parser.document, SENTENCES_COUNT)
        #print(ranked_sentences)
        ranked_abs = [ent._text[:-1] for ent in ranked_sentences if ent._text[:-1] in entities_abs]
        ranked = [ent._text[:-1] for ent in ranked_sentences]

        '''
        f.write(str(index)+'#################\n')
        for sen in ranked_abs:
            f.write(str(sen)+'\n')
        
        f.write('-------------------------------\n')
        for sen in ranked_sentences:
            f.write(str(sen)+'\n')
        '''
        f.write(str(index+1)+'#################\n')
        if len(ranked_abs) ==0:
            f.write('[]\n')
            if len(ranked) == 0:
                entity_dic[index][entity_type] = []    
            else:
                entity_dic[index][entity_type] = ranked[0]
        else:
            f.write(str(ranked_abs[0])+'\n')
            entity_dic[index][entity_type] = ranked_abs[0]
        os.remove(path_temp)
    f.close()
    return entity_dic

def get_most_important_entity_BERT(entity_dic,corpus,entity_type):
    for index_article, (dic, sentences) in enumerate(zip(entity_dic, corpus)):

        tokenized_text, tokenized_text_dic, token_vecs_cat,token_vecs_sum, article_embedding = Scibert_for_Clustering.get_scibert_embedding(sentences)
        entities = dic[entity_type]

        print(tokenized_text_dic)

        if len(entities) != 0:
            scores = {}
            for entity in entities:
                sentence = get_entity_corresponding_sentence(sentences,entity)
                tokenized_text, tokenized_text_dic, token_vecs_cat,token_vecs_sum, sentence_embedding = Scibert_for_Clustering.get_scibert_embedding(sentence)
            
                score = 0
            
                print('current entity: %s'%entity)
                print('current sentence: %s'%sentence)
                indexs_task = get_entity_index(tokenized_text,tokenized_text_dic,entity)
                embedding_task = get_entity_embedding(indexs_task,token_vecs_sum)

                score = torch.cosine_similarity(article_embedding, embedding_task, dim=0)

                scores[entity] = score
            entity_dic[index_article][entity_type] = sorted(scores.items(),key = lambda item: item[1],reverse = True)[0][0]
    
    return entity_dic


if __name__ == '__main__':
    path_folders = r'../dataset'
    folder_indexs  = os.listdir(path_folders)
    
    for folder_index in folder_indexs[:5]:
        print(folder_index)
        path_folder = os.path.join(path_folders,folder_index)
        path_entity = os.path.join(path_folder,folder_index+r'_output_Abstract_set.json')
        entity_dic,corpus = get_entities(path_entity)
        path_write = os.path.join(path_folder,r'entity_task_RAKE_full.txt')

        #path_docs = os.path.join(path_folder,r'reference papers')
        path_docs = os.path.join(path_folder,folder_index+'_Abstract_set')

        path_fulltext_entity = os.path.join(path_folder,r'./predictedner_bodytext_corpus_survey.jsonl')
        entity_dic_full = get_fulltext_entities(path_fulltext_entity)
        #entity_dic = get_most_important_entity_RAKE(entity_dic,path_docs,'Task')

        entity_type = ['Task','Method','Material','Metric']
        path_write = os.path.join(path_folder,r'entity_tuples_textrank.txt')
        entity_tuple = []
        with open(path_write,'w',encoding = 'utf-8') as f:
            for index1,_type in enumerate(entity_type):
                entity_dic = get_most_important_entity_TextrankEntity(entity_dic,entity_dic_full,_type)
                #entity_dic = get_most_important_entity_BERT(entity_dic,corpus,'Task')

                for index2,dic2 in enumerate(entity_dic):
                    if index1 == 0:
                        entity_tuple.append([])
                    
                    entity_tuple[index2].append(dic2[_type])

            for index,tup in enumerate(entity_tuple):
                f.write(str(index+1)+'\n')
                f.write(str(tup))
                f.write('\n')
                
             