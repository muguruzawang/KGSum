# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:02:51 2020

@author: Muguruza
"""

from rake import Rake
import pandas as pd
from functools import reduce
import operator
import os
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import json

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

def get_entities(line):    
    dic = {}
    dic['Task'] = []
    dic['Method'] = []
    dic['Material'] = []
    dic['Metric'] = []
    dic['OtherScientificTerm'] = []
    dic['Generic'] = []
    
    metadata_dict = json.loads(line)
    sentences = metadata_dict['sentences']
        
    ners = metadata_dict['predicted_ner']
    doc = metadata_dict['doc_key']

    index = 0
    sentences_id = {}
    for sentence in sentences:
        for word in sentence:
            sentences_id[index] = word
            index += 1
    
    if len(ners) == 0:
        return [], None, doc
    
    ners = reduce(operator.add, ners)
    
    for ner in ners:
        phrase = [sentences_id[index] for index in range(int(ner[0]),int(ner[1])+1)]
        
        phrase = ' '.join(phrase).lower()
        
        if ner[2] not in dic:
            dic[ner[2]] = []
        
        dic[ner[2]].append(phrase)
    

    sentences = reduce(operator.add,sentences)
    sentences = ' '.join(sentences).lower()

    return dic, sentences, doc

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

def get_most_important_entity_RAKE(entity_dic,content,entity_type):
    entities = entity_dic[entity_type]

    r = Rake()
    r.extract_keywords_from_text(content)
    
    scores = {}
    if len(entities) != 0:
        scores = {}
        
        for entity in entities:
            lemma_ents = Lemma_word(entity)
            score = 0
            for ent in lemma_ents:
                try:
                    score += r.get_word_score(ent)
                except:
                    score += 0
            scores[entity] = score
        
        max_value = sorted(scores.items(),key = lambda item: item[1],reverse = True)[0][1]
        if max_value == 0:
            max_value = 10e20
        if entity_type == 'Generic':
            for ent in scores.keys():
                scores[ent] = scores[ent]/(max_value*2)
        else:
            for ent in scores.keys():
                scores[ent] = scores[ent]/max_value
        #entity_dic = sorted(scores.items(),key = lambda item: item[1],reverse = True)
        entity_dic[entity_type] = scores
        
    return scores, entity_dic

if __name__ == '__main__':
    # path_json is the path to file of extracted entities and relations by DYGIE++, each line of ***.json is the result of a single paper.
    path_json = r'F:\From now to 2023\Survey Generation\Multi_XScience\multi_xscience_perpaper\output_train_processed_coref.json'
    # path_write is the path to write importance_score.json
    path_write = r'F:\From now to 2023\Survey Generation\Multi_XScience\cache\multi_xscience\val.train_importance_score.json'
    types = ['Task','Method','Material','OtherScientificTerm','Metric', 'Generic']
    
    entitys_dic = []
    #entitys_dic_type = []
    with open(path_json,'r',encoding='utf-8') as f:
        lines = f.readlines()
        now_key = 19541
        total_dic = {}
        for line in tqdm(lines):
            entity_dic, corpus, key = get_entities(line)
            
            if key != now_key:
                entitys_dic.append(total_dic)
                total_dic = {}
                now_key = key
            
            if entity_dic == []:
                continue
            
            for _type in types:
                scores,entity_dic = get_most_important_entity_RAKE(entity_dic,corpus,_type)
                total_dic.update(scores)
        
        entitys_dic.append(total_dic)
    
    with open(path_write,'w',encoding = 'utf-8') as g:
        for _dict in entitys_dic:
             g.write(json.dumps(_dict) + "\n")
            