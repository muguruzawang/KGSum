import os
import argparse
import json

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

import pandas as pd
import operator
from functools import reduce
import os
import json
from tqdm import tqdm
import spacy
from nltk.corpus import stopwords

def get_relation_vocabulary(e,relVocab):
    relations = e['relations']
    clusters = e['clusters']
    entitys = e['entities']
    for r in relations:
        ent1 = r[0].lower().split()
        ent2 = r[2].lower().split()
        if ent1 in self.raw_ent_text and ent2 in self.raw_ent_text:
            self.raw_rel.append([ent1, r[1], ent2])

    for cluster in clusters:
        cluster = list(set(cluster))
        combs = list(combinations(cluster,2))
        for comb in combs:
            comb1 = comb[0].lower().split()
            comb2 = comb[1].lower().split()
            if comb1 in self.raw_ent_text and comb2 in self.raw_ent_text: 
                self.raw_rel.append([comb1, 'Coreference', comb2])
    raw_ent_text = []
    for key in entities:
        for x in entities[key]:
            e = at_least(x.lower().split())
            if e not in self.raw_ent_text:
                self.raw_ent_text.append(e)
        
        for phrase in phrases:
            if phrase not in entityVocab:
                entityVocab[phrase] = 1
                entity_list[index].append(phrase)
            else:
                entityVocab[phrase] += 1
                entity_list[index].append(phrase)
            
    return entity_list, entityVocab
        
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='multi_xscience', help='dataset name')
    parser.add_argument('--data_path', type=str, default='/data/home/scv0028/run/wpc/survey_generation/HeterEnReTSumGraph/cache', help='dataset name')
    args = parser.parse_args()

    save_dir = os.path.join(args.data_path, args.dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    datatypes = ["train","val","test"]

    ###这个变量用来存储所有的entity和id的对应关系，是全局变量
    entityVocab = {}

    entityVocabPath = os.path.join(save_dir,'RelVocab')
    for _type in datatypes:
        fname = _type + ".ent_type_relation.jsonl"
        
        path_json = os.path.join(args.data_path, jname)

        #dockey从1开始计数
        now_key = 1
        data_to_be_written = []
        _dict = {}
        with open(path_json,'r',encoding='utf-8') as f:
            for line in f:
                e = json.loads(line)
                relVocab = get_entity_relation_vocabulary(e,relVocab)


        with open(saveFile,'w',encoding = 'utf-8') as g:
            for _dict in data_to_be_written:
                 g.write(json.dumps(_dict) + "\n")
    
    fout = open(entityVocabPath, "w")
    
    entityVocab = sorted(entityVocab.items(), key=lambda item:item[1], reverse = True)
    for key in entityVocab:
        try:
            fout.write("%s\t%d\n" % (key[0], key[1]))
        except UnicodeEncodeError as e:
            # print(repr(e))
            # print(key, val)
            continue

    fout.close()

if __name__ == '__main__':
    main()
        
