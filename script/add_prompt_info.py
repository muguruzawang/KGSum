#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import json

path= r'/data/run01/scv0028/wpc/survey_generation/HeterEnReTSumGraph_ABS/datasets/multi_xscience'

Types = ['train','val','test']
for t in Types:
    path_type_json = os.path.join(path,t+'_summary.ent_type_relation.jsonl')
    path_summary = os.path.join(path,t+'.label.jsonl')
    path_w = os.path.join(path,t+'.ent_promptsummary.jsonl')
    with  open(path_type_json,'r',encoding='utf-8') as f2, open(path_w,'w',encoding='utf-8') as f3 ,\
            open(path_summary,'r',encoding='utf-8') as f4:
        lines2 = f2.readlines()
        lines4 = f4.readlines()
        for line2,line4 in zip(lines2,lines4):
            json2 = json.loads(line2)
            json4 = json.loads(line4)

            entities = json2['entities']
            types = json2['types']
            relations = json2['relations']
            for i in range(types.count([])):
                types.remove([])

            ent_list = []
            for key in entities:
                ent_list += entities[key]
            assert len(ent_list) == len(types)

            summary = 'the entities and types are : '
            if len(types) == 0:
                summary += ' none . '
            else:
                for ent, t in zip(ent_list[:-1],types[:-1]):
                    sent = ' '+'<'+str(t.lower())+'>'+' '+ent+' , '
                    summary += sent
                sent = ' '+'<'+str(types[-1].lower())+'>'+' '+ent_list[-1]+' . '
                summary += sent
            summary += ' the relations are : '
            if len(relations) == 0:
                summary += ' none . '
            elif len(relations) == 1:
                rel = relations[0]
                summary += rel[0]+' <'+str(rel[1].lower())+'> '+ rel[2]+' . '
            else:
                for rel in relations[:-1]:
                    sent = rel[0]+' <'+str(rel[1].lower())+'> '+ rel[2]+' , '
                    summary += sent
                sent = relations[-1][0]+' <'+str(relations[-1][1].lower())+'> '+ relations[-1][2]+' . '
                summary += sent

            json4['summary'] = summary

            print(json.dumps(json4),file=f3)

