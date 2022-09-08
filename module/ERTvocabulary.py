#!/usr/bin/python
# -*- coding: utf-8 -*-

class Vocab(object):
    def __init__(self, max_vocab=2**31, min_freq=-1, sp=['<PAD>', '<BOS>', '<EOS>', '<UNK>']):
        self.i2s = []
        self.s2i = {}
        self.wf = {}
        self.max_vocab, self.min_freq, self.sp = max_vocab, min_freq, sp

    def __len__(self):
        return len(self.i2s)

    def __str__(self):
        return 'Total ' + str(len(self.i2s)) + str(self.i2s[:10])

    def update(self, token):
        if isinstance(token, list):
            for t in token:
                self.update(t)
        else:
            self.wf[token] = self.wf.get(token, 0) + 1

    def build(self):
        self.i2s.extend(self.sp)
        # self.wf是频率统计，按频率进行逆向排序
        sort_kv = sorted(self.wf.items(), key=lambda x:x[1], reverse=True)
        for k,v in sort_kv:
            if len(self.i2s)<self.max_vocab and v>=self.min_freq and k not in self.sp:
                self.i2s.append(k)
        # dict.update 可以是列表，第一个元素是item,第二个元素是索引
        self.s2i.update(list(zip(self.i2s, range(len(self.i2s)))))

    def __call__(self, x):
        if isinstance(x, int):
            return self.i2s[x]
        else:
            return self.s2i.get(x, self.s2i['<UNK>'])
    
    def save(self, fname):
        pass

    def load(self, fname):
        pass

def at_least(x):
    # handling the illegal data
    if len(x) == 0:
        return ['<UNK>']
    else:
        return x