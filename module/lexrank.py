import os
import math
import numpy
import copy
import nltk
import re
from sklearn.metrics.pairwise import cosine_similarity
import pdb

class Lexrank(object):
    def __init__(self,threshold):
        self.t = threshold
    def score(self, sentences, tfidfs):
        CM = [[0 for s in sentences] for s in sentences]
        
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                CM[i][j] = 0

        Degree = [0 for i in sentences]
        L = [0 for i in sentences]
        n = len(sentences)
        for i in range(n):
            for j in range(n):
                CM[i][j] = numpy.dot(tfidfs[i], tfidfs[j])
                
                if CM[i][j] > self.t:
                    CM[i][j] = 1
                    Degree[i] += 1
                    
                else:
                    CM[i][j] = 0

        for i in range(n):
            for j in range(n):
                CM[i][j] = CM[i][j]/float(Degree[i])
                
        L = self.PowerMethod(CM, n, 0.2)
        normalizedL = self.normalize(L)
        pdb.set_trace()
        for i in range(len(normalizedL)):
            score = normalizedL[i]
            sentence = sentences[i]
            sentence.setLexRankScore(score)
            
        return sentences

    def PowerMethod(self, CM, n, e):
        Po = numpy.array([1/float(n) for i in range(n)])
        t = 0
        delta = float('-inf')
        M = numpy.array(CM)
  
        while delta < e:
            t = t + 1
            M = M.transpose()
            P1 = numpy.dot(M, Po)
            diff = numpy.subtract(P1, Po)
            delta = numpy.linalg.norm(diff)
            Po = numpy.copy(P1)
            
        return list(Po)
    def buildMatrix(self, sentences):

        # build our matrix
        CM = [[0 for s in sentences] for s in sentences]
        
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                CM[i][j] = 0
        return CM
    def buildSummary(self, sentences, n):
        sentences = sorted(sentences,key=lambda x: x.getLexRankScore(), reverse=True)
        summary = []
        # sum_len = 0

        # while sum_len < n:
        #     summary += [sentences[i]]
        #     sum_len += len(sentences[i].getStemmedWords())

        for i in range(n):
            summary += [sentences[i]]
        return summary

    def normalize(self, numbers):
        max_number = max(numbers)
        normalized_numbers = []
        
        for number in numbers:
            normalized_numbers.append(number/max_number)
            
        return normalized_numbers
    def main(self, n, path):
        sentences  = self.text.openDirectory(path)
        idfs = self.sim.IDFs(sentences)
        CM = self.buildMatrix(sentences)
    
        sentences = self.score(sentences, idfs,CM, 0.1)

        summary = self.buildSummary(sentences, n)

        return summary
