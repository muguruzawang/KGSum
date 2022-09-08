import os
import math
import numpy
import copy
import nltk
import re
from sklearn.metrics.pairwise import cosine_similarity
import pdb

class Textrank(object):
    epsilon = 1e-4
    damping = 0.85

    def __call__(self,tfidfs):
        sentences_count = tfidfs.shape[0]
        weights = numpy.zeros((sentences_count, sentences_count))

        '''
        for i in range(sentences_count):
            for j in range(sentences_count):
                weights[i][j] = cosine_similarity(tfidfs[i], tfidfs[j])
        '''
        weights = cosine_similarity(tfidfs)

        weights /= weights.sum(axis=1)[:, numpy.newaxis]
        matrix = numpy.full((sentences_count, sentences_count), (1.-self.damping) / sentences_count) \
            + self.damping * weights
        ranks = self.power_method(matrix, self.epsilon)
        pdb.set_trace()
        return ranks

    @staticmethod
    def power_method(matrix, epsilon):
        transposed_matrix = matrix.T
        sentences_count = len(matrix)
        p_vector = numpy.array([1.0 / sentences_count] * sentences_count)
        lambda_val = 1.0

        while lambda_val > epsilon:
            next_p = numpy.dot(transposed_matrix, p_vector)
            lambda_val = numpy.linalg.norm(numpy.subtract(next_p, p_vector))
            p_vector = next_p

        return p_vector