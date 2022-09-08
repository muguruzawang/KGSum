#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Danqing Wang"

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
from tools.logger import *

class Entity_Embedding(object):
    def __init__(self, gloveembed, entityvocab):
        """
        :param path: string; the path of word embedding
        :param vocab: object;
        """
        logger.info("[INFO] Loading external entity embedding...")
        self._gloveembed = gloveembed
        self._entityvocab = entityvocab


    def calculate_entity_embedding_by_average(self):
        list_entity2vec = []
        for i in range(self._entityvocab.size()):
            entity = self._entityvocab.id2entity(i)
            words = entity.split()
            embedding = []
            for word in words:
                if word in self._gloveembed:
                    embedding.append(self._gloveembed[word])
            
            assert len(embedding)!=0, "empty embedding error!!!"

            embedding = np.array(embedding)
            embedding = np.mean(embedding,axis = 0)
            embedding = embedding.tolist()
            list_entity2vec.append(embedding)

        return list_entity2vec