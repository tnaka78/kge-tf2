import os
import random
import numpy as np
import pandas as pd


class KnowledgeGraph:
    def __init__(self, data_dir):
        self.entity_def = self.__read_iddef(os.path.join(data_dir, 'entity2id.txt'))
        self.relation_def = self.__read_iddef(os.path.join(data_dir, 'relation2id.txt'))
        self.entity2id = self.__make_name2id_dict(self.entity_def)
        self.id2entity = self.__make_id2name_dict(self.entity_def)
        self.relation2id = self.__make_name2id_dict(self.relation_def)
        self.num_entities = len(self.entity_def)
        self.num_relations = len(self.relation_def)
        self.train_triples = self.__read_triples(os.path.join(data_dir, 'train2id.txt'))
        self.valid_triples = self.__read_triples(os.path.join(data_dir, 'valid2id.txt'))
        self.test_triples = self.__read_triples(os.path.join(data_dir, 'test2id.txt'))
        self.head_tail_ratio = self.__calc_head_tail_ratio(self.train_triples)
        self.train_triples_set = set(self.train_triples)

    @staticmethod
    def __read_iddef(path):
        return pd.read_csv(path, sep='\t', names=['name', 'id'], skiprows=1)

    @staticmethod
    def __read_triples(path):
        return pd.read_csv(path, sep=' ', skiprows=1, names=['head', 'tail', 'relation'])

    @staticmethod
    def __make_name2id_dict(iddef_df):
        return {row.name: row.id for row in iddef_df.itertuples()}

    @staticmethod
    def __make_id2name_dict(iddef_df):
        return {row.id: row.name for row in iddef_df.itertuples()}

    @staticmethod
    def __calc_head_tail_ratio(triple_df):
        head_count = triple_df.groupby('relation')['head'].apply(set).apply(len)
        tail_count = triple_df.groupby('relation')['tail'].apply(set).apply(len)
        return tail_count / (head_count + tail_count)

    def train_data_generator(self):
        for head, tail, relation in np.random.permutation(self.train_triples.values):
            neg_head = None
            neg_tail = None
            if random.random() > self.head_tail_ratio[relation]:
                # replace head
                neg_tail = tail
                while True:
                    neg_head = random.randrange(self.num_entities)
                    if (neg_head, tail, relation) not in self.train_triples_set:
                        # yield [head, tail, relation, neg_head, tail]
                        break
            else:
                # replace tail
                neg_head = head
                while True:
                    neg_tail = random.randrange(self.num_entities)
                    if (head, neg_tail, relation) not in self.train_triples_set:
                        # yield [head, tail, relation, head, neg_tail]
                        break
            yield [head, tail, relation, neg_head, neg_tail]

    def valid_data_generator(self):
        for head, tail, relation in self.valid_triples.values:
            yield [head, tail, relation]

    def test_data_generator(self):
        for head, tail, relation in self.test_triples.values:
            yield [head, tail, relation]
