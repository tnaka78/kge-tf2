import math
import tensorflow as tf


class TransE:
    @property
    def entity_embedding(self):
        return self.__entity_embedding

    @property
    def relation_embedding(self):
        return self.__relation_embedding

    def __init__(self, num_entities: int, num_relations: int, margin: float = 1.0, dimension: int = 50,
                 learning_rate: float = 0.01, normalize: bool = True, score_method: str = 'L1'):
        self.__num_entities = num_entities
        self.__num_relations = num_relations
        self.__margin = margin
        self.__dimension = dimension
        self.__learning_rate = learning_rate
        self.__normalize = normalize
        self.__score_method = score_method

        bound = 6 / math.sqrt(self.__dimension)
        initializer = tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound)
        self.__entity_embedding = tf.keras.layers.Embedding(self.__num_entities,
                                                            self.__dimension,
                                                            name='embedding_entity',
                                                            embeddings_initializer=initializer)
        self.__relation_embedding = tf.keras.layers.Embedding(self.__num_relations,
                                                              self.__dimension,
                                                              name='embedding_relation',
                                                              embeddings_initializer=initializer)

        self.__optimizer = tf.keras.optimizers.SGD(learning_rate=self.__learning_rate)

    def __score_func(self, heads, tails, relations):
        if self.__score_method == 'L1':
            return tf.norm(heads + relations - tails, ord=1, axis=-1)
        elif self.__score_method == 'L2':
            return tf.square(tf.norm(heads + relations - tails, ord=2, axis=-1))
        else:
            raise Exception('Invalid score method:', self.__score_method)

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            heads = self.entity_embedding(inputs[:, 0])
            tails = self.entity_embedding(inputs[:, 1])
            relations = self.relation_embedding(inputs[:, 2])
            neg_heads = self.entity_embedding(inputs[:, 3])
            neg_tails = self.entity_embedding(inputs[:, 4])
            if self.__normalize:
                heads = tf.nn.l2_normalize(heads, axis=-1)
                tails = tf.nn.l2_normalize(tails, axis=-1)
                relations = tf.nn.l2_normalize(relations, axis=-1)
                neg_heads = tf.nn.l2_normalize(neg_heads, axis=-1)
                neg_tails = tf.nn.l2_normalize(neg_tails, axis=-1)
            pos_scores = self.__score_func(heads, tails, relations)
            neg_scores = self.__score_func(neg_heads, neg_tails, relations)
            loss = tf.reduce_sum(tf.maximum(pos_scores + self.__margin - neg_scores, 0.0))
        variables = self.entity_embedding.trainable_variables + self.relation_embedding.trainable_variables
        grads = tape.gradient(loss, variables)
        self.__optimizer.apply_gradients(zip(grads, variables))
        return loss

    @tf.function
    def test_step(self, inputs, all_entity_ids):
        batch_size = tf.shape(inputs)[0]
        batch_heads = self.entity_embedding(inputs[:, 0])
        batch_relations = self.relation_embedding(inputs[:, 2])
        all_entity_embeddings = self.entity_embedding(all_entity_ids)
        if self.__normalize:
            batch_heads = tf.nn.l2_normalize(batch_heads, axis=1)
            batch_relations = tf.nn.l2_normalize(batch_relations, axis=1)
            all_entity_embeddings = tf.nn.l2_normalize(all_entity_embeddings, axis=1)
        batch_heads = tf.tile(tf.expand_dims(batch_heads, 1), [1, all_entity_ids.shape[0], 1])
        batch_relations = tf.tile(tf.expand_dims(batch_relations, 1), [1, all_entity_ids.shape[0], 1])
        batch_tails = tf.tile(tf.expand_dims(all_entity_embeddings, 0), [batch_size, 1, 1])
        scores = self.__score_func(batch_heads, batch_tails, batch_relations)
        ranks = tf.argsort(tf.argsort(scores, axis=1), axis=1) + 1
        #indices = tf.stack([tf.range(batch_size, dtype=tf.int64), inputs[:, 1]], axis=1)
        rows = tf.cast(tf.range(batch_size), dtype=tf.int64)
        indices = tf.stack([rows, inputs[:, 1]], axis=1)
        return tf.gather_nd(ranks, indices)
