import numpy as np
import tensorflow as tf
from KnowledgeGraph import KnowledgeGraph
from TransE import TransE


def evaluate(model, kg, ds):
    all_ranks = np.array([])
    for triple in ds:
        ranks = model.test_step(triple, kg.entity_def['id'].values)
        all_ranks = np.append(all_ranks, ranks.numpy())
    hits1 = len(np.where(all_ranks == 1)[0]) / len(all_ranks)
    hits10 = len(np.where(all_ranks <= 10)[0]) / len(all_ranks)
    mr = np.mean(all_ranks)
    mrr = np.mean(1.0 / all_ranks)
    print("hits@1={:4f}, hits@10={:.4f}, MR={:.4f}, MRR={:.4f}".format(hits1, hits10, mr, mrr))


# settings
DATA_DIR = '../data/FB15k'
EMBEDDING_DIM = 50
LEARNING_RATE = 0.01
TRAIN_BATCH_SIZE = 4096
TEST_BATCH_SIZE = 32
SCORE_METHOD = 'L1'  # 'L1' or 'L2'
NORMALIZE = True  # set True to normalize embedding
EPOCHS = 20

kg = KnowledgeGraph(DATA_DIR)
model = TransE(kg.num_entities, kg.num_relations, dimension=EMBEDDING_DIM, normalize=NORMALIZE,
               score_method=SCORE_METHOD)
train_ds = tf.data.Dataset.from_generator(kg.train_data_generator, output_types=(tf.int64), output_shapes=(5,))
train_ds = train_ds.batch(TRAIN_BATCH_SIZE)
valid_ds = tf.data.Dataset.from_generator(kg.valid_data_generator, output_types=(tf.int64)).batch(TEST_BATCH_SIZE)
test_ds = tf.data.Dataset.from_generator(kg.test_data_generator, output_types=(tf.int64)).batch(TEST_BATCH_SIZE)
#valid_ds = tf.data.Dataset.from_tensor_slices(kg.valid_triples.values).batch(TEST_BATCH_SIZE)
#test_ds = tf.data.Dataset.from_tensor_slices(kg.test_triples.values).batch(TEST_BATCH_SIZE)

for e in range(1, EPOCHS+1):
    total_loss = 0.0
    for batch_data in train_ds:
        loss = model.train_step(batch_data)
        total_loss += loss.numpy()
    print("Epoch {}: loss={:.6f}".format(e, total_loss))
    #evaluate(valid_ds)

evaluate(model, kg, test_ds)