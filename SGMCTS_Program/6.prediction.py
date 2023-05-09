import random
import numpy as np
import tensorflow as tf
from collections import Counter
import os


def preprocess(text, freq=1):
    words = text.split()
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > freq]
    return trimmed_words


def data_trans(init_vocab):
    V = []
    vocab = []
    for each in init_vocab:
        V.append(each)
    ordered = sorted(V)
    for a in ordered:
        a = str(a)
        vocab.append(a)
    return vocab


def get_targets(words, idx, window_size=5):
    start_point = idx - window_size if (idx - window_size) > 0 else 0
    end_point = idx + window_size
    targets = set(words[start_point: idx] + words[idx + 1: end_point + 1])
    return list(targets)


def get_batches(words, batch_size, window_size=5):
    n_batches = len(words) // batch_size
    words = words[:n_batches * batch_size]
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx: idx + batch_size]
        for i in range(len(batch)):
            batch_x = batch[i]
            batch_y = get_targets(batch, i, window_size)
            x.extend([batch_x] * len(batch_y))
            y.extend(batch_y)
        yield x, y


def policy_network(epochs=80,embedding_size=300,input_gene=None):
    with open(r'E:\AD\output.txt') as f:
        text = f.read()
    freq = 0
    words = preprocess(text, freq)
    init_vocab = list(set(words))
    gene_vocab = data_trans(init_vocab)
    gene_to_index = {gene: int for int, gene in enumerate(gene_vocab)}
    index_to_gene = {int: gene for int, gene in enumerate(gene_vocab)}
    train_words = [gene_to_index[gene] for gene in words]
    Action_priors1,Action_priors2,Action_priors = [], [], []
    train_graph = tf.Graph()
    vocab_size = len(index_to_gene)
    n_sampled = 2
    learning_rate = 0.001    


    with tf.Session(graph=train_graph) as sess:
        inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
        labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs)

        softmax_w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(vocab_size))
        loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, vocab_size)
        cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        if not input_gene:
            input_gene = random.choices(gene_vocab)[0]
        input_gene_index = gene_to_index.get(input_gene)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
        normalized_embedding = embedding / norm

        saver = tf.train.Saver()
        path = "E:\AD\model_to_save\model "+str(epochs)+"epochs "+str(embedding_size)+"dem_p"
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.all_model_checkpoint_paths[0])

        input_gene_embed = normalized_embedding.eval()[input_gene_index].reshape(1,embedding_size)
        similarity = tf.matmul(input_gene_embed, tf.transpose(normalized_embedding))
        sim = similarity.eval()
        prob_value = softmax(sim[0,:])

    for each in zip(index_to_gene.values(),(prob_value)):
        Action_priors1.append(each)

    Action_priors2 = sorted(Action_priors1,key=lambda x:x[1],reverse=True)

    for each in Action_priors2:
        Action_priors.append(each[0])
    return Action_priors[1:201]

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

if __name__ == '__main__':
    a = open(r"E:\AD\model_output\output_120epochs_300dem.txt", 'w')
    with open(r"E:\AD\test_set.txt") as f:
        genes = f.read().split('\n')
        test_gene = genes[:-1]
        for i in range(len(test_gene)):
            input_gene = test_gene[i]
            action_priors = policy_network(80,300,input_gene)
            q = ' '.join(action_priors)
            a.write(input_gene + ' ' + q + '\n')
            print(input_gene + '\taction_priors:' + q + '\n')
    a.close()


