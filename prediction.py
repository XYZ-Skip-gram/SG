import os
import csv
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from collections import Counter

import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

def preprocess(text, freq=1):
    words = text.split('\t')
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

def random_select_mul(valid_size=16,valid_window=10):
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000, 1000 + valid_window), valid_size//2))
    return valid_examples

def get_disease_geneset(freq,file_position):
    '''获取疾病相关的基因列表'''
    total_set,drop_set = [],[]
    data = pd.read_csv(file_position,header=None)
    data_x, data_y = data.iloc[:, 0], data.iloc[:, 1]
    for i in range(len(data_x)):
        if data_y[i] <= freq:
            drop_set.append(data_x[i])
    for each in data_x:
        total_set.append(each)
    for gene in drop_set:
        total_set.remove(gene)
    return total_set


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


with open(r'E:\drug\training\jieguo.txt') as f:
    text = f.read()
freq = 1
words = preprocess(text,freq)
init_vocab = list(set(words))
gene_vocab = data_trans(init_vocab)

gene_to_index = {gene: int for int, gene in enumerate(gene_vocab)}
index_to_gene = {int: gene for int, gene in enumerate(gene_vocab)}

train_words =  [gene_to_index[gene] for gene in words]
drug_path = r'E:\drug\data\origin\unique_drugs.txt'

drugs = []
with open(drug_path) as f:
    drugset = f.read().split('\n')
    for drug in drugset:
        drugs.append(drug)

gene_total_path = r'E:\drug\data\origin\unique_genes.txt'
total_genes = []
with open(gene_total_path) as f:
    total_genes_set = f.read().split('\n')
    for genes in total_genes_set:
        total_genes.append(genes)



def policy_network(input_D_or_T):
    with open(r'E:\drug\training\jieguo.txt') as f:
        text = f.read()
    freq = 1
    words = preprocess(text,freq)
    init_vocab = list(set(words))
    gene_vocab = data_trans(init_vocab)

    gene_to_index = {gene: int for int, gene in enumerate(gene_vocab)}
    index_to_gene = {int: gene for int, gene in enumerate(gene_vocab)}

    train_words =  [gene_to_index[gene] for gene in words]

    train_graph = tf.Graph()
    vocab_size = len(index_to_gene)

    Action_priors, Action_priors1, Action_priors2 = [], [], []
    Pathway,Pathway2=[],[]
    embedding_size = 40  
    n_sampled = 2 
    epochs = 70 
    batch_size = 1000 
    window_size = 5  

    with tf.Session(graph=train_graph) as sess:
        inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
        labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        embedding = tf.Variable(tf.random.uniform([vocab_size, embedding_size], -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs)

        softmax_w = tf.Variable(tf.random.truncated_normal([vocab_size, embedding_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(vocab_size))

        loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, vocab_size)

        cost = tf.reduce_mean(loss)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        input_DT_index = gene_to_index.get(input_D_or_T)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
        normalized_embedding = embedding / norm
        saver = tf.train.Saver()  
        path = "E:\drug\model_new\model "+str(epochs)+"epochs "+str(embedding_size)+"dem_p"

        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.all_model_checkpoint_paths[0])
        input_gene_embed = normalized_embedding.eval()[input_DT_index].reshape(1,embedding_size)

        print('input_Drug_or_Target:', input_D_or_T)

        similarity = tf.matmul(input_gene_embed, tf.transpose(normalized_embedding))
        sim = similarity.eval()
        nearest = softmax(sim[0,:])

        
    for each in zip(index_to_gene.values(),(nearest)):
        Action_priors1.append(each)

    Action_priors2 = sorted(Action_priors1,key=lambda x:x[1],reverse=True)

    for each in Action_priors2:
        Action_priors.append(each[0])

    return Action_priors[1:11] 

if __name__ == '__main__':
    gene_path = r'E:\drug\data\TOP1\prediction_new\gene_prediction-70-40.txt'
    test_path = r'E:\drug\data\origin\test_set.txt'
    a = open(gene_path, 'w')
    i = 0
    with open(test_path) as f:
        genes = f.read().split('\n')
        test_genes = genes[:]
        for input_gene in test_genes:
            if input_gene in total_genes:
                i += 1
                action_priors = policy_network(input_gene)
                q = '\t'.join(action_priors)
                a.write(input_gene + '\t' + q + '\n')
        print(str(i))
    a.close()