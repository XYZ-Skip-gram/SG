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

def random_select_one():
    gene_id = random.randint(0, len(gene_vocab))
    query_gene = gene_vocab[gene_id]
    query_int = gene_to_index[query_gene]
    valid_examples = [query_int]
    print('query_gene:', query_gene)
    print('query_gen_int:', valid_examples)
    return query_gene,query_int

def model_train_save(path):
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = get_batches(train_words, batch_size, window_size)
        start = time.time()
        for x, y in batches:
            feed = {inputs: x,
                    labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

            loss += train_loss
            if iteration % 1000 == 0:
                end = time.time()
                print("Epochs：{}/{}".format(e,epochs),
                      "Iteration: {}".format(iteration),
                      "Avg.Training loss: {:.4f}".format(loss / 100),
                      "{:.4f} sec/batch".format((end - start) / 100))
                loss = 0
                start = time.time()
                if iteration % 2000 == 0:
                    print('第{}次迭代结果：'.format(iteration))
                    sim = similarity.eval()
                    for i in range(valid_size):
                        valid_word = index_to_gene[int_of_valid_gene[i]]
                        top_k = 6  
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log = 'Nearest to [%s]:' % valid_word
                        for k in range(top_k):
                            close_word = index_to_gene[nearest[k]]
                            log = '%s %s,' % (log, close_word)
                        print(log)
            iteration += 1

        save_path = saver.save(sess, path)
        embed_mat = sess.run(normalized_embedding)

def model_import(path):
    ckpt = tf.train.get_checkpoint_state(path)
    saver.restore(sess, ckpt.all_model_checkpoint_paths[0])
    top_k = 8  
    sim = similarity.eval()
    for i in range(valid_size):
        valid_word = index_to_gene[int_of_valid_gene[i]]
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
    most_pro = index_to_gene[nearest[0]]

    print('valid_word:',valid_word)
    print('The most relevant gene:',most_pro)
    gene_candidient = []
    for k in range(top_k):
        gene_candidient.append(index_to_gene[nearest[k]])
    print('The most relevant %d genes: %s' % (top_k,gene_candidient))
    return gene_candidient

def get_disease_geneset(freq,file_position):
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

with open(r'E:\drug\training\jieguo.txt') as f:
    text = f.read()
freq = 1
words = preprocess(text,freq)
if freq >= 1:
    print('-----Delet low frequent genes----')
else:
    print('-----All drugs and targets join training-----')
init_vocab = list(set(words))
gene_vocab = data_trans(init_vocab)

gene_to_index = {gene: int for int, gene in enumerate(gene_vocab)}
index_to_gene = {int: gene for int, gene in enumerate(gene_vocab)}
print("total drugs-targets: {}".format(len(text.split())))
print("unique drugs-targets: {}".format(len(set(words))))

train_words =  [gene_to_index[gene] for gene in words]
print('train_DTs:',train_words[:20])

train_graph = tf.Graph()
vocab_size = len(index_to_gene)

Pathway,Pathway2=[],[]
embedding_size = 40  
n_sampled = 2 
epochs = 70 
batch_size = 1000 
window_size = 5  
print('embedding_size: {} ; epochs: {} ; batch_size: {} ; window_size: {}'.
      format(embedding_size,epochs,batch_size,window_size))


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

    input_D_or_T = random.choices(gene_vocab)[0]
    input_DT_index = gene_to_index.get(input_D_or_T)
    print('input_D_or_T:',input_D_or_T)

    int_of_valid_gene = [input_DT_index]
    valid_size = len(int_of_valid_gene)
    valid_dataset = tf.constant(int_of_valid_gene, dtype=tf.int32)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))

    saver = tf.train.Saver()  
    path = "E:\drug\model_new\model "+str(epochs)+"epochs "+str(embedding_size)+"dem_p"

    model_train_save(path + '\model.ckpt')
