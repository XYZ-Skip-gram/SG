import random
import pandas as pd


gene_dict = {}
seq, total, genes, times = [], [], [], []
weight_dict = {}


with open(r'E:\AD\gene_numbers.txt') as f:
    reader = f.readline().split()
    while reader:
        root_gene, sub_set = reader[0], reader[1:]
        if root_gene in sub_set:
            sub_set.remove(root_gene)
        if len(sub_set) == 0:
            reader = f.readline().split()
            continue
        weight_dict.setdefault(root_gene,sub_set)
        reader = f.readline().split()
print(list(weight_dict.items())[0])
print(str(5772- len(weight_dict)) + ' genes have been clear.')


def loop(random_gene):
    if random_gene in weight_dict.keys():
        seq.append(random_gene)
        random_gene = random.choice(weight_dict[random_gene])
        if seq.count(random_gene) <= 2:
            loop(random_gene)
        else:
            seq.append(random_gene)
            if seq not in total:
                total.append(seq)
    else:
        seq.append(random_gene)
        if seq not in total:
            total.append(seq)
    return seq


for keys in weight_dict.keys():

    iteration = 150

    for i in range(iteration):
        seq = []
        random_root = keys
        seq.append(random_root)
        random1 = random.choice(weight_dict[random_root])
        seq_list = loop(random1)
unique = []
with open(r'E:\AD\output.txt', 'w') as w:
    for seq_list in total:
        for each in seq_list:
            unique.append(each)
            w.write(each + ' ')
        w.write('\n')


with open(r'E:\AD\proportion.txt', 'a') as q:
    total = len(unique)
    unique_gene = len(set(unique))
    proportion = len(set(unique))/5772
    q.write(iteration + 'loop' + '\n')
    q.write('total:' + str(total) + '\n')
    q.write('unique gene:' + str(unique_gene) + '\n')
    q.write('proportion:' + str(proportion) + '\n\n')

# print('iter:',iteration)
print('len(unique):', len(unique))
print('len(set(unique)):', len(set(unique)))
print('proportion:', (len(set(unique)))/5772)


