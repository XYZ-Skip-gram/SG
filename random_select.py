import pandas as pd
import random

total = []
gene_drugs = {}
drug_genes = {}
with open(r'E:\drug\data\origin\drug-targets.txt', 'r') as f:
    reader = f.readline().strip().split('\t')
    while reader:
        drug, genes = reader[0], reader[1:]
        if len(genes) == 0:
            reader = f.readline().strip().split('\t')
            continue
        drug_genes.setdefault(drug, genes)
        reader = f.readline().strip().split('\t')
        if reader == ['']:
            break
with open(r'E:\drug\data\origin\target-drugs.txt') as f:
    reader = f.readline().strip().split('\t')
    while reader:
        gene, drugs = reader[0], reader[1:]
        if len(genes) == 0:
            reader = f.readline().strip().split('\t')
            continue
        gene_drugs.setdefault(gene, drugs)
        reader = f.readline().strip().split('\t')
        if reader == ['']:
            break

def loop(random_D_T):
    if random_D_T in gene_drugs.keys():
        seq.append(random_D_T)
        random_D_T = random.choice(gene_drugs[random_D_T])
        if seq.count(random_D_T) < 2:
            loop(random_D_T)
        else:
            seq.append(random_D_T)
            if seq not in total:
                total.append(seq)
    elif random_D_T in drug_genes.keys():
        seq.append(random_D_T)
        random_D_T = random.choice(drug_genes[random_D_T])
        if seq.count(random_D_T) < 2:
            loop(random_D_T)
        else:
            seq.append(random_D_T)
            if seq not in total:
                total.append(seq)
    else:
        seq.append(random_D_T)
        if seq not in total:
            total.append(seq)

    return seq


path1 = r'E:\drug\data\origin\test_others.txt'
with open(path1) as f:
    reader = f.read().split('\n')
    for gene in reader:
        iteration = 60
        print(gene)
        for i in range(iteration):
            seq = []
            random_root = gene
            seq_list = loop(random_root)

unique = []
with open(r'E:\drug\jieguo1.txt', 'w') as w:
    
    for seq_list in total:
        for each in seq_list:
            unique.append(each)
            w.write(each + '\t')

with open(r'E:\drug\bili.txt', 'a') as q:
    zongshu = len(unique)
    geshu = len(set(unique))
    unique_D_T = set(unique)
    for unique_gene in gene_drugs.keys():
        unique_D_T.remove(unique_gene)
    bili = len(set(unique))/(6031+3064)
    drug_bili = len(unique_D_T)/6031
    q.write(str(iteration) + '\n')
    q.write(str(zongshu) + '\n')
    q.write(str(geshu) + '\n')
    q.write(str(bili) + '\n')
    q.write(str(drug_bili) + '\n\n')