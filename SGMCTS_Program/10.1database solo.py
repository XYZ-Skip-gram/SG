import csv
import pandas as pd
import os


def genecards_gene(r,disease_position):
    genecands_list=[]
    file = pd.read_csv(disease_position)
    score = list(file['Relevance score'])
    type = list(file['Category'])
    gene = list(file['Gene Symbol'])
    for each in score:
        if int(each) > r and type[score.index(each)] == 'Protein Coding':
            if each not in genecands_list:
                genecands_list.append(gene[score.index(each)])
    return genecands_list


def AD_gene1(file_position):
    total_set = []
    reader = file_position.read().split()
    gene_set = reader[:-1]
    for gene in gene_set:
        if gene not in total_set:
            total_set.append(gene)
    return total_set


genecard = r'E:\AD\database\GeneCards_AD.csv'
total = r'E:\AD\database\output_20-200'
path2 = r'E:\AD\unique_20-200'
database_gene = genecards_gene(1, genecard)
for file in os.listdir(path2):
    with open(path2 + '\\' + file) as gene_set:
        right = 0
        total_number = 0
        right_gene = []
        false_gene = []
        w = open(total + '\\' + file, 'w')
        AD_gene_set = AD_gene1(gene_set)
        for gene in AD_gene_set:
            if gene in database_gene:
                right_gene.append(gene)
                w.write(gene + '\n')
                right += 1
            else:
                false_gene.append(gene)
            total_number += 1
        w.write(str(right/total_number))
        w.close()
