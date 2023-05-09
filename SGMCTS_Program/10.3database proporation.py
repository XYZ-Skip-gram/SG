import csv
import pandas as pd
import os
import math


def database_gene(disease_position):
    database_list=[]
    file = pd.read_csv(disease_position)
    gene = list(file['Symbol'])

    for each in gene:
        if each not in database_list:
            database_list.append(each)
    return database_list


def AD_gene(file_position):
    total_set = []
    with open(file_position) as f:
        reader = f.read().split()
        gene_set = reader[:-1]
        for gene in gene_set:
            if gene not in total_set:
                total_set.append(gene)
    return total_set


def AD_gene1(file_position):
    total_set = []
    file = pd.read_csv(file_position)
    gene = file.iloc[:, 1]
    for each in gene:
        if each not in total_set:
            total_set.append(each)
    return total_set


right = 0
total_number = 0
right_gene = []
false_gene = []
database_path = r'E:\AD\database\AD GeneCards gene.csv'
w = open(r'E:\AD\database\output_GeneCards_pina\TopProteins_pina-100%.txt', 'w')
database_total_gene = database_gene(database_path)
path2 = r'E:\AD\database\TopProteins_pina.csv'
AD_gene_set = AD_gene1(path2)
r = 1
number = math.ceil(len(database_total_gene) * r)
while total_number < number:
    for gene in AD_gene_set:
        if gene == database_total_gene[total_number]:
            right_gene.append(gene)
            w.write(gene + '\n')
            right += 1 
    total_number += 1
w.write(str(right/len(AD_gene_set)))
w.close()
