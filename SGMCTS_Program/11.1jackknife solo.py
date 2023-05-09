import os 
import pandas as pd

def database_gene(disease_position):
    database_list=[]
    file = pd.read_csv(disease_position)
    gene = list(file['Gene'])
    for each in gene:
        if each not in database_list:
            database_list.append(each)
    return database_list


def AD_gene(unique_path):
    unique_set = []
    for file in os.listdir(unique_path):
        with open(unique_path + '\\' + file) as f:
            reader1 = f.read().split()
            reader = reader1[:-1]
            for gene in reader:
                if gene not in unique_set:
                    unique_set.append(gene)
    return unique_set


def AD_solo_gene(file_position):
    solo_set = []
    reader = file_position.read().split()
    gene_set = reader[:-1]
    for gene in gene_set:
        if gene not in solo_set:
            solo_set.append(gene)
    return solo_set


unique_path = r'E:\AD\gene_total_unique\#AD-12 gene-unique_20-70'
solo_path = r'E:\AD\gene_total_unique\#AD-12 gene-unique_20-70'
database_path = r'E:\AD\database\AD Phenolyzer gene.csv'
output_path = r'E:\AD\jackknife\output_AD_unique'
database_total_gene = database_gene(database_path)


for file in os.listdir(solo_path):
    w = open(output_path + '\\' + file, 'w')
    right_gene = []
    false_gene = []
    T = 0
    F = 0
    with open(solo_path + '\\' + file) as f:
        unique_set = AD_gene(unique_path)
        solo_gene_set = AD_solo_gene(f)
        for solo_gene in solo_gene_set:
            if solo_gene in unique_set:
                unique_set.remove(solo_gene)
    i = len(unique_set)
    print(i)
    for gene in unique_set:
        if gene in database_total_gene:
            right_gene.append(gene)
            w.write(gene + '\n')
            T += 1
        else:
            false_gene.append(gene)
            F += 1
    w.write(str(T/i))
    w.close()


