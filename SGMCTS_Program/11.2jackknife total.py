import os 
import pandas as pd


def database_gene(disease_position):
    database_list=[]
    file = pd.read_csv(disease_position)
    gene = list(file['Gene'])
    for each in gene:
        # if int(each) >= r:
        if each not in database_list:
            database_list.append(each)
    return database_list


top_gene_path = r'E:\AD\gene_total_unique\total.txt'
database_path = r'E:\AD\database\AD Phenolyzer gene.csv'
total_path = r'E:\AD\gene_total_unique\#AD-seed gene-12-20-70'
output_path = r'E:\AD\jackknife\output_AD_total'
database_total_gene = database_gene(database_path)
with open(top_gene_path) as f:
    top_gene_set = f.read().split()


for file in os.listdir(total_path):
    total_set = []
    right_gene = []
    false_gene = []
    T = 0
    F = 0
    w = open(output_path + '\\' + file, 'w')
    for gene_name in top_gene_set:
        if file != gene_name:
            with open(total_path + '\\' + gene_name) as f:
                reader1 = f.read().split()
                reader = reader1[1:]
                for gene in reader:
                    total_set.append(gene)
    i = len(total_set)
    for gene in total_set:
        if gene in database_total_gene:
            right_gene.append(gene)
            w.write(gene + '\n')
            T += 1
        else:
            false_gene.append(gene)
            F += 1
    print(F)
    w.write(str(T/i))
    w.close()

