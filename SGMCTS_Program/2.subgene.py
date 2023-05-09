import os
import pandas as pd
import csv


def value_network(gene1,gene2):
    count = 0
    path = r'E:\AD\pathway'
    for file in os.listdir(path):
        with open(path + "\\" + file) as pathway:
            hang = pathway.readline()
            lines = pathway.read()
            #print(lines)
            if (gene1 in lines) and (gene2 in lines):
                count += 1
            else:
                continue
    return count


gene_dict = {}
with open(r'E:\AD\gene1_gene2_count.txt') as f:
    _ = f.readline().split('\t')
    gene = f.readline().split('\t')
    root_gene = gene[0]
    sub_gene = []
    while gene[0]:
        if root_gene != gene[0]:
            gene_dict[root_gene] = sub_gene
            sub_gene = []
            root_gene = gene[0]
            for i in range(int(gene[2])):
               sub_gene.append(gene[1])
        else:
            for i in range(int(gene[2])):
               sub_gene.append(gene[1])
        gene = f.readline().split('\t')
    gene_dict[root_gene] = sub_gene


final_gene = pd.DataFrame([gene_dict]).T
final_gene = final_gene.reset_index().rename(columns={'index':'up',0:'down'})
final_gene['down1']=None
for i,j in enumerate(final_gene['down']):
    final_gene['down1'][i]=' '.join(j)
final_gene['down'] = final_gene['down1']
final_gene['down1'] = None
print(final_gene)
final_gene.to_csv(r'E:\AD\gene_numbers.csv', index=None)
data = pd.read_csv(open(r'E:\AD\gene_numbers.csv', 'r'))
with open(r'E:\AD\gene_numbers.txt', 'w') as f:
    for line in data.values:
        f.write((str(line[0])+' '+str(line[1])+'\n'))