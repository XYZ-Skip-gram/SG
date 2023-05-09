import os


path1 = r'E:\AD\gene_total_unique\#AD-seed gene-12-20-70'
w = open(r'E:\AD\gene_total_unique\total_unique\#AD-12 gene-unique_20-70.txt', 'w')
unique_gene = []

for file in os.listdir(path1):
    with open(path1 + '\\' + file) as gene_set:
        gene1 = gene_set.read().split()
        print(gene1[0])
        for gene in gene1[1:]:
            if gene not in unique_gene:
                unique_gene.append(gene)
                w.write(gene + '\n')
w.write(str(len(unique_gene)))
w.close()    
