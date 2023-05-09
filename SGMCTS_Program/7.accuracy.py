import os
import csv


def value_network2(gene1,gene2):
    count = 0
    path = r'E:\AD\pathway'
    for file in os.listdir(path):
        with open(path + "\\" + file) as pathway:
            lines = pathway.read()
            if (gene1 in lines) and (gene2 in lines):
                count = 1
                break
            else:
                continue
    return count


def value_network1(gene1, gene2):
    count = 0
    path = r'E:\AD\pathway'
    for file in os.listdir(path):
        with open(path + "\\" +file) as pathway:
            reader = csv.reader(pathway)
            for row in reader:
                # print(row)
                if (gene1 in row) and (gene2 in row):
                    count = 1
                    break
                else:
                    continue
    return count


a = open(r'E:\AD\output\output_100epochs_250dem.txt', 'w')
with open(r'E:\AD\model_output\output_100epochs_250dem.txt') as f:
    gene = f.readline().split()
    interaction = 0
    same_pathway = 0
    while gene:
        gene1 = gene[0]
        gene2 = gene[1]
        interaction += value_network1(gene1, gene2)
        same_pathway += value_network2(gene1, gene2)
        gene = f.readline().split()
    interaction_proportion = interaction/1438
    same_pathway_proportion = same_pathway/1438
    accuracy = (interaction_proportion + same_pathway_proportion) / 2
    a.write(str(interaction) + ' ' + str(interaction_proportion) + ' ' + str(same_pathway) + ' ' + str(same_pathway_proportion) + ' ' + str(accuracy))
a.close()