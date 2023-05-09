import os


def value_network(gene1,gene2):
    count = 0
    path = r'E:\AD\pathway'
    for file in os.listdir(path):
        with open(path + "\\" + file) as pathway:
            lines = pathway.read()
            if (gene1 in lines) and (gene2 in lines):
                count += 1
            else:
                continue
    return count