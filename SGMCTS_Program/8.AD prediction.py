import os
import random
import numpy as np
import pandas as pd
from copy import deepcopy
import evaluation_function
import policy_function


class TreeNode(object):
    def __init__(self,parent):
        self.Q = 0.0
        self.U = 0.0
        self.visits = 0

        self.parent = parent
        self.children = {}


    def __str__(self):
        return '''(Parent：{0.parent}, Children：{0.children} , Visits：{0.visits},
            Q：{0.Q}，U：{0.U})'''.format(self)


    def UCB(self,c_puct):
        self.U = c_puct * np.sqrt(self.parent.visits) / (1e-8 + self.visits)  #可能修改c_put值
        return self.Q + self.U


    def select(self):
        return max(self.children.items(), key=lambda act_node: act_node[1].UCB(c_puct=2))


    def expand(self,action_prior):
        for gene in action_prior:
            if gene not in self.children:
                self.children[gene] = TreeNode(parent=self)


    def rollout(self,current_gene1,current_gene2):
        leaf_value = evaluation_function.value_network(gene1=current_gene1,
                                                       gene2=current_gene2)
        return leaf_value


    def update(self, leaf_value):
        self.visits += 1
        self.Q += 1.0 * (leaf_value - self.Q) / self.visits


    def update_recursive(self,leaf_value):
        if self.parent:
            self.parent.update_recursive(leaf_value)
        self.update(leaf_value)


def get_disease_geneset_with_freq(freq, file_position): 
    total_set, drop_set = [], []
    data = pd.read_csv(file_position)
    data_x, data_y = data.iloc[:, 0], data.iloc[:, 1]
    for i in range(len(data_x)):
        if data_y[i] <= freq:
            drop_set.append(data_x[i])
    for each in data_x:
        total_set.append(each)
    for gene in drop_set:
        total_set.remove(gene)
    return total_set

def get_disease_geneset_without_freq(file_position):
    gene_set = []
    with open(file_position) as file:
        line = file.readline().split()
        while line:
            gene_set.append(line[0])
            line = file.readline().split()
    return gene_set

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def omim(disease_position):

    file = pd.read_csv(disease_position)
    gene = file['Gene/Locus']
    for each in gene:
        input_gene = each
        return input_gene


i = 0
index = 0
search_tree = []
path = r'E:\AD\\'
disease_genes = get_disease_geneset_without_freq(
    r'E:\AD\Alzheimer Disease.txt')
print('Total_disease_genes:',len(disease_genes))


while index < len(disease_genes):
    w = open(path + str(disease_genes[index]) + '.txt','w')
    search_tree = []
    input_gene = disease_genes[index]
    print(input_gene)
    root_dict = {input_gene: TreeNode(parent=None)}
    reversed_root_dict = {TreeNode(parent=None): input_gene}
    root_gene = input_gene
    root_node = root_dict[root_gene]
    root_node.visits = 1
    try:
        actions = policy_function.policy_network(input_gene=root_gene)
    except ValueError:
        continue
    if actions == []:
        print('INPUT GENE %s HAS NO CHILD IN DISEASE_SET,CHOSICE AGAIN!'  % root_gene)
        continue
    else:
        root_dict[root_gene].expand(actions)
        w.write(root_gene+'\n')

    print(actions)


    while len(set(search_tree)) <= 19:
        most_pro_tuple = root_node.select()
        most_gene,most_node = most_pro_tuple[0],most_pro_tuple[1]
        if most_node.visits == 0:
            scores = most_node.rollout(most_gene,root_gene)
            most_node.update_recursive(scores)
        else:
            search_tree.append(root_gene)
            search_tree.append(most_gene)
            print(most_gene)
            w.write(most_gene + '\n')

            actions = policy_function.policy_network(input_gene=most_gene)
            print(actions)

            for little_gene in deepcopy(actions):
                if little_gene in search_tree:
                    if search_tree.count(little_gene) > 2:
                        actions.remove(little_gene)
            most_node.expand(actions)
            most_node.update_recursive(leaf_value=0)

            for children in list(most_node.children.items()):
                if children[1].visits != 0:
                    root_gene = root_gene
                    root_node = root_node
                else:
                    root_gene = most_gene
                    root_node = most_node

            if len(actions) == 1 and actions[0] == most_gene:
                break
            if actions == []:
                break
    else:
        pass
    print('%s random select have finished' % (index+1))
    index += 1
    w.close()
