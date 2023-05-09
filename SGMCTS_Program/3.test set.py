import random


real_test, test_set = [], []
with open(r'E:\AD\output.txt') as traning_set:
    reader = traning_set.readline().split()
    while reader:
        for each in reader:
            if each not in real_test:
                real_test.append(each)
        reader = traning_set.readline().split()
test_set = random.sample(set(real_test),k=round(len(set(real_test))*0.25))


w = open(r'E:\AD\test_set.txt','w')
for test in test_set:
    w.write(test + '\n')
w.close()


others = []
a = open(r'E:\AD\test_others.txt','w')
for gene in real_test:
    if gene not in test_set:
        a.write(gene + '\n')
a.close()
print(test_set)
