import random


test_other = []
test_others = []
test_sets = []
other_set = open(r'E:\AD\test_others.txt')
w = open(r'E:\AD\random_select.txt', 'w')
test_set = open(r'E:\AD\test_set.txt')


reader = test_set.readline().split()
while reader:
    test_sets.append(reader)
    reader = test_set.readline().split()


other = other_set.readline().split()
while other:
    test_others.append(other)
    other = other_set.readline().split()


for test in test_sets:
    a = ' '.join(test)
    w.write(a)
    w.write(' ')
    test_other = random.sample(test_others, k=2)
    for other1 in test_other:
        b = ' '.join(other1)
        w.write(b)
        w.write(' ')
    w.write('\n')

test_set.close()
w.close()
other_set.close()
