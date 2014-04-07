import csv
from sklearn import svm

def read_csv(file_path, has_header = True):
    with open(file_path) as f:
        if has_header: f.readline()
        data = []
        for line in f:
            line = line.strip().split(",")
            data.append([str(x) for x in line])
    return data

def write_csv(file_path, data, header=''):
    with open(file_path,"w") as f:
        if header != '':
            f.write(header)
        else:
            for line in data: f.write(",".join(line) + "\n")


train = read_csv("../Data/train.csv", True)
train_labels = [x[0] for x in train]
train_data = [x[1:-1] for x in train]

test = read_csv("../Data/test.csv", True)
test_labels = [x[0] for x in test]
test_data = [x[1:-1] for x in test]

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(train_data, train_labels)

clf.predict(test_data)