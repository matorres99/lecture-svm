import kfold_template

import pandas
#reads dataset

from matplotlib import pyplot
from sklearn import svm

dataset = pandas.read_csv("datasets/dataset_svm_1.csv")

print(dataset)

target = dataset.iloc[:,0].values
#iloc finds the location of the dataset (columns)
data = dataset.iloc[:,1:3].values
#colon sets that we want all of that column

print(target)
print(data)
##if data is sensitive, should not put dataset into git repository since it is public

pyplot.scatter(data[:,0], data[:,1], c=target)
pyplot.savefig("scatterplot_1.png")
pyplot.close()

r2_scores, accuracy_scores, confusion_matrices = kfold_template.run_kfold(data, target, 4, svm.SVC(kernel="linear"), 1, 1)

print(r2_scores)
print(accuracy_scores)
print(confusion_matrices)
for i in confusion_matrices:
	print(i)