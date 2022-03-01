import kfold_template
import pandas
from matplotlib import pyplot

from sklearn import svm

dataset = pandas.read_csv("datasets/dataset_svm_2.csv")

target = dataset.iloc[:,0].values
data = dataset.iloc[:,1:3].values

print(target)
print(dataset)

pyplot.scatter(data[:,0], data[:,1], c=target)
pyplot.savefig("scatterplot_2.png")
pyplot.close()

r2_scores, accuracy_scores, confusion_matrices = kfold_template.run_kfold(data, target, 4, svm.SVC(kernel="rbf"), 1, 1)

print(r2_scores)
print(accuracy_scores)
print(confusion_matrices)
for i in confusion_matrices:
	print(i)

