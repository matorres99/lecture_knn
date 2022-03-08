import pandas
import kfold_template
import numpy
from sklearn import KNeighborsRegressor

dataset = pandas.read_csv("abalone.data", header=None)

dataset.columns = ["Sex", "Length", "Diameter", "Height", "Whole", "Shucked", "Viscera", "Shell", "Rings"]

dataset = dataset.drop("Sex", axis=1)

print(dataset)

target = dataset.iloc[:,7].values
data = dataset.iloc[:,0:7].values

print(target)
print(data)

machine = KNeighborsRegressor(n_neighbors=20)
r2_scores = kfold_template.run_kfold(data, target, 4, machine, 0, 0)
r2_scores = r2_scores[0]
r2_score = numpy.mean(r2_scores)
print(r2_scores)

for i in numpy.arange(1,60):
	machine = KNeighborsRegressor(n_neighbors=i)
	r2_scores = kfold_template.run_kfold(data, target, 4, machine, 0, 0)
	r2_scores = r2_scores[0]
	r2_score = numpy.mean(r2_scores)
	print("n_neighbors: ", i)
	print("r2_score: ", r2_score)