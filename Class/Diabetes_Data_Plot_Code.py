#Classifier

# 1. Prepare Problem
# a) Load libraries
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
import seaborn as sb
import numpy

# b) Load dataset
filename = 'diabetes.csv'
dataset = read_csv(filename)

# 2. Summarize Data
# a) Descriptive statistics

# shape
print(dataset.shape)

# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('Outcome').size())

#Correlation
set_option('display.width', 100)
set_option('display.precision', 3)

correlations = dataset.corr(method='pearson')
print(correlations)

#Skew
skew = dataset.skew()
print(skew)
# b) Data visualizations
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()

# histograms
dataset.hist()
pyplot.show()

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()
#SB PLOT
sb.pairplot(dataset,hue='Outcome')

#Density Plot
dataset.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.show()

#Correlation Matrix Plot
# plot correlation matrix

names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age','Outcome']
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

# 3. Prepare Data
# a) Data Cleaning
dataset=dataset.dropna()