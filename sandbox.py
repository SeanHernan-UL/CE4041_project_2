
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# import train data as a dataframe
train = pd.read_csv("datasets/adaboost-test-24.txt", sep=r'\s+', header=None, names=['x', 'y', 'class'])
test = pd.read_csv("datasets/adaboost-train-24.txt", sep=r'\s+', header=None, names=['x', 'y', 'class'])

# plot data, with different colour for each class
pallete = ['#008080','#FF00FF']
plt.figure(1)
sns.scatterplot(x='x', y='y', data=train, hue='class', palette=pallete)
plt.title('adaboost-train-24')

plt.figure(2)
sns.scatterplot(x='x', y='y', data=test, hue='class', palette=pallete)
plt.title('adaboost-test-24')

# they are circly...
# obvs not linearly separable in 2d
# so might have to increase dimensionality
# adaboost might be able to handle??
plt.show()
