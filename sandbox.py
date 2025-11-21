
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json

from adaboost import Adaboost

# import train data as a dataframe
train_df = pd.read_csv("datasets/adaboost-test-24.txt", sep=r'\s+', header=None, names=['x', 'y', 'class'])
test_df = pd.read_csv("datasets/adaboost-train-24.txt", sep=r'\s+', header=None, names=['x', 'y', 'class'])

# # plot data, with different colour for each class
# pallete = ['#008080','#FF00FF']
# plt.figure(1)
# sns.scatterplot(x='x', y='y', data=train_df, hue='class', palette=pallete)
# plt.title('adaboost-train-24')
#
# plt.figure(2)
# sns.scatterplot(x='x', y='y', data=test_df, hue='class', palette=pallete)
# plt.title('adaboost-test-24')

# they are circly...
# obvs not linearly separable in 2d
# so might have to increase dimensionality
# adaboost might be able to handle??

plt.figure(1)
# create adaboost object
ada = Adaboost(debug=False)

# get model from csv
final_model = pd.read_csv(f'models/adaboost_model_20251120_212548.csv')

# cast the polarities back to lists
for idx in range(0,len(final_model['polarity'])):
    final_model['polarity'][idx] = json.loads(final_model['polarity'][idx])

# visualise final model...
limits = [[-2.2,2.2],[-2.2,2.2]]
# getting original dataset, so its not modified
ada.visualise_model(final_model,limits,train_df)

plt.show()
