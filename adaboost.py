import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Adaboost():

    def train_treestump(self, df, feature):

        # sort the data for the given feature...
        df.sort_values(feature, inplace=True, ignore_index=True)

        data = []

        # try and split to minimize the misclassifications...
        # we just brute force :)
        smallest_misclassifications = len(df)
        idx_misclassified = pd.DataFrame()
        best_split = 0
        polarity = [0,0]
        for split in range(0,len(df)):
            idx, num, pol= Adaboost._test_split(self,df, split)
            data.append([idx,num])

            # find the smallest number of misclassifications...
            if num < smallest_misclassifications:
                smallest_misclassifications = num
                idx_misclassified = idx
                best_split = split
                polarity = pol

        print(f'idx_misclassified: {idx_misclassified}')
        print(f'smallest_misclassifications: {smallest_misclassifications}')
        print(f'best_split: {best_split}')

        # find where we should draw line based off best split...
        # get value of feature for point on either side, take average...
        if best_split == 0:
            line = df[feature].iloc[0]
        elif best_split == len(df)-1:
            line = df[feature].iloc[len(df)-1]
        else:
            line = (df[feature].iloc[best_split-1] + df[feature].iloc[best_split])/2

        # calculate our weighted error, epsilon...
        # iterate over our original sorted dataframe, referring to new idx_misclassifications to see
        # which weights to add up...
        epsilon = 0
        for i in range(0, len(df)):
            if idx_misclassified.iloc[i]:
                epsilon += df['weights'].iloc[i]

        return ({'feature': feature}, {'location' : float(line)}, {'ploarity' :polarity}), epsilon

    def _test_split(self, df, split):

        ## will handle trying the class labels both ways itself...

        aboveIdx = [pd.DataFrame(), pd.DataFrame()]
        aboveMisclassified = [0, 0]

        belowIdx = [pd.DataFrame(), pd.DataFrame()]
        belowMisclassified = [0, 0]

        # split the dataframe
        # already sorted so all we care about are the class and index
        if split != 0:
            aboveSplit = df.iloc[:split,:]

            aboveIdx[0] =  aboveSplit['class'].isin([-1])
            aboveMisclassified[0] = sum(aboveIdx[0])

            aboveIdx[1] = aboveSplit['class'].isin([1])
            aboveMisclassified[1] = sum(aboveIdx[1])

        if split != len(df):
            belowSplit = df.iloc[split:,:]

            belowIdx[0] = belowSplit['class'].isin([1])
            belowMisclassified[0] = sum(belowIdx[0])

            belowIdx[1] = belowSplit['class'].isin([-1])
            belowMisclassified[1] = sum(belowIdx[1])

        # print for debug
        print(f'{split}\t{aboveMisclassified}\t{belowMisclassified}')

        # work out which case has less misclassifications
        case = 0
        polarity = [1,-1]
        if (aboveMisclassified[1]+belowMisclassified[1]) < (aboveMisclassified[0]+belowMisclassified[0]):
            case = 1
            polarity = [-1, 1]

        return pd.concat([aboveIdx[case],belowIdx[case]]), (aboveMisclassified[case] + belowMisclassified[case]), polarity







