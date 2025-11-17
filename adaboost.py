import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Adaboost():

    def train_treestump(self, df, feature):

        # sort the data for the given feature...
        df.sort_values(feature, inplace=True, ignore_index=True)

        scores = np.zeros(len(df))
        # try and split to minimize the misclassifications...
        # we just brute force :)
        smallest_misclassifications = len(df)
        idx_misclassified = pd.DataFrame()
        best_split = 0
        best_score = 0
        polarity = [0,0]
        for split in range(0,len(df)):
            idx, num, score, pol= Adaboost._test_split(self,df, split)
            scores[split] = score

            # find the smallest number of misclassifications...
            if score > best_score:
                smallest_misclassifications = num
                idx_misclassified = idx
                best_split = split
                best_score = score
                polarity = pol

        print(f'idx_misclassified: {idx_misclassified}')
        print(f'smallest_misclassifications: {smallest_misclassifications}')
        print(f'best_split: {best_split}')
        print(f'best_score: {best_score}')

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
            if idx_misclassified.iloc[i].any():
                epsilon += df['weights'].iloc[i]

        scores_df = pd.DataFrame({'Score': scores}, index=df.index)

        return (({'feature': feature}, {'location' : float(line)}, {'polarity' :polarity}),
                (epsilon, smallest_misclassifications))

    def update_weights(self,df, stump_vals, bonus_vals):

        # extract stump_vals
        feature = stump_vals[0]['feature']
        line_location = stump_vals[1]['location']
        polarity = stump_vals[2]['polarity']

        epsilon = bonus_vals[0]
        num_misclassifications = bonus_vals[1]

        # sort the data for the given feature...
        df.sort_values(feature, inplace=True, ignore_index=True)

        # loop through feature values
        # work them out in a numpy array, then overwrite the dataframe col
        arr = df['weights'].to_numpy()
        for i in range(0, len(df)):
            # if feature values > line location
            if (df[feature].iloc[i] > line_location):
                if (df['class'].iloc[i] == polarity[0]):
                    # weight = 0.5/misclassifications
                    arr[i] *= 1/(2*epsilon) # 0.5/num_misclassifications
                else:
                    # weight = 0.5/(len(df)-misclassifications)
                    arr[i] *= 1/(2*(1-epsilon)) #0.5/(len(df) - num_misclassifications)
            else:
                if (df['class'].iloc[i] == polarity[1]):
                    # weight = 0.5/misclassifications
                    arr[i] *= 1/(2*epsilon) # 0.5/num_misclassifications
                else:
                    # weight = 0.5/(len(df)-misclassifications)
                    arr[i] *= 1/(2*(1-epsilon)) #0.5/(len(df) - num_misclassifications)

        # update dataframe
        df['weights'] = arr

        return df # with update weights...

    def visualise_model(self,model_vals, limits):

        # contour & contourf

        ## this is horrible...

        # we want to get the steppy plot that Colin had

        limits = [[-2.2,2.2],[-2.2,2.2]]

        # init big grid to all zeros
        size = 100
        grid = np.zeros([size,size])
        print(grid.shape)

        # iterate over all weak models
        for weak in model_vals:

            # extract stump_vals
            feature = weak[0][0]['feature']
            line_location = weak[0][1]['location']
            polarity = weak[0][2]['polarity']

            alpha = weak[1]

            # if feature == 'x'
            axis = 0
            if feature == 'y':
                axis = 1

            actual_numbers = np.linspace(limits[axis][0], limits[axis][1], size)

            for idx in range(0,size):
                # 'draw' line and add the polarity multiplied by the models alpha to the corresponding
                # values on the grid

                if actual_numbers[idx] > line_location:
                    if axis == 0:
                        grid[:,idx] += polarity[1]*alpha
                    else:
                        grid[idx,:] += polarity[1]*alpha
                else:
                    if axis == 0:
                        grid[:,idx] += polarity[0]*alpha
                    else:
                        grid[idx,:] += polarity[0]*alpha

        # round to 1 or -1
        grid_rounded = np.zeros([size,size])
        for i in range(0,size):
            for j in range(0,size):
                if grid[i,j] > 0:
                    grid_rounded[i,j] = 1
                else:
                    grid_rounded[i,j] = -1


        # do a 3d plot with levels visualised with colour
        # Create a 3D line plot with Seaborn
        # doing heatmap for now
        plt.figure(1)
        ax = sns.heatmap(grid, annot=False, square=True)
        ax.invert_yaxis()
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted value')
        plt.title(f'Confusion Matrix')

        plt.figure(2)
        ax = sns.heatmap(grid_rounded, annot=False, square=True)
        ax.invert_yaxis()
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted value')
        plt.title(f'Confusion Matrix')

        plt.show()

    def _test_split(self, df, split):

        ## beginning to get slow enough that its probably best to switch to numpy

        ## will handle trying the class labels both ways itself...

        above_idx = pd.DataFrame()
        above_misclassified = 0
        above_weight_score = [0,0]

        below_idx = pd.DataFrame()
        below_misclassified = 0
        below_weight_score = [0,0]

        total_weight_score = [0,0]

        # split the dataframe
        # already sorted so all we care about are the class and index
        if split != 0:
            above_split = df.iloc[:split, :]

            above_idx = above_split['class'].isin([-1])
            above_misclassified = sum(above_idx)
            
            for i in range(0, len(above_split)):
                if above_idx.iloc[i]:
                    above_weight_score[0] += above_split['weights'].iloc[i]
                else:
                    above_weight_score[1] += above_split['weights'].iloc[i]

        if split != len(df):
            below_split = df.iloc[split:, :]

            below_idx = below_split['class'].isin([1])
            below_misclassified = sum(below_idx)

            for i in range(0, len(below_split)):
                if below_idx.iloc[i]:
                    below_weight_score[0] += below_split['weights'].iloc[i]
                else:
                    below_weight_score[1] += below_split['weights'].iloc[i]

        total_weight_score[0] = above_weight_score[0] + below_weight_score[0]
        total_weight_score[1] = above_weight_score[1] + below_weight_score[1]

        # work out which case has better score...
        polarity = [1, -1]
        above_misclassified = sum(above_idx)
        below_misclassified = sum(below_idx)
        if total_weight_score[1] < total_weight_score[0]:
            polarity = [-1, 1]
            # invert misclassified
            above_idx = ~above_idx
            below_idx = ~below_idx
            # calculate number
            above_misclassified = sum(above_idx)
            below_misclassified = sum(below_idx)

        # print for debug
        # print(f'{split}\t{above_misclassified}\t{below_misclassified}\t\t{total_weight_score[0]}\t{total_weight_score[1]}')

        return pd.concat([above_idx, below_idx], axis=0), (above_misclassified + below_misclassified), max(total_weight_score), polarity
