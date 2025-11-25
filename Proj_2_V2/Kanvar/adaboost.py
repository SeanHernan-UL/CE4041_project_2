# ce4041 adaboost project
# name: Kanvar Murray
# student id: 22374698
# date: 21/11/25

# members = [('Kanvar Murray', 22374698), ('Seán Hernan', 22348948), ('Madeline Ware', 21306591)]

import numpy as np


class AdaBoost:

    # set parameters and controls
    def __init__(self, base_learner_class, n_weak_learners=50,
                 target_train_accuracy=None, avoid_perfect=False):

        # store weak learner class
        self.base_learner_class = base_learner_class
        # max weak learners
        self.n_weak_learners = n_weak_learners
        # target train accuracy under 100 percent to avoid that overfit boi
        self.target_train_accuracy = target_train_accuracy
        # stop at perfect if that happens so we can step back a lil
        self.avoid_perfect = avoid_perfect

        # learners and weights
        self.learners = []
        self.alphas = []

    # training the BOOSHT
    def fit(self, point, point_class):
        point = np.asarray(point)
        point_class = np.asarray(point_class).astype(int)
        n_samples = point.shape[0]

        # start uniform weights
        weights = np.ones(n_samples) / n_samples

        # reset stored learners
        self.learners = []
        self.alphas = []

        # running strong score
        strong_scores = np.zeros(n_samples)

        # loop over boosting rounds
        for m in range(self.n_weak_learners):

            # train weak learner
            learner = self.base_learner_class()
            learner.fit(point, point_class, weights)
            y_predicted = learner.predict(point)

            # weighted error
            misclassified = (y_predicted != point_class)
            epsilon = np.sum(weights[misclassified])

            # handle perfect weak learner
            if epsilon <= 0:
                alpha = 0.5 * np.log((1 - 1e-10) / (1e-10))

                # add tentatively
                self.learners.append(learner)
                self.alphas.append(alpha)
                strong_scores += alpha * y_predicted

                # check strong accuracy
                y_strong = np.sign(strong_scores)
                y_strong[y_strong == 0] = 1
                accuracy = np.mean(y_strong == point_class)

                # remove if perfect not allowed
                if self.avoid_perfect and accuracy == 1.0:
                    self.learners.pop()
                    self.alphas.pop()

                break

            # stop if too weak
            if epsilon >= 0.5:
                break # TODO what accuracy would this correspond to? below 50%?

            # compute alpha weight
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)

            # update sample weights
            weights = weights * np.exp(-alpha * point_class * y_predicted)
            weights /= np.sum(weights)

            # store weak learner
            self.learners.append(learner)
            self.alphas.append(alpha)

            # update strong score
            strong_scores += alpha * y_predicted

            # check train accuracy
            y_strong = np.sign(strong_scores)
            y_strong[y_strong == 0] = 1
            accuracy = np.mean(y_strong == point_class)

            # stop if target accuracy reached but not overfitted so ideal fr fr
            if self.target_train_accuracy is not None and accuracy >= self.target_train_accuracy and accuracy < 1.0:
                break

            # stop early
            if self.avoid_perfect and accuracy == 1.0:
                self.learners.pop()
                self.alphas.pop()
                break

        return self

    # predict labels
    def predict(self, x, T=None):
        # ensure model trained
        if not self.learners:
            raise ValueError("model not fitted")

        x = np.asarray(x)

        # limit T to number of learners
        if T is None or T > len(self.learners):
            T = len(self.learners)

        # accumulate weighted votes (its like america)
        weights = np.zeros(x.shape[0])
        for t in range(T):
            weights += self.alphas[t] * self.learners[t].predict(x)

        # convert score to sign label
        y_predicted = np.sign(weights)
        y_predicted[y_predicted == 0] = 1

        return y_predicted.astype(int)
