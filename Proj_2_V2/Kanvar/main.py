# ce4041 adaboost project
# name: Kanvar Murray
# student id: 22374698
# date: 21/11/25

import numpy as np
import matplotlib.pyplot as plt
from weak_learner import WeightedWeakLinear, WeightedDecisionStump
from adaboost import AdaBoost


def load_data(filename):
    data = np.loadtxt(filename)
    point = data[:, :2]
    point_class = data[:, 2].astype(int)
    return point, point_class


# plot decision boundary
def plot_decision_boundary(ax, ada, X, y, T, title):
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches

    # SHAPES AND COLOURS!
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = ada.predict(grid, T=T).reshape(xx.shape)
    cmap = ListedColormap(["#ffcccc", "#ccccff"])
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1], cmap=cmap)

    # plot training points
    class1 = (y == 1)
    class2 = (y == -1)
    # SHAPES AND COLOURS!!
    p1 = ax.scatter(X[class1, 0], X[class1, 1], marker='o', label='class +1')
    p2 = ax.scatter(X[class2, 0], X[class2, 1], marker='x', label='class -1')
    patch_neg = mpatches.Patch(color="#ffcccc", label="predict -1")
    patch_pos = mpatches.Patch(color="#ccccff", label="predict +1")

    ax.set_title(f"{title}")
    ax.grid(True)
    ax.legend(handles=[p1, p2, patch_pos, patch_neg])


def main():
    # load training and test sets
    points_train, classes_train = load_data("adaboost-train-24.txt") # this has 400 datapoints
    points_test, classes_test = load_data("adaboost-test-24.txt") # this has 1200 datapoints <-- feels like this should be the training set?
    
    ## First train model to 100 % to see how many weak learners are needed...
    print("## Training model to 100 % to see how many weak learners are needed ##")

    # set target accuracy to be just below the 100% cause overfitting is bad and makes me sad
    # target_train_accuracy = 1.0 - (1.0 / n_train)
    target_train_accuracy = 1 # set training accuracy to 100 %

    # train adabost strong learner with early stopping
    ada = AdaBoost(
        base_learner_class=WeightedDecisionStump,
        n_weak_learners=170, # manually setting this since we know its value
        target_train_accuracy=target_train_accuracy,
        avoid_perfect=False # setting this to true results in 169 weak learners needed
    )
    ada.fit(points_train, classes_train)  # train the strong model

    M = len(ada.learners)
    print("strong model learners used to get to 100%:", M)

    # final train and test accuracy using all learners
    predictions_from_train_set = ada.predict(points_train)
    predictions_from_test_set = ada.predict(points_test)
    accuracy_on_train_set = np.mean(predictions_from_train_set == classes_train)
    accuracy_on_test_set = np.mean(predictions_from_test_set == classes_test)

    print("final train accuracy:", accuracy_on_train_set)
    print("final test accuracy:", accuracy_on_test_set)
    print('')

    # accuracy vs weak_learner curves
    train_accuracy = []
    test_accuracy = []
    for weak_learner in range(1, M + 1):
        train_accuracy.append(np.mean(ada.predict(points_train, weak_learner) == classes_train))
        test_accuracy.append(np.mean(ada.predict(points_test, weak_learner) == classes_test))

    train_accuracy = np.array(train_accuracy)
    test_accuracy = np.array(test_accuracy)

    n = None
    for i, a in enumerate(train_accuracy, start=1):
        if a == 1.0:
            n = i
            break

    max_test_acc = float(np.max(test_accuracy))
    ntest = int(np.argmax(test_accuracy) + 1)

    print('## Iterating over saved model accuracies to find optimal model ## ')
    print("first weak_learner with 100 percent training accuracy n:", n)

    print("max test accuracy:", max_test_acc, "at weak_learner:", ntest)
    print(f"train accuracy at weak_learner={ntest}:", train_accuracy[ntest - 1])
    print(f"test accuracy at weak_learner={ntest}:", test_accuracy[ntest - 1], '<-- this is our best strong model')
    print(f"train accuracy at weak_learner={M}:", train_accuracy[-1])
    print(f"test accuracy at weak_learner={M}:", test_accuracy[-1])

    # figure 1 acccuracy curves
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(range(1, M + 1), train_accuracy, label="train")
    ax1.plot(range(1, M + 1), test_accuracy, label="test")
    ax1.set_title("Accuracy vs Number of Weak Learners")
    ax1.set_xlabel("Number of Weak Learners")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True)
    ax1.legend()

    # figure 2 strong model decision boundary at best test weak_learner
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle(f'Best Strong Model (n_weak_learners={ntest})')
    plot_decision_boundary(ax2[0], ada, points_train, classes_train, ntest, "adaboost-train-24.txt")
    plot_decision_boundary(ax2[1], ada, points_test, classes_test, ntest, "adaboost-test-24.txt")
    plt.savefig("best_strong_model.png", dpi=500)

    # figure 2 strong model decision boundary at best test weak_learner
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle(f'Best Strong Model (n_weak_learners={n})')
    plot_decision_boundary(ax2[0], ada, points_train, classes_train, n, "adaboost-train-24.txt")
    plot_decision_boundary(ax2[1], ada, points_test, classes_test, n, "adaboost-test-24.txt")
    plt.savefig("100percent_strong_model.png", dpi=500)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    indices = np.arange(2)
    width = 0.35
    ax3.bar(indices - width / 2, [train_accuracy[ntest - 1], train_accuracy[-1]], width, label="train")
    ax3.bar(indices + width / 2, [test_accuracy[ntest - 1], test_accuracy[-1]], width, label="test")
    ax3.set_xticks(indices)
    ax3.set_xticklabels([f"weak_learner={ntest}", f"weak_learner={M}"])
    ax3.set_ylabel("accuracy")
    ax3.set_title(f"train and test accuracy at weak_learner={n} and test and weak_learner={M}")
    ax3.legend()
    ax3.grid(True, axis="y")

    k = min(10, M)
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.bar(range(1, k + 1), test_accuracy[:k])
    ax4.set_xlabel("weak_learner")
    ax4.set_ylabel("test accuracy")
    ax4.set_title("test accuracy for first weak learners")
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, axis="y")

    # plt.show()
    
    ## reversing test and train sets ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('\n~~~~~ Swapping test and train datasets (in the case that they are labelled wrong) ~~~~~~\n')
    
    # load training and test sets
    points_train, classes_train = load_data("adaboost-test-24.txt")  # this has 400 datapoints
    points_test, classes_test = load_data("adaboost-train-24.txt")  # this has 1200 datapoints <-- feels like this should be the training set?

    ## First train model to 100 % to see how many weak learners are needed...
    print("## Training model to 100 % to see how many weak learners are needed ##")

    # set target accuracy to be just below the 100% cause overfitting is bad and makes me sad
    # target_train_accuracy = 1.0 - (1.0 / n_train)
    target_train_accuracy = 1  # set training accuracy to 100 %

    # train adabost strong learner with early stopping
    ada = AdaBoost(
        base_learner_class=WeightedDecisionStump,
        n_weak_learners=792,  # manually setting this since we know its value
        target_train_accuracy=target_train_accuracy,
        avoid_perfect=False  # setting this to true results in 791 weak learners needed
    )
    ada.fit(points_train, classes_train)  # train the strong model

    M = len(ada.learners)
    print("strong model learners used to get to 100%:", M)

    # final train and test accuracy using all learners
    predictions_from_train_set = ada.predict(points_train)
    predictions_from_test_set = ada.predict(points_test)
    accuracy_on_train_set = np.mean(predictions_from_train_set == classes_train)
    accuracy_on_test_set = np.mean(predictions_from_test_set == classes_test)

    print("final train accuracy:", accuracy_on_train_set)
    print("final test accuracy:", accuracy_on_test_set)
    print('')

    # accuracy vs weak_learner curves
    train_accuracy = []
    test_accuracy = []
    for weak_learner in range(1, M + 1):
        train_accuracy.append(np.mean(ada.predict(points_train, weak_learner) == classes_train))
        test_accuracy.append(np.mean(ada.predict(points_test, weak_learner) == classes_test))

    train_accuracy = np.array(train_accuracy)
    test_accuracy = np.array(test_accuracy)

    n = None
    for i, a in enumerate(train_accuracy, start=1):
        if a == 1.0:
            n = i
            break

    max_test_acc = float(np.max(test_accuracy))
    ntest = int(np.argmax(test_accuracy) + 1)

    print('## Iterating over saved model accuracies to find optimal model ## ')
    print("first weak_learner with 100 percent training accuracy n:", n)

    print("max test accuracy:", max_test_acc, "at weak_learner:", ntest)
    print(f"train accuracy at weak_learner={ntest}:", train_accuracy[ntest - 1])
    print(f"test accuracy at weak_learner={ntest}:", test_accuracy[ntest - 1], '<-- this is our best strong model')
    print(f"train accuracy at weak_learner={M}:", train_accuracy[-1])
    print(f"test accuracy at weak_learner={M}:", test_accuracy[-1])

    # figure 1 acccuracy curves
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    ax5.plot(range(1, M + 1), train_accuracy, label="train")
    ax5.plot(range(1, M + 1), test_accuracy, label="test")
    ax5.set_title("Accuracy vs Number of Weak Learners")
    ax5.set_xlabel("Number of Weak Learners")
    ax5.set_ylabel("Accuracy")
    ax5.grid(True)
    ax5.legend()

    # figure 2 strong model decision boundary at best test weak_learner
    fig6, ax6 = plt.subplots(1, 2, figsize=(12, 5))
    fig6.suptitle(f'Best Strong Model (n_weak_learners={ntest})')
    plot_decision_boundary(ax6[0], ada, points_train, classes_train, ntest, "adaboost-train-24.txt")
    plot_decision_boundary(ax6[1], ada, points_test, classes_test, ntest, "adaboost-test-24.txt")
    plt.savefig("best_strong_model.png", dpi=500)

    # figure 2 strong model decision boundary at best test weak_learner
    fig6, ax6 = plt.subplots(1, 2, figsize=(12, 5))
    fig6.suptitle(f'Best Strong Model (n_weak_learners={n})')
    plot_decision_boundary(ax6[0], ada, points_train, classes_train, n, "adaboost-train-24.txt")
    plot_decision_boundary(ax6[1], ada, points_test, classes_test, n, "adaboost-test-24.txt")
    plt.savefig("100percent_strong_model.png", dpi=500)

    fig7, ax7 = plt.subplots(figsize=(8, 5))
    indices = np.arange(2)
    width = 0.35
    ax7.bar(indices - width / 2, [train_accuracy[ntest - 1], train_accuracy[-1]], width, label="train")
    ax7.bar(indices + width / 2, [test_accuracy[ntest - 1], test_accuracy[-1]], width, label="test")
    ax7.set_xticks(indices)
    ax7.set_xticklabels([f"weak_learner={ntest}", f"weak_learner={M}"])
    ax7.set_ylabel("accuracy")
    ax7.set_title(f"train and test accuracy at weak_learner={n} and test and weak_learner={M}")
    ax7.legend()
    ax7.grid(True, axis="y")

    k = min(10, M)
    fig8, ax8 = plt.subplots(figsize=(10, 5))
    ax8.bar(range(1, k + 1), test_accuracy[:k])
    ax8.set_xlabel("weak_learner")
    ax8.set_ylabel("test accuracy")
    ax8.set_title("test accuracy for first weak learners")
    ax8.set_ylim(0, 1.05)
    ax8.grid(True, axis="y")

    plt.show()


if __name__ == "__main__":
    main()
