#ce4041 adaboost project
#name: Kanvar Murray
#student id: 22374698
#date: 21/11/25

import numpy as np
import matplotlib.pyplot as plt
from weak_linear import WeightedWeakLinear, WeightedDecisionStump
from adaboost import AdaBoost

def load_data(filename):
    data=np.loadtxt(filename)
    X=data[:,:2]
    y=data[:,2].astype(int)
    return X,y





#plot decision boundary
def plot_decision_boundary(ax,ada,X,y,T,title):
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches

    #SHAPES AND COLOURS!
    x_min,x_max=X[:,0].min()-0.5,X[:,0].max()+0.5
    y_min,y_max=X[:,1].min()-0.5,X[:,1].max()+0.5
    xx,yy=np.meshgrid(
        np.linspace(x_min,x_max,300),
        np.linspace(y_min,y_max,300)
    )
    grid=np.c_[xx.ravel(),yy.ravel()]
    Z=ada.predict(grid,T=T).reshape(xx.shape)
    cmap=ListedColormap(["#ffcccc","#ccccff"])
    ax.contourf(xx,yy,Z,levels=[-1,0,1],cmap=cmap,alpha=0.4)

    #plot training points
    class1=(y==1)
    class2=(y==-1)
    #SHAPES AND COLOURS!!
    p1=ax.scatter(X[class1,0],X[class1,1],marker='o',label='class +1')
    p2=ax.scatter(X[class2,0],X[class2,1],marker='x',label='class -1')
    patch_neg=mpatches.Patch(color="#ffcccc",label="predict -1")
    patch_pos=mpatches.Patch(color="#ccccff",label="predict +1")

    ax.set_title(f"{title} (T={T})")
    ax.grid(True)
    ax.legend(handles=[p1,p2,patch_pos,patch_neg])




def main():

    #load training and test sets
    X_train,y_train=load_data("adaboost-train-24.txt")
    X_test,y_test=load_data("adaboost-test-24.txt")

    n_train=X_train.shape[0]

    #train weak linear alone all by itsself so sad :(
    uniform=np.ones(n_train)/n_train
    weak=WeightedWeakLinear()
    weak.fit(X_train,y_train,uniform)
    y_pred_weak=weak.predict(X_train)
    weak_acc=np.mean(y_pred_weak==y_train)
    print("weak linear training accuracy:",weak_acc)

    #set target accuracy to be just below the 100% cause overfitting is bad and makes me sad
    target_train_acc=1.0-1.0/n_train

    #train adabost with early stopping
    ada=AdaBoost(
        base_learner_class=WeightedDecisionStump,
        n_estimators=200,
        target_train_accuracy=target_train_acc,
        avoid_perfect=True
    )
    ada.fit(X_train,y_train)

    M=len(ada.learners)
    print("strong model learners used:",M)

    #final train and test accuracy using all learners
    y_train_pred=ada.predict(X_train)
    y_test_pred=ada.predict(X_test)
    train_final_acc=np.mean(y_train_pred==y_train)
    test_final_acc=np.mean(y_test_pred==y_test)

    print("final train accuracy:",train_final_acc)
    print("final test accuracy:",test_final_acc)

    #accuracy vs T curves
    train_acc=[]
    test_acc=[]
    for T in range(1,M+1):
        train_acc.append(np.mean(ada.predict(X_train,T)==y_train))
        test_acc.append(np.mean(ada.predict(X_test,T)==y_test))

    train_acc=np.array(train_acc)
    test_acc=np.array(test_acc)

    n=None
    for i,a in enumerate(train_acc,start=1):
        if a==1.0:
            n=i
            break

    max_test_acc=float(np.max(test_acc))
    ntest=int(np.argmax(test_acc)+1)
    print("first T with 100 percent training accuracy n:",n)
    print("max test accuracy:",max_test_acc,"at T:",ntest)
    print("train accuracy at T=ntest:",train_acc[ntest-1])
    print("test accuracy at T=ntest:",test_acc[ntest-1])
    print("train accuracy at T=M:",train_acc[-1])
    print("test accuracy at T=M:",test_acc[-1])

    #figure 1 acccuracy curves
    fig1,ax1=plt.subplots(figsize=(10,5))
    ax1.plot(range(1,M+1),train_acc,label="train")
    ax1.plot(range(1,M+1),test_acc,label="test")
    ax1.set_title("accuracy vs number of weak learners")
    ax1.set_xlabel("T")
    ax1.set_ylabel("acccuracy")
    ax1.grid(True)
    ax1.legend()

    #figure 2 strong model decision boundary at best test T
    fig2,ax2=plt.subplots(1,2,figsize=(12,5))
    plot_decision_boundary(ax2[0],ada,X_train,y_train,ntest,"Final Stong model")
    plot_decision_boundary(ax2[1],ada,X_test,y_test,ntest,"Final Stong model")

    fig3,ax3=plt.subplots(figsize=(8,5))
    indices=np.arange(2)
    width=0.35
    ax3.bar(indices-width/2,[train_acc[ntest-1],train_acc[-1]],width,label="train")
    ax3.bar(indices+width/2,[test_acc[ntest-1],test_acc[-1]],width,label="test")
    ax3.set_xticks(indices)
    ax3.set_xticklabels([f"T={ntest}",f"T={M}"])
    ax3.set_ylabel("accuracy")
    ax3.set_title("train and test accuracy at T=ntest and T=M")
    ax3.legend()
    ax3.grid(True,axis="y")

    k=min(10,M)
    fig4,ax4=plt.subplots(figsize=(10,5))
    ax4.bar(range(1,k+1),test_acc[:k])
    ax4.set_xlabel("T")
    ax4.set_ylabel("test accuracy")
    ax4.set_title("test accuracy for first weak learners")
    ax4.set_ylim(0,1.05)
    ax4.grid(True,axis="y")

    plt.show()

if __name__=="__main__":
    main()
