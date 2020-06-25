import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    results = pd.read_csv("Results/linear_svm/sample1.csv", index_col=0)

    sns.set()
    ax = sns.lineplot(data=results, dashes=False)
    ax.set(ylim=(0, 1), xlabel='Time', ylabel='Accuracy', title='Cross Val Accuracy Linear SVM')
    plt.show()


if __name__ == "__main__":
    main()