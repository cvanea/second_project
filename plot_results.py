import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    sample = 1
    exp_name = "ICA"

    results = pd.read_csv("Results/log_reg/{}/sample{}.csv".format(exp_name, sample), index_col=0)

    sns.set()
    ax = sns.lineplot(data=results, dashes=False)
    ax.set(ylim=(0, 0.6), xlabel='Time', ylabel='Accuracy', title='Cross Val Accuracy Logistic Regression')
    plt.axvline(x=15, color='b', linestyle='--')
    plt.show()


if __name__ == "__main__":
    main()