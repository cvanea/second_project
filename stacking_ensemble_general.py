import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils import get_y_train


def main():
    base_model_type = "lda"
    base_model_dir = "wavelet_class/lsqr/complex"
    save_dir = "Results/stacking_ensemble/"

    load_dir = "Results/{}/{}".format(base_model_type, base_model_dir)

    # (21, 15, 50, epochs, 8)
    all_sample_preds = np.array(pickle.load(open(load_dir + "/all_proba_preds.pkl", "rb")))

    # (21, all_time)
    all_y_train = []

    for sample in range(1, 22):
        all_y_train.append(get_y_train(sample))

    results = np.zeros(50)

    for time in range(50):
        print("time {}".format(time))

        # (21, 15, epochs, 8)
        sample_preds = all_sample_preds[:, :, time]

        # (21, epochs)
        sample_y_train = []

        for sample in range(21):
            intervals = np.arange(start=time, stop=all_y_train[sample].shape[0], step=50)
            sample_y_train.append(all_y_train[sample][intervals])

        sample_predictions_proba = [np.vstack(sample).astype(np.float) for sample in sample_preds]

        sample_y_train = np.array(sample_y_train)
        sample_y_train = np.repeat(sample_y_train[:, np.newaxis], 15, axis=1)
        sample_y_train = [np.vstack(sample).astype(np.int) for sample in sample_y_train]
        final = []
        for sample in range(21):
            final.append([data for freq_data in sample_y_train[sample] for data in freq_data])
        sample_y_train = final

        all_val_acc = []

        for val_sample in range(21):
            print("left out validation subject: {}".format(val_sample + 1))

            x_train = sample_predictions_proba[:val_sample] + sample_predictions_proba[val_sample + 1:]
            x_train = [data for freq_data in x_train for data in freq_data]
            y_train = sample_y_train[:val_sample] + sample_y_train[val_sample + 1:]
            y_train = [data for freq_data in y_train for data in freq_data]

            x_val = sample_predictions_proba[val_sample]
            # x_val = [data for freq_data in x_val for data in freq_data]
            y_val = sample_y_train[val_sample]
            # y_val = [data for freq_data in y_val for data in freq_data]

            meta_model = LogisticRegression()
            meta_model.fit(x_train, y_train)

            acc_score = meta_model.score(x_val, y_val)
            all_val_acc.append(acc_score)

        all_val_acc = np.array(all_val_acc)
        avg_val_acc = np.mean(all_val_acc, axis=0)
        print("average cross val score: {}".format(avg_val_acc))
        results[time] = avg_val_acc

    sns.set()
    ax = sns.lineplot(data=results, dashes=False)
    ax.set(ylim=(0, 1), xlabel='Timepoints', ylabel='Accuracy',
           title='Average Cross Val Accuracy Stacking Ensemble {} Base Models'.format(base_model_type))
    plt.axvline(x=15, color='b', linestyle='--')
    ax.figure.savefig("{}/LOOCV.png".format(save_dir), dpi=300)
    plt.clf()

    results_df = pd.DataFrame(results)
    results_df.to_csv("{}/LOOCV.csv".format(save_dir))


if __name__ == "__main__":
    main()
