from matplotlib import pyplot as plt

from utils.util_funcs import Bunch


def main(args):
    bunch = Bunch(args)
    performance = bunch.performance

    # TODO: edit plot_model_results to accommodate multiple outcomes
    # make one subplot for each non-nan metric
    # for all sets, plot data with label, add legend
    # suptitle as below
    # performance[:performance['estop_epoch'].values].loc[dict(metrics=['loss','accuracy'])]

    metrics = ['MAE', 'pearsonR', 'loss']
    fig, axs = plt.subplots(nrows=len(performance.outcome.values), ncols=len(metrics),
                            figsize=(len(metrics) * 6, len(performance.outcome.values) * 2), sharex=True)
    fig.subplots_adjust(wspace=.2, hspace=.2, top=.9, bottom=.05)
    axs = axs.ravel()
    fig.suptitle(
        f"BrainNetCNN trained to predict {', '.join(bunch.predicted_outcome)} \nfrom {', '.join(bunch.chosen_Xdatavars)}")

    count = 0

    for j, outcome in enumerate(performance.outcome.values):
        for i, metric in enumerate(metrics):

            axs[count].plot(performance.loc[dict(set='train', metrics=metric, outcome=outcome)].values.squeeze(),
                            label='train')
            axs[count].plot(performance.loc[dict(set='test', metrics=metric, outcome=outcome)].values.squeeze(),
                            label='test')
            axs[count].set_title(f"{outcome}, {metric}")
            axs[count].set_ylabel(f'{metric}')
            axs[count].legend()

            if j == len(performance.outcome.values) - 1:
                axs[count].set_xlabel('epochs')

            count += 1


if __name__ == '__main__':
    main()
