import numpy as np
import torch
import xarray as xr


def main():
    # Adjust to load desired model
    loadpath_model = input('Filepath of model.pt : ')

    model = torch.load(loadpath_model)
    model.eval()

    loadpath_performance = input('Filepath of performance.nc : ')
    performance = xr.load_dataset(loadpath_performance)

    try:
        if len(performance.outcome) > 1:
            best_test_epoch = performance[list(performance.data_vars)[0]].loc[
                dict(set='test', metrics='MAE')].mean(axis=-1).argmin().values
        else:
            best_test_epoch = performance[list(performance.data_vars)[0]].loc[
                dict(set='test', metrics='MAE')].argmin().values
    except ValueError:
        best_test_epoch = performance[list(performance.data_vars)[0]].loc[
            dict(set='test', metrics='accuracy')].argmax().values

    # formatting float print-out
    float_formatter = "{:.3f}".format
    np.set_printoptions(formatter={'float_kind': float_formatter})

    print(f'\nBest test performance'
          f'\nepoch: {best_test_epoch}'
          f"\nMAE: {performance[list(performance.data_vars)[0]].loc[dict(set='test', metrics='MAE', epoch=best_test_epoch)].values.squeeze()}"
          f"\npearson R: {performance[list(performance.data_vars)[0]].loc[dict(set='test', metrics='pearsonR', epoch=best_test_epoch)].values.squeeze()}"
          f"\npearson p-value: {performance[list(performance.data_vars)[0]].loc[dict(set='test', metrics='p_value', epoch=best_test_epoch)].values.squeeze()}"
          f"\naccuracy: {performance[list(performance.data_vars)[0]].loc[dict(set='test', metrics='accuracy', epoch=best_test_epoch)].values.squeeze()}")


if __name__ == '__main__':
    main()
