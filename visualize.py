import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_predictions(dataset_name: str, outname: str, true_path, pred_path, true_mask=None, pred_mask=None, true_mag=None, pred_mag=None):
    """Save prediction visualization inside `outputs/val/{dataset_name}/{outname}`.
    Arguments:
    - `true_path`, `pred_path`: 1D arrays (required)
    - `true_mask`, `pred_mask`: 1D arrays or None
    - `true_mag`, `pred_mag`: 1D arrays or None
    """
    base_dir = os.path.join('outputs', 'val', dataset_name)
    os.makedirs(base_dir, exist_ok=True)
    outpath = os.path.join(base_dir, outname)

    L = len(true_path)
    x = np.arange(L)
    # determine number of subplots
    nplots = 1 + (1 if (true_mask is not None or pred_mask is not None) else 0) + (1 if (true_mag is not None or pred_mag is not None) else 0)
    fig, axs = plt.subplots(nplots, 1, figsize=(10, 3 * nplots), constrained_layout=True)
    if nplots == 1:
        axs = [axs]

    axs[0].plot(x, true_path, label='true_path', lw=1)
    axs[0].plot(x, pred_path, label='pred_path', lw=1)
    axs[0].legend(); axs[0].set_title(outname)

    idx = 1
    if (true_mask is not None) or (pred_mask is not None):
        tm = np.zeros(L) if true_mask is None else np.asarray(true_mask)[:L]
        pm = np.zeros(L) if pred_mask is None else np.asarray(pred_mask)[:L]
        axs[idx].plot(x, tm, label='true_mask', lw=1)
        axs[idx].plot(x, pm, label='pred_mask', lw=1)
        axs[idx].legend(); axs[idx].set_title('Mask (volume / predicted)')
        idx += 1

    if (true_mag is not None) or (pred_mag is not None):
        tg = np.zeros(L) if true_mag is None else np.asarray(true_mag)[:L]
        pg = np.zeros(L) if pred_mag is None else np.asarray(pred_mag)[:L]
        axs[idx].plot(x, tg, label='true_mag', lw=1)
        axs[idx].plot(x, pg, label='pred_mag', lw=1)
        axs[idx].legend(); axs[idx].set_title('Jump Magnitudes')

    fig.savefig(outpath)
    plt.close(fig)


def plot_training_curves(loss_list, outname='training_loss.png'):
    os.makedirs(os.path.join('outputs'), exist_ok=True)
    outpath = os.path.join('outputs', outname)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(loss_list) + 1), loss_list, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(outpath)
    plt.close()

