import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import zoom
import torch

KEY_TO_LABEL = {'loss': 'Loss', 'rmsd': 'RMSD', 'lr_density': 'Learning Rate Density', 't': 'Diffusion Time',
                'loss_d': 'Density Error', 'loss_m': 'Model Error', 'loss_s': 'Sequence Loss',
                'loss_dist': 'Distance Error', 'loss_p': 'Profile Error', 'rmsd_ca': 'RMSD CA', 'rho_hqs': 'Rho HQS', 'resolution': 'Resolution',
                'sampling_rate': 'Sampling Rate', 'loss_c': 'Inter-CA Loss'}
KEY_TO_LOG = {'loss': True, 'rmsd': True, 'lr_density': True, 't': False, 'loss_d': False, 'loss_m': True,
              'loss_s': True, 'loss_dist': True, 'rmsd_ca': True, 'rho_hqs': True, 'resolution': False,
              'sampling_rate': False, 'loss_c': True}


def plot_density(density, extent=1., res=32):
    """
    density: [res_in, res_in, res_in] (numpy)
    """
    res_in = density.shape[0]
    density_ds = zoom(density, (res / res_in, res / res_in, res / res_in))

    grid_1d = np.linspace(-extent, extent, res)
    X, Y, Z = np.meshgrid(grid_1d, grid_1d, grid_1d)

    isolevel_min = 0.1 * np.percentile(density_ds, 99.9)
    isolevel_max = np.percentile(density_ds, 99)

    plt.figure(figsize=(5, 2))
    bins = plt.hist(density_ds.flatten(), bins=50, histtype='step')
    plt.vlines([isolevel_min, isolevel_max], 0., np.max(bins[0]), color='k')
    plt.yscale('log')
    plt.yticks([])
    plt.show()

    fig = go.Figure(
        data=go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=density_ds.flatten(),
            isomin=isolevel_min,
            isomax=isolevel_max,
            surface_count=3,
            opacity=0.6,
            caps=dict(x_show=False, y_show=False)
        ),
        layout=go.Layout(height=500, width=500)
    )
    fig.show()


def save_trajectory(trajectory, name, total_time=3):
    X_gt = trajectory[0]
    Y = trajectory[1]
    if len(trajectory[1]) == 2:
        X = trajectory[1][0]
    else:
        X = trajectory[1]

    fig, ax = plt.subplots()

    xmin = np.min(X_gt[..., 0])
    xmax = np.max(X_gt[..., 0])
    xrange = xmax - xmin
    ymin = np.min(X_gt[..., 1])
    ymax = np.max(X_gt[..., 1])
    yrange = ymax - ymin
    xmin_plt = xmin - 0.05 * xrange
    xmax_plt = xmax + 0.05 * xrange
    ymin_plt = ymin - 0.05 * yrange
    ymax_plt = ymax + 0.05 * yrange

    scat_gt = ax.scatter(X_gt[:, :, 1, 0], X_gt[:, :, 1, 1], color='orange', label='GT', alpha=.5)
    line_gt = ax.plot(X_gt[0, :, 1, 0], X_gt[0, :, 1, 1], color='orange', linestyle='--')[0]
    scat_ma = ax.scatter(Y[:, :, 1, 0], Y[:, :, 1, 1], color='red', label='MA', alpha=.5)
    scat = ax.scatter(X[:, :, 1, 0], X[:, :, 1, 1], color='blue', alpha=.5)
    line = ax.plot(X[0, :, 1, 0], X[0, :, 1, 1], color='blue', linestyle='--')[0]
    ax.set(xlim=[xmin_plt, xmax_plt], ylim=[ymin_plt, ymax_plt])
    ax.legend()

    def update(i):
        if len(trajectory[i]) == 2:
            X = trajectory[i][0]
        else:
            X = trajectory[i]
        data = np.stack([X[0, :, 1, 0], X[0, :, 1, 1]]).T
        scat.set_offsets(data)
        line.set_xdata(X[0, :, 1, 0])
        line.set_ydata(X[0, :, 1, 1])
        return (scat, line)

    interval_ms = int((total_time * 1000) / (len(trajectory) - 1))
    ani = FuncAnimation(fig=fig, func=update, frames=range(2, len(trajectory)), interval=interval_ms)
    ani.save(name, writer='ffmpeg', fps=int(1000 / interval_ms))


def plot_metric(metrics, key, name):
    plt.figure(figsize=(7, 5), dpi=200)
    if isinstance(metrics[key][0], list):
        metric = np.array(metrics[key]).reshape(-1, len(metrics[key][0]))
        for i in range(metric.shape[1]):
            plt.plot(metrics['epoch'], metric[:, i])
    else:
        plt.plot(metrics['epoch'], metrics[key])
    plt.ylabel(KEY_TO_LABEL[key])
    plt.xlabel('Epoch')
    if KEY_TO_LOG[key]:
        plt.yscale('log')
    plt.grid(True)
    plt.show()
    plt.savefig(name, bbox_inches='tight')


def plot_rmsd_ca_vs_completeness(X_gt, X, mask_gt, name):
    n_residues = X.shape[1]
    distance_sq = ((X[:, mask_gt, 1] - X_gt[:, mask_gt, 1]) ** 2).sum(-1).cpu().numpy()
    rmsd_sorted = np.sqrt(np.cumsum(np.sort(distance_sq)) / (np.arange(mask_gt.sum().item()) + 1))
    completeness = 100. * np.arange(mask_gt.sum().item()) / n_residues
    plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(completeness, rmsd_sorted, color='teal', label='ADP-3D', linewidth=3)
    plt.xlim(0, 100)
    plt.ylim(0.05, 2. * np.max(rmsd_sorted))
    plt.ylabel('RMSD CA')
    plt.xlabel('Completeness')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.grid(which='both')
    plt.show()
    plt.savefig(name, bbox_inches='tight')

def plot_SAXS_profile(profile, exp_profile, chi_square, fitted_params, name):
    plt.figure(figsize=(6, 4), dpi=300)
    q_exp = exp_profile.q_
    I_exp = exp_profile.intensity_
    error_exp = exp_profile.error_
    plt.scatter(q_exp, I_exp, s=0.5, label='Exp. Profile')
    plt.errorbar(q_exp, I_exp, yerr=error_exp, fmt='none', elinewidth=0.5, alpha=0.6)
    q = profile.q_
    I = profile.intensity_
    plt.plot(q, I, label='Model Profile', color='red')
    plt.yscale('log')
    plt.xlabel('q (\u00c5⁻¹)')
    plt.ylabel('log(I(q))')
    plt.legend()
    plt.grid(True)
    plt.title(f'SAXS Profile Comparison\n\u03C7\u00B2 = {chi_square:.2f} \n c1 = {fitted_params.c1:.2f}, c2 = {fitted_params.c2:.2f}')
    plt.show()
    plt.savefig(name, bbox_inches='tight')
