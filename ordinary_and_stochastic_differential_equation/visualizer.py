from abc import ABC, abstractmethod
from typing import Optional
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
import torch
import torch.distributions as D
from torch.func import vmap, jacrev
from tqdm import tqdm
import seaborn as sns

from package import ODE, SDE, Simulator, BrownianMotion, EulerSimulator, EulerMaruyamaSimulator, OUProcess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_trajectories_1d(x0: torch.Tensor, simulator: Simulator, timesteps: torch.Tensor, ax: Optional[Axes] = None, show_hist: bool = False, decouple_hist_axis: bool = False):
        """
        Graphs the trajectories of a one-dimensional SDE with given initial values (x0) and simulation timesteps (timesteps).
        Args:
            - x0: state at time t, shape (num_trajectories, 1)
            - simulator: Simulator object used to simulate
            - t: timesteps to simulate along, shape (num_timesteps,)
            - ax: pyplot Axes object to plot on
            - decouple_hist_axis: if True, do not share y-axis between trajectories and histogram
        """
        if ax is None:
            ax = plt.gca()
        trajectories = simulator.simulate_with_trajectory(x0, timesteps) # (num_trajectories, num_timesteps, ...)

        line_color = sns.color_palette("crest", 1)[0]
        hist_color = sns.color_palette("flare", 1)[0]
        label_size = 12
        tick_size = 10

        timesteps_cpu = timesteps.detach().cpu().numpy()
        for trajectory_idx in range(trajectories.shape[0]):
            trajectory = trajectories[trajectory_idx, :, 0].detach().cpu().numpy() # (num_timesteps,)
            sns.lineplot(
                x=timesteps_cpu,
                y=trajectory,
                ax=ax,
                color=line_color,
                alpha=0.45,
                linewidth=1.1,
                legend=False,
            )

        ax.set_xlabel(r"time ($t$)", fontsize=label_size)
        ax.set_ylabel(r"$X_t$", fontsize=label_size)
        ax.tick_params(axis='both', labelsize=tick_size)
        ax.grid(alpha=0.2, linewidth=0.6)

        if show_hist:
            terminal_points = trajectories[:, -1, 0].detach().cpu().numpy()
            data_range = float(terminal_points.max() - terminal_points.min()) if terminal_points.size else 1.0
            binwidth = max(data_range / 25.0, 0.05)

            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            sharey = None if decouple_hist_axis else ax
            hist_ax = divider.append_axes("right", size="22%", pad=0.45, sharey=sharey)
            sns.histplot(
                y=terminal_points,
                ax=hist_ax,
                binwidth=binwidth,
                color=hist_color,
                alpha=0.7,
                edgecolor="white",
                linewidth=0.5,
            )
            hist_ax.set_xlabel("count", fontsize=label_size)
            hist_ax.set_ylabel("")
            hist_ax.tick_params(axis='both', labelsize=tick_size)
            if decouple_hist_axis:
                hist_ax.tick_params(axis='y', left=True, labelleft=True)
            else:
                hist_ax.tick_params(axis='y', left=False, labelleft=False)
            hist_ax.grid(axis='x', alpha=0.2, linewidth=0.6)

        fig = ax.figure
        if fig is not None:
            title = ax.get_title()
            if title:
                title_size = ax.title.get_fontsize()
                ax.set_title("")

                axes = [ax]
                if show_hist:
                    axes.append(hist_ax)

                fig.canvas.draw()
                bboxes = [a.get_position() for a in axes]

                left = min(b.x0 for b in bboxes)
                right = max(b.x1 for b in bboxes)
                top = max(b.y1 for b in bboxes)

                x_center = 0.5 * (left + right)
                y = top + 0.005

                fig.text(
                    x_center,
                    y,
                    title,
                    ha="center",
                    va="bottom",
                    fontsize=title_size,
                )

sigma = 1.0
n_traj = 500
brownian_motion = BrownianMotion(sigma)
simulator = EulerMaruyamaSimulator(sde=brownian_motion)
x0 = torch.zeros(n_traj,1).to(device) # Initial values - let's start at zero
ts = torch.linspace(0.0,5.0,500).to(device) # simulation timesteps

plt.figure(figsize=(9, 6))
ax = plt.gca()
ax.set_title(r'Trajectories of Brownian Motion with $\sigma=$' + str(sigma), fontsize=18)
ax.set_xlabel(r'time ($t$)', fontsize=18)
ax.set_ylabel(r'$x_t$', fontsize=18)
plot_trajectories_1d(x0, simulator, ts, ax, show_hist=True)
plt.show()

# Try comparing multiple choices side-by-side
thetas_and_sigmas = [
    (0.25, 0.0),
    (0.25, 0.5),
    (0.25, 2.0),
]
simulation_time = 10.0

num_plots = len(thetas_and_sigmas)
fig, axes = plt.subplots(2, num_plots, figsize=(10.5 * num_plots, 15))

# Top row: dynamics
n_traj = 10
for idx, (theta, sigma) in enumerate(thetas_and_sigmas):
    ou_process = OUProcess(theta, sigma)
    simulator = EulerMaruyamaSimulator(sde=ou_process)
    x0 = torch.linspace(-10.0,10.0,n_traj).view(-1,1).to(device) # Initial values - let's start at zero
    ts = torch.linspace(0.0,simulation_time,1000).to(device) # simulation timesteps

    ax = axes[0,idx]
    ax.set_title(f'Trajectories of OU Process with $\\sigma = ${sigma}, $\\theta = ${theta}', fontsize=15)
    plot_trajectories_1d(x0, simulator, ts, ax, show_hist=False)

# Bottom row: distribution
n_traj = 500
for idx, (theta, sigma) in enumerate(thetas_and_sigmas):
    ou_process = OUProcess(theta, sigma)
    simulator = EulerMaruyamaSimulator(sde=ou_process)
    x0 = torch.linspace(-10.0,10.0,n_traj).view(-1,1).to(device) # Initial values - let's start at zero
    ts = torch.linspace(0.0,simulation_time,1000).to(device) # simulation timesteps

    ax = axes[1,idx]
    ax.set_title(f'Trajectories of OU Process with $\\sigma = ${sigma}, $\\theta = ${theta}', fontsize=15)
    ax = plot_trajectories_1d(x0, simulator, ts, ax, show_hist=True, decouple_hist_axis=True)
plt.show()

# Let's compare various OU processes!
sigmas = [1.0, 2.0, 10.0]
ds = [0.25, 1.0, 4.0] # sigma**2 / 2t
simulation_time = 15.0
n_traj = 500

fig, axes = plt.subplots(len(ds), len(sigmas), figsize=(8 * len(sigmas), 8 * len(ds)))
axes = axes.reshape((len(ds), len(sigmas)))
for d_idx, d in enumerate(ds):
    for s_idx, sigma in enumerate(sigmas):
        theta = sigma**2 / 2 / d
        ou_process = OUProcess(theta, sigma)
        simulator = EulerMaruyamaSimulator(sde=ou_process)
        x0 = torch.linspace(-20.0,20.0,n_traj).view(-1,1).to(device)
        time_scale = sigma**2
        ts = torch.linspace(0.0,simulation_time / time_scale,1000).to(device) # simulation timesteps
        ax = axes[d_idx, s_idx]
        ax.set_title(f'OU Trajectories with Sigma={sigma}, Theta={theta}, D={d}')
        plot_trajectories_1d(x0=x0, simulator=simulator, timesteps=ts, ax=ax, show_hist=True, decouple_hist_axis=True)
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'X_t')
plt.show()