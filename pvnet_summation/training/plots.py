"""Plots logged during training"""
from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import pylab
import torch
import wandb

from pvnet_summation.data.datamodule import SumTensorBatch


def wandb_line_plot(
    x: Sequence[float], 
    y: Sequence[float], 
    xlabel: str, 
    ylabel: str, 
    title: str | None = None,
) -> wandb.plot.CustomChart:
    """Make a wandb line plot"""
    data = [[xi, yi] for (xi, yi) in zip(x, y)]
    table = wandb.Table(data=data, columns=[xlabel, ylabel])
    return wandb.plot.line(table, xlabel, ylabel, title=title)


def plot_sample_forecasts(
    batch: SumTensorBatch,
    y_hat: torch.Tensor,
    y_loc_sum: torch.Tensor,
    quantiles: list[float] | None,
) -> plt.Figure:
    """Plot a batch of data and the forecast from that batch"""

    y = batch["target"].cpu().numpy()
    y_hat = y_hat.cpu().numpy()
    y_loc_sum = y_loc_sum.cpu().numpy()
    times_utc = pd.to_datetime(batch["valid_times"].cpu().numpy().astype("datetime64[ns]"))
    batch_size = y.shape[0]

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for i, ax in enumerate(axes.ravel()[:batch_size]):

        ax.plot(times_utc[i], y[i], marker=".", color="k", label=r"$y$")

        ax.plot(
            times_utc[i],
            y_loc_sum[i], 
            marker=".",
            linestyle="-.",
            color="r",
            label=r"$\hat{y}_{loc-sum}$",
        )

        if quantiles is None:
            ax.plot(
                times_utc[i],
                y_hat[i], 
                marker=".", 
                color="r", 
                label=r"$\hat{y}$",
            )
        else:
            cm = pylab.get_cmap("twilight")
            for nq, q in enumerate(quantiles):
                ax.plot(
                    times_utc[i],
                    y_hat[i, :, nq],
                    color=cm(q),
                    label=r"$\hat{y}$" + f"({q})",
                    alpha=0.7,
                )

        ax.set_title(f"{times_utc[i][0].date()}", fontsize="small")

        xticks = [t for t in times_utc[i] if t.minute == 0][::2]
        ax.set_xticks(ticks=xticks, labels=[f"{t.hour:02}" for t in xticks], rotation=90)
        ax.grid()

    axes[0, 0].legend(loc="best")

    if batch_size<16:
        for ax in axes.ravel()[batch_size:]:
            ax.axis("off")
    
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (hour of day)")

    plt.tight_layout()

    return fig
