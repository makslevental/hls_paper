import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import ticker, font_manager
import matplotlib.patheffects as PathEffects
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D


font = {
    # "family": "normal",
    "size": 18
}

matplotlib.rc("font", **font)


def myticks(x, pos):
    if x <= 0:
        return f"${x}$"
    exponent = int(np.log2(x))
    coeff = x / 2 ** exponent
    # return r"$2^{{ {:2d} }}$".format(exponent)
    return str(x)


def ticks_format(value, index):
    """
    get the value and returns the value as:
       integer: [0,99]
       1 digit float: [0.1, 0.99]
       n*10^m: otherwise
    To have all the number of the same size they are all returned as latex strings
    """
    if index % 2:
        return ""

    exp = np.floor(np.log10(value))
    base = value / 10 ** exp
    if exp == 0 or exp == 1:
        return "${0:d}$".format(int(value))
    if exp == -1:
        return "${0:.1f}$".format(value)
    else:
        # return "${0:d}\\times 10^{{{1:d}}}$".format(int(base), int(exp))
        # return "${0:d}e{{{1:d}}}$".format(int(base), int(exp))
        return ""


BAR_CHART_COLORS = {
    "DSP": (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
    "BRAM": (1.0, 0.4980392156862745, 0.054901960784313725, 1.0),
    "LUT": (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0),
    "FF": (0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0),
}
PATTERNS = {"DSP": "+", "BRAM": "x", "LUT": "o", "FF": "O"}


def plot_metrics(sub_df, ax1, spacing=4, scale=10):
    i = 0
    for metric in ["DSP", "BRAM", "LUT", "FF"]:
        if metric not in dict(sub_df.index.values):
            continue

        vals = sub_df.loc[metric].sort_index()
        xs = np.array(vals.index.values)

        if xs[-1] == 1024:
            xs[-1] = xs[-2] + spacing

        vals = vals.values.flatten()
        if np.all(vals == 0):
            continue
        diffs = np.diff(xs) / scale
        if not len(diffs):
            diffs = [spacing / scale]
        diffs = np.append(diffs, [diffs[-1] * 1])
        ax1.bar(
            xs + diffs * (-1 + i),
            vals,
            width=diffs,
            log=False,
            ec="k",
            label=metric,
            color=BAR_CHART_COLORS[metric],
            hatch=PATTERNS[metric],
        )
        i += 1


def make_tables():
    vitis_df = pd.read_csv("vitis_reports.csv", header=0, index_col=[0, 1, 2])
    bragghls_df = pd.read_csv("bragghls_reports.csv", header=0, index_col=[0, 1, 2])
    with open("vitis_unroll.tex", "w") as f:
        piv = vitis_df.reset_index().pivot(
            index=["module", "unroll_factor"],
            columns=["metric_name"],
            values=["metric_val"],
        )
        piv = piv.drop(
            columns=[
                ("metric_val", "LatencyAvg"),
                ("metric_val", "LatencyWorst"),
                ("metric_val", "PipelineII"),
            ],
        )
        piv = piv.astype(
            {
                ("metric_val", "LatencyBest"): "int32",
            }
        )
        piv.columns = piv.columns.remove_unused_levels()
        piv.columns = piv.columns.set_levels(
            ["BRAM", "DSP", "FF", "LUT", "Latency", "Clock Period", "Runtime"], level=1
        )
        f.write(
            piv.to_latex(
                longtable=True,
                float_format="%.2E",
                caption="Latency, resource usage, and runtimes for all \\texttt{BraggHLS} evaluations.",
                label="tab:all_unrolls",
            )
        )
    with open("bragghls_unroll.tex", "w") as f:
        piv = bragghls_df.reset_index().pivot(
            index=["module", "unroll_factor"],
            columns=["metric_name"],
            values=["metric_val"],
        )
        # piv.drop(columns=["LatencyAvg", "LatencyWorst"])
        f.write(piv.to_latex(float_format="%.2E"))


def plot_unrolls():
    vitis_df = pd.read_csv("vitis_reports.csv", header=0, index_col=[0, 1, 2])
    bragghls_df = pd.read_csv("bragghls_reports.csv", header=0, index_col=[0, 1, 2])
    for i, mod in enumerate(vitis_df.index.levels[0]):
        fig, (lat_ax_vitis, resource_ax_bragghls) = plt.subplots(
            1, 2, figsize=(10, 7), gridspec_kw={"width_ratios": [10, 1]}
        )
        resource_ax_vitis = lat_ax_vitis.twinx()
        for ax in [
            resource_ax_bragghls,
            resource_ax_vitis,
            lat_ax_vitis,
        ]:
            ax.set_yscale("log")

        resource_ax_vitis.grid(which="both", axis="y", linestyle="--", zorder=-1)
        resource_ax_vitis.set_axisbelow(True)
        resource_ax_bragghls.grid(which="both", axis="y", linestyle="--", zorder=-1)
        resource_ax_bragghls.set_axisbelow(True)

        resource_ax_bragghls.set_xlim(2046, 2050)

        sub_df = vitis_df.loc[mod].swaplevel(0, 1)
        plot_metrics(sub_df, resource_ax_vitis)
        if mod in bragghls_df.index.levels[0]:
            bragghls_sub_df = bragghls_df.loc[mod].swaplevel(0, 1)
            plot_metrics(bragghls_sub_df, resource_ax_bragghls)

        if mod in {"addmm", "conv", "soft_max", "braggnn"}:
            lat_ax_vitis.set_ylabel("time (μs)", fontdict={"fontsize": 25})
        if mod in {"batch_norm", "max_pool_2d", "soft_max", "braggnn"}:
            resource_ax_bragghls.set_ylabel("utilization (%)", fontdict={"fontsize": 25})
        lat_ax_vitis.set_xlabel("unroll factor", fontdict={"fontsize": 25})

        clock_vals = sub_df.loc["clock_period_minus_wns"].sort_index()
        latency_vals = sub_df.loc["LatencyBest"].sort_index()
        common_unroll_factors = sorted(
            set(latency_vals.index.values) & set(clock_vals.index.values)
        )
        times = (
            clock_vals.loc[common_unroll_factors].values
            * latency_vals.loc[common_unroll_factors].values
            / 1e3
        ).flatten()

        last_pos = [
            common_unroll_factors[-2] + 4
            if common_unroll_factors[-1] == 1024
            else common_unroll_factors[-1]
        ]
        lat_ax_vitis.plot(
            common_unroll_factors[:-1] + last_pos,
            times,
            # linestyle=":",
            linewidth=2,
            label="Vitis HLS",
            color="magenta",
            marker="x",
        )

        text = lat_ax_vitis.text(
            last_pos[0],
            times[-1],
            f"{times[-1]:02f} μs",
            ha="right",
            va="bottom",
            color="magenta",
            weight="bold",
            fontdict={"fontsize": 20},
        )
        text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="black")])
        lat_ax_vitis.plot(
            last_pos,
            [times[-1]],
            marker="X",
            markersize=7,
            color="magenta",
            markeredgecolor="k",
        )

        resource_ax_vitis.legend(
            loc="upper left", title="Resources", framealpha=0.8, fontsize=20
        ).set_zorder(102)
        lat_ax_vitis.legend(
            loc="upper right", title="Latency", framealpha=0.8, fontsize=20
        ).set_zorder(102)
        lat_legend = lat_ax_vitis.get_legend().legendHandles

        if mod in bragghls_df.index.levels[0]:
            red_patch = Line2D(
                [0],
                [0],
                color="w",
                label="BraggHLS",
                markerfacecolor="r",
                marker="X",
                markeredgecolor="k",
                markersize=15,
            )
            lat_ax_vitis.legend(
                handles=lat_legend + [red_patch],
                loc="upper right",
                title="Latency",
                framealpha=0.8,
                fontsize=20,
            )

        lat_ax_vitis.xaxis.set_ticks(common_unroll_factors[:-1] + last_pos)
        lat_ax_vitis.xaxis.set_ticklabels(["base"] + common_unroll_factors[1:])
        resource_ax_bragghls.set_xticks([2048])
        resource_ax_bragghls.set_xticklabels(["BraggHLS"])

        resource_ax_vitis.tick_params(left=False, right=False)
        resource_ax_vitis.yaxis.set_ticks([])
        resource_ax_vitis.yaxis.set_ticklabels([])

        resource_ax_bragghls.tick_params(left=False, right=False)
        resource_ax_bragghls.yaxis.set_label_position("right")
        resource_ax_bragghls.yaxis.tick_right()
        resource_ax_bragghls.set_xticks([2048])
        resource_ax_bragghls.set_xticklabels(["BraggHLS"])
        resource_ax_bragghls.tick_params(axis="x", rotation=45)

        lat_ax_vitis.tick_params(axis="x", rotation=45)
        for ax in [
            resource_ax_bragghls,
            lat_ax_vitis,
        ]:

            ax.yaxis.set_major_formatter(ticks_format)  # remove the major ticks
            ax.yaxis.set_minor_formatter(
                ticker.FuncFormatter(ticks_format)
            )  # add the custom ticks

        max_ax, max_y = max(
            [
                (resource_ax_vitis, resource_ax_vitis.get_ylim()[1]),
                (resource_ax_bragghls, resource_ax_bragghls.get_ylim()[1]),
                (lat_ax_vitis, lat_ax_vitis.get_ylim()[1]),
            ],
            key=lambda x: x[1],
        )
        min_ax, min_y = min(
            [
                (resource_ax_vitis, resource_ax_vitis.get_ylim()[0]),
                (resource_ax_bragghls, resource_ax_bragghls.get_ylim()[0]),
                (lat_ax_vitis, lat_ax_vitis.get_ylim()[0]),
            ],
            key=lambda x: x[1],
        )
        # lat_ax_vitis.set_ylim(min_y, max_y)
        resource_ax_vitis.set_ylim(min_y, max_y)
        resource_ax_bragghls.set_ylim(min_y, max_y)

        if mod in bragghls_df.index.levels[0]:
            bragghls_lat = (
                bragghls_df.loc[mod, 2048, "LatencyBest"].item()
                * bragghls_df.loc[mod, 2048, "clock_period_minus_wns"].item()
                / 1e3
            )
            text = resource_ax_bragghls.text(
                2048,
                min_y,
                f"{bragghls_lat:02f} μs",
                ha="center",
                va="bottom",
                color="red",
                weight="bold",
                fontdict={"fontsize": 20},
            )
            text.set_path_effects(
                [PathEffects.withStroke(linewidth=2, foreground="black")]
            )
            resource_ax_bragghls.plot(
                [2048],
                [min_y],
                marker="X",
                markersize=7,
                color="red",
                markeredgecolor="k",
            )

        # https://stackoverflow.com/a/59395256
        lat_ax_vitis.set_zorder(resource_ax_vitis.get_zorder() + 1)
        lat_ax_vitis.patch.set_visible(False)
        # https://github.com/matplotlib/matplotlib/issues/3706#issuecomment-817268918
        all_axes = fig.get_axes()
        for axis in all_axes:
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
                lat_ax_vitis.add_artist(legend)

        fig.tight_layout()
        fig.savefig(f"{mod}.pdf")
        # fig.show()

    # plt.rcParams.update({
    #     # "text.usetex": True,
    #     "font.family": "monospace",
    # })


def plot_elapsed_times():
    vitis_df = pd.read_csv("vitis_reports.csv", header=0, index_col=[0, 1, 2])
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.grid(which="major", linestyle="--")
    # ax1.minorticks_off()  # turns off minor ticks
    elapsed_time_common = set()

    for i, mod in enumerate(vitis_df.index.levels[0]):
        sub_df = vitis_df.loc[mod].swaplevel(0, 1)
        elapsed_time = sub_df.loc["elapsed_time"].sort_index()
        xs = np.array(elapsed_time.index.values)
        if xs[-1] == 1024:
            xs[-1] = xs[-2] + 4
        elapsed_time_common.update(set(xs))

        # mod = mod.replace("_", r"\_")
        ax1.plot(
            xs,
            elapsed_time.values,
            # label=rf"$\mathtt{{{mod}}}$",
            linewidth=2,
            marker="x",
            label=f"{mod}",
        )

    # ax1.set_xscale("log")
    ax1.set_ylabel("time (s)", fontdict={"fontsize": 25})
    ax1.set_yscale("log")
    xticks = sorted(elapsed_time_common)
    ax1.set_xticks(xticks)
    ax1.xaxis.set_ticklabels(["base"] + xticks[1:-1] + ["1024"])
    # ax1.xaxis.set_major_formatter(ticker.FuncFormatter(myticks))
    font = font_manager.FontProperties(family="Courier New", style="normal", size=20)
    ax1.legend(loc="upper left", framealpha=0.8, fontsize=20, prop=font)
    ax1.set_xlabel("unroll factor", fontdict={"fontsize": 25})
    ax1.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(f"elapsed_time.pdf")


def plot_conv_unroll_times():
    bragghls_df = pd.read_csv("unroll_conv.csv", header=0, index_col=[0])
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(
        bragghls_df.index.values,
        bragghls_df.values,
        linewidth=2,
        color="magenta",
        marker="x",
    )
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax1.grid(which="major", linestyle="--")
    # ax1.minorticks_off()  # turns off minor ticks
    # ax1.set_xscale("log")
    # ax1.set_yscale("log")
    ax1.set_ylabel("time (s)", fontdict={"fontsize": 25})
    ax1.set_xticks(sorted(bragghls_df.index.values))
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(myticks))
    font = font_manager.FontProperties(family="Courier New", style="normal", size=20)
    # ax1.legend(loc="upper left", framealpha=0.8, fontsize=20, prop=font)
    ax1.set_xlabel("image size", fontdict={"fontsize": 25})
    fig.tight_layout()
    fig.savefig("conv_unroll.pdf")


if __name__ == "__main__":
    # make_tables()
    plot_unrolls()
    plot_conv_unroll_times()
    plot_elapsed_times()
