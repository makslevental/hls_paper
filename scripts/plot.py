import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import ticker, font_manager
import matplotlib.patheffects as PathEffects
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D

vitis_df = pd.read_csv("vitis_reports.csv", header=0, index_col=[0, 1, 2])
bragghls_df = pd.read_csv("bragghls_reports.csv", header=0, index_col=[0, 1, 2])

font = {
    # "family": "normal",
    "size": 18
}

matplotlib.rc("font", **font)


# matplotlib.rcParams['text.usetex'] = True


def myticks(x, pos):
    if x <= 0:
        return f"${x}$"
    exponent = int(np.log2(x))
    coeff = x / 2 ** exponent
    # return r"$2^{{ {:2d} }}$".format(exponent)
    return str(x)


MAX_REGULAR = None
BAR_CHART_COLORS = {}
PATTERNS = {}


def plot_metrics(sub_df, ax1, spacing=4, scale=10):
    i = 0
    patterns = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    for metric in ["DSP", "BRAM", "LUT", "FF"]:
        try:
            vals = sub_df.loc[metric].sort_index()
        except:
            continue
        xs = np.array(vals.index.values)
        bars = [i for i in ax1.containers if isinstance(i, BarContainer)]

        global MAX_REGULAR
        if xs[-1] == 1024:
            MAX_REGULAR = xs[-2]
            xs[-1] = MAX_REGULAR + spacing
        if xs[-1] == 2048:
            if MAX_REGULAR is None:
                MAX_REGULAR = bars[1].patches[-1].xy[0] + 2 * spacing
            xs[-1] = MAX_REGULAR + 2 * spacing

        vals = vals.values.flatten()
        if np.all(vals == 0):
            continue
        diffs = np.diff(xs) / scale
        if not len(diffs):
            diffs = [spacing / scale]
        diffs = np.append(diffs, [diffs[-1] * 1])
        if metric in BAR_CHART_COLORS:
            r = ax1.bar(
                xs + diffs * (-1 + i),
                vals,
                width=diffs,
                log=False,
                ec="k",
                color=BAR_CHART_COLORS[metric],
                hatch=PATTERNS[metric],
            )
        else:
            PATTERNS[metric] = patterns[i + 4]
            r = ax1.bar(
                xs + diffs * (-1 + i),
                vals,
                width=diffs,
                log=False,
                label=metric,
                ec="k",
                hatch=PATTERNS[metric],
            )
            BAR_CHART_COLORS[metric] = r.patches[0].get_facecolor()
        i += 1
        # ax1.plot(vals.index.values, vals.values, label=metric)


def plot_unrolls():
    for i, mod in enumerate(vitis_df.index.levels[0]):
        global MAX_REGULAR, BAR_CHART_COLORS, PATTERNS
        MAX_REGULAR = None
        BAR_CHART_COLORS = {}
        PATTERNS = {}
        fig, ax1 = plt.subplots(figsize=(10, 7))

        sub_df = vitis_df.loc[mod].swaplevel(0, 1)
        plot_metrics(sub_df, ax1)
        if mod in bragghls_df.index.levels[0]:
            bragghls_sub_df = bragghls_df.loc[mod].swaplevel(0, 1)
            plot_metrics(bragghls_sub_df, ax1)

        # ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.grid(which="major", axis="x", linestyle="--")
        ax1.minorticks_off()  # turns off minor ticks

        ax2 = ax1.twinx()

        # if i % 2 == 0:
        #     ax1.set_ylabel("utilization (%)", fontdict={"fontsize": 25})
        # else:
        #     ax2.set_ylabel("time (μs)", fontdict={"fontsize": 25})
        ax1.set_ylabel("utilization (%)", fontdict={"fontsize": 25})
        ax2.set_ylabel("time (μs)", fontdict={"fontsize": 25})

        ax1.set_xlabel("unroll factor", fontdict={"fontsize": 25})

        try:
            clock_vals = sub_df.loc["clock_period_minus_wns"].sort_index()
            latency_vals = sub_df.loc["LatencyBest"].sort_index()
            common_unroll_factors = sorted(
                set(latency_vals.index.values) & set(clock_vals.index.values)
            )
            times = (
                clock_vals.loc[common_unroll_factors].values
                * latency_vals.loc[common_unroll_factors].values
                / 1e3
            )
            times = times.flatten()

            if common_unroll_factors[-1] == 1024:
                common_unroll_factors[-1] = common_unroll_factors[-2] + 4

            ax2.plot(
                common_unroll_factors,
                times,
                # linestyle=":",
                linewidth=2,
                label="Vitis HLS",
                color="magenta",
                marker="x",
            )
        except:
            pass
        
        vitis_lat = times[-1]
        vitis_va = "top"
        if mod in bragghls_df.index.levels[0]:
            bragghls_lat = bragghls_df.loc[mod, 2048, "LatencyBest"].item() * bragghls_df.loc[mod, 2048, "clock_period_minus_wns"].item() / 1e3
            if vitis_lat < bragghls_lat:
                vitis_va = "top"
                bragghls_va = "bottom"
            else:
                vitis_va = "bottom"
                bragghls_va = "top"

        text = ax2.text(
            common_unroll_factors[-1],
            times[-1],
            f"{times[-1]:02f} μs",
            ha="right",
            va=vitis_va,
            color="magenta",
            weight="bold",
        )
        text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="black")])
        ax2.plot(
            [common_unroll_factors[-1]],
            [times[-1]],
            marker="X",
            markersize=7,
            color="magenta",
            markeredgecolor="k"
        )

        if mod in bragghls_df.index.levels[0]:
            common_unroll_factors.append(48)
            bragghls_lat = bragghls_df.loc[mod, 2048, "LatencyBest"].item() * bragghls_df.loc[mod, 2048, "clock_period_minus_wns"].item() / 1e3
            text = ax2.text(
                MAX_REGULAR + 2 * 4,
                bragghls_lat,
                f"{bragghls_lat:02f} μs",
                ha="right",
                va=bragghls_va,
                color="red",
                weight="bold",
            )
            text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="black")])
            ax2.plot(
                [MAX_REGULAR + 2 * 4],
                [bragghls_lat],
                marker="X",
                markersize=7,
                color="red",
                markeredgecolor="k"
            )

        # ax2.plot([0, 2048], [0.221, 0.221], marker='o', linestyle="--", label="BraggHLS", linewidth=2, color="red")
        # text = ax2.text(
        #     common_unroll_factors[-1], times[-1] + 1, f"{0.221} μs", ha="center", va="bottom", color="red", weight="bold"
        # )
        # text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
        # # ax2.plot([common_unroll_factors[-1]], [0.221], marker='X', markersize=5, color="red")
        # ax2.set_xlim([0, 1500])

        ax2.grid(which="both", axis="y", linestyle="--")
        # ax2.minorticks_off()  # turns off minor ticks

        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(myticks))
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(myticks))

        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax1.set_xticks(common_unroll_factors)
        labels = [item.get_text() for item in ax1.get_xticklabels()]
        if common_unroll_factors[-1] == 48:
            labels[-1] = "full"
        if common_unroll_factors[-3] == 40:
            labels[-2] = "1024"
        ax1.set_xticklabels(labels)
        ax2.set_xticklabels(labels)
        ax1.tick_params(axis="x", rotation=45)

        # if i == 1:
        #     ax2.legend(loc="upper right", framealpha=1, fontsize=20).set_zorder(102)
        # if i == 0:
        #     ax1.legend(loc="upper left", title="Resources", framealpha=1, fontsize=20).set_zorder(102)
        ax1.legend(
            loc="upper left", title="Resources", framealpha=0.8, fontsize=20
        ).set_zorder(102)
        ax2.legend(
            loc="upper right", title="Latency", framealpha=0.8, fontsize=20
        ).set_zorder(102)
        lat_legend = ax2.get_legend().legendHandles

        if mod in bragghls_df.index.levels[0]:
            red_patch = Line2D([0], [0], color='w', label='BraggHLS',
                              markerfacecolor='r', marker="X", markeredgecolor="k",
                    markersize=15)
            ax2.legend(handles=lat_legend + [red_patch], loc="upper right", title="Latency", framealpha=0.8, fontsize=20)


        # https://github.com/matplotlib/matplotlib/issues/3706#issuecomment-817268918
        all_axes = fig.get_axes()
        for axis in all_axes:
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
                all_axes[-1].add_artist(legend)

        fig.tight_layout()
        fig.savefig(f"{mod}.pdf")
        # fig.show()

    # plt.rcParams.update({
    #     # "text.usetex": True,
    #     "font.family": "monospace",
    # })

    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.grid(which="major", linestyle="--")
    ax1.minorticks_off()  # turns off minor ticks
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
            label=f"{mod}",
        )

    # ax1.set_xscale("log")
    ax1.set_ylabel("time (s)", fontdict={"fontsize": 25})
    ax1.set_yscale("log")
    ax1.set_xticks(sorted(elapsed_time_common))
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(myticks))
    font = font_manager.FontProperties(family="Courier New", style="normal", size=20)
    ax1.legend(loc="upper left", framealpha=0.8, fontsize=20, prop=font)
    ax1.set_xlabel("unroll factor", fontdict={"fontsize": 25})

    fig.tight_layout()
    fig.savefig(f"elapsed_time.pdf")


# plot_unrolls()

def plot_unroll_times():
    bragghls_df = pd.read_csv("unroll_conv.csv", header=0, index_col=[0])
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(bragghls_df.index.values, bragghls_df.values)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax1.grid(which="major", linestyle="--")
    ax1.minorticks_off()  # turns off minor ticks
    # ax1.set_xscale("log")
    # ax1.set_yscale("log")
    ax1.set_ylabel("time (s)", fontdict={"fontsize": 25})
    ax1.set_xticks(sorted(bragghls_df.index.values))
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(myticks))
    font = font_manager.FontProperties(family="Courier New", style="normal", size=20)
    # ax1.legend(loc="upper left", framealpha=0.8, fontsize=20, prop=font)
    ax1.set_xlabel("unroll factor", fontdict={"fontsize": 25})
    fig.tight_layout()
    fig.savefig("conv_unroll.pdf")

plot_unroll_times()

