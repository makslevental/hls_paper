import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import ticker, font_manager
import matplotlib.patheffects as PathEffects

df = pd.read_csv("all_reports.csv", header=0, index_col=[0, 1, 2])

font = {
    # "family": "normal",
    "size": 18}

matplotlib.rc("font", **font)
# matplotlib.rcParams['text.usetex'] = True


def myticks(x, pos):
    if x <= 0:
        return f"${x}$"
    exponent = int(np.log2(x))
    coeff = x / 2 ** exponent
    # return r"$2^{{ {:2d} }}$".format(exponent)
    return str(x)


for i, mod in enumerate(df.index.levels[0]):
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10), sharex=True)
    fig, ax1 = plt.subplots(figsize=(10, 7))
    # fig.suptitle(mod, fontsize=16)

    patterns = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    bar_chart_colors = {}
    sub_df = df.loc[mod].swaplevel(0, 1)
    metrics = sub_df.index.levels[0]
    for i, metric in enumerate(["Arithmetic", "BLOCKRAM", "CLB", "Register"]):
        try:
            vals = sub_df.loc[metric].sort_index()
        except:
            continue
        xs = vals.index.values
        diffs = np.diff(xs)/ 10
        diffs = np.append(diffs, [diffs[-1] * 2])
        r = ax1.bar(xs + diffs * (-1 + i), vals.values.flatten(), width=diffs, log=False, label=metric, ec="k", hatch=patterns[i+4])
        bar_chart_colors[metric] = r.patches[0].get_facecolor()
        # ax1.plot(vals.index.values, vals.values, label=metric)

    ax1.set_xscale("log")
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
        times = clock_vals.loc[common_unroll_factors].values * latency_vals.loc[common_unroll_factors].values / 1e3
        times = times.flatten()
        ax2.plot(
            common_unroll_factors,
            times,
            # linestyle=":",
            linewidth=2,
            label="Vitis HLS",
            color="magenta",
            marker="x"
        )
    except:
        pass

    text = ax2.text(
        common_unroll_factors[-1], times[-1] + 1, f"{times[-1]:02f} μs", ha="center", va="bottom", color="magenta", weight="bold"
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
    ax2.plot([common_unroll_factors[-1]], [times[-1]], marker='X', markersize=5, color="magenta")

    # ax2.plot([0, 2048], [0.221, 0.221], marker='o', linestyle="--", label="BraggHLS", linewidth=2, color="red")
    # text = ax2.text(
    #     common_unroll_factors[-1], times[-1] + 1, f"{0.221} μs", ha="center", va="bottom", color="red", weight="bold"
    # )
    # text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
    # # ax2.plot([common_unroll_factors[-1]], [0.221], marker='X', markersize=5, color="red")
    # ax2.set_xlim([0, 1500])

    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # ax2.set_xticks(latency_vals.index.values)
    ax1.set_xticks(latency_vals.index.values)
    ax1.tick_params(axis='x', rotation=45)

    ax2.grid(which="both", axis="y", linestyle="--")
    # ax2.minorticks_off()  # turns off minor ticks

    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(myticks))
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(myticks))

    # if i == 1:
    #     ax2.legend(loc="upper right", framealpha=1, fontsize=20).set_zorder(102)
    # if i == 0:
    #     ax1.legend(loc="upper left", title="Resources", framealpha=1, fontsize=20).set_zorder(102)
    ax1.legend(loc="upper left", title="Resources", framealpha=1, fontsize=20).set_zorder(102)
    ax2.legend(loc="upper right",title="Latency", framealpha=1, fontsize=20).set_zorder(102)

    #https://github.com/matplotlib/matplotlib/issues/3706#issuecomment-817268918
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

for i, mod in enumerate(df.index.levels[0]):
    sub_df = df.loc[mod].swaplevel(0, 1)
    elapsed_time = sub_df.loc["elapsed_time"].sort_index()
    elapsed_time_common.update(set(elapsed_time.index.values))

    # mod = mod.replace("_", r"\_")
    ax1.plot(
        elapsed_time.index.values,
        elapsed_time.values,
        # label=rf"$\mathtt{{{mod}}}$",
        label=f"{mod}",
    )

ax1.set_xscale("log")
ax1.set_ylabel("time (s)", fontdict={"fontsize": 25})
ax1.set_yscale("log")
ax1.set_xticks(sorted(elapsed_time_common))
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(myticks))
font = font_manager.FontProperties(family='Courier New',
                                   style='normal', size=20)
ax1.legend(loc="upper left", framealpha=1, fontsize=20, prop=font)

fig.tight_layout()
fig.savefig(f"elapsed_time.pdf")