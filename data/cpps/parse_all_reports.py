import glob
import json
import pathlib
import re
from collections import defaultdict
from functools import reduce
from os import remove
from pprint import pprint

import pandas as pd


def parse_design_analysis_routed(fp):
    report_lines = open(fp).readlines()
    for line in report_lines:
        if line.startswith("| Requirement"):
            _, _, req, _ = line.split("|", 3)
            req = float(req.strip())
        if line.startswith("| Slack"):
            _, _, slack, _ = line.split("|", 3)
            slack = float(slack.strip())
            break

    return req, slack


# def parse_utilization_routed(fp):
#     report_lines = open(fp).readlines()
#     primitives = {}
#     for line in report_lines:
#         for primitive in [
#             "FDRE",
#             "LUT3",
#             "LUT6",
#             "LUT4",
#             "SRL16E",
#             "LUT5",
#             "LUT2",
#             "LUT1",
#             "RAMB18E2",
#             "FDSE",
#             "DSP48E2",
#             "MUXF7",
#         ]:
#             if line.startswith(f"| {primitive}"):
#                 _, _, used, _ = line.split("|", 3)
#                 used = int(used.strip())
#                 primitives[primitive] = used
#
#                 if primitive == "MUXF7":
#                     break
#
#     return primitives


def parse_utilization_routed(fp):
    report_lines = open(fp).readlines()
    resources = {}
    resource_lines = {
        "| CLB LUTs": "LUTs",
        "| CLB Registers": "Registers",
        "|   RAMB18": "RAMs",
        "| DSPs": "DSPs",
    }

    for line in report_lines:
        for resource_starts, resource_name in resource_lines.items():
            if resource_name in resources:
                continue
            if line.startswith(resource_starts):
                util = float(line.strip().split("|")[-2].strip())
                if util == 0.0:
                    print()
                resources[resource_name] = util

                if resource_name == "DSPs":
                    break

    return dict(sorted(resources.items()))


modules = list(
    map(
        lambda x: x.strip(),
        """
addmm
batch_norm
braggnn
conv
matmul
max_pool_2d
soft_max
""".split(),
    )
)


# forward_design_analysis_routed.rpt
# forward_utilization_routed.rpt
# solution1_data.json


# /home/mlevental/dev_projects/hls_paper/data/cpps/addmm/addmm_unroll_2/solution1/impl/verilog/report
name = re.compile(r"/(.*)_unroll_(.*)/solution1/")

all_reports = defaultdict(lambda: defaultdict(dict))

for mod in modules:
    for fp in glob.glob(
        f"{mod}/**/solution1/impl/verilog/report/*.rpt", recursive=True
    ):
        _mod_name, unroll_factor = name.findall(fp)[0]
        unroll_factor = int(unroll_factor)
        if fp.endswith("forward_design_analysis_routed.rpt"):
            req, slack = parse_design_analysis_routed(fp)
            all_reports[mod][unroll_factor]["target_clock_period"] = req
            all_reports[mod][unroll_factor]["slack"] = slack
            all_reports[mod][unroll_factor]["clock_period_minus_wns"] = req - slack
        elif fp.endswith("forward_utilization_routed.rpt"):
            resources = parse_utilization_routed(fp)
            # _primitives = dict(primitives)
            # luts = reduce(lambda acc, k: acc + (_primitives.pop(k) if "LUT" in k else 0), primitives.keys(), 0)
            # _primitives["LUT"] = luts
            # # FDRE Primitive: D Flip-Flop with Clock Enable and Synchronous Reset
            # # FDSE Primitive: D Flip-Flop with Clock Enable and Synchronous Set
            # dfs = reduce(lambda acc, k: acc + (_primitives.pop(k) if "FD" in k else 0), primitives.keys(), 0)
            # _primitives["FD"] = dfs
            all_reports[mod][unroll_factor].update(resources)

    all_reports[mod] = dict(sorted(all_reports[mod].items()))


for mod in modules:
    for fp in glob.glob(f"{mod}/**/*.json", recursive=True):
        _mod_name, unroll_factor = name.findall(fp)[0]
        unroll_factor = int(unroll_factor)
        sol_json = json.load(open(fp))
        lat = sol_json["ModuleInfo"]["Metrics"]["forward"]["Latency"]
        for k, v in lat.items():
            try:
                v = int(v)
                lat[k] = v
            except:
                continue
        lat.pop("PipelineDepth")
        lat.pop("PipelineType")
        if unroll_factor in all_reports[mod]:
            all_reports[mod][unroll_factor].update(lat)

    all_reports[mod] = dict(sorted(all_reports[mod].items()))

# INFO: [HLS 200-111] Finished Command csynth_design CPU user time: 1822.87 seconds. CPU system time: 8.56 seconds. Elapsed time: 1860.74 seconds; current allocated memory: 2.120 GB.
csynth_time = re.compile(
    r"INFO: \[HLS 200-111\] Finished Command csynth_design CPU user time: (.*) seconds. CPU system time: (.*) seconds. Elapsed time: (.*) seconds; current allocated memory: (.*)"
)

for mod in modules:
    for fp in pathlib.Path(".").glob(f'{mod}/**/autopilot.flow.log'):
        fp = str(fp.resolve())
        _mod_name, unroll_factor = name.findall(fp)[0]
        unroll_factor = int(unroll_factor)
        log = open(fp).readlines()
        try:
            csynth_time_line = next(line for line in log if "Finished Command csynth_design" in line)
            user_time, cpu_time, elapsed_time, memory = csynth_time.findall(csynth_time_line)[0]
            elapsed_time = float(elapsed_time)
            if "elapsed_time" in all_reports[mod][unroll_factor]:
                elapsed_time = max(elapsed_time, all_reports[mod][unroll_factor]["elapsed_time"])
            print(mod, unroll_factor, elapsed_time)
            all_reports[mod][unroll_factor]["elapsed_time"] = elapsed_time
        except:
            pass


json.dump(all_reports, open("all_reports.json", "w"), indent=2)

csv = open("all_reports.csv", "w")
csv.write("module,unroll_factor,metric_name,metric_val\n")
for mod in all_reports:
    for unroll_factor in all_reports[mod]:
        for metric_name in all_reports[mod][unroll_factor]:
            metric_val = all_reports[mod][unroll_factor][metric_name]
            csv.write(f"{mod},{unroll_factor},{metric_name},{metric_val}\n")
