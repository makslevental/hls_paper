import glob
import json
import os
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


from io import StringIO


def parse_utilization_routed(fp):
    report_lines = open(fp).readlines()
    for i, line in enumerate(report_lines):
        if ". Primitives" in line and "-------------" in report_lines[i + 1]:
            for j in range(100):
                next_line = report_lines[i + j]
                if ". Black Boxes" in next_line and "---------------":
                    break
            test = StringIO(
                "\n".join(report_lines[i + 2 : i + j])
                .replace("+", "")
                .replace("-", "")
                .replace("|", ",")
                .strip()
            )
            df = pd.read_csv(test, sep="\s*,\s*", engine="python")
            df = df[["Ref Name", "Used", "Functional Category"]]
            df = df.groupby(by=["Functional Category"]).sum(numeric_only=True)

            d = dict(df.to_dict()["Used"].items())
            res = {}
            if "Arithmetic" in d:
                res["DSP"] = d.pop("Arithmetic")
            if "CLB" in d:
                res["LUT"] = d.pop("CLB")
            if "Register" in d:
                res["FF"] = d.pop("Register")
            return res


modules = list(
    map(
        lambda x: x.strip(),
        """
addmm
batch_norm
braggnn
conv
max_pool_2d
soft_max
""".split(),
    )
)

name = re.compile(r"/.*?/(\d+)/")


# forward_design_analysis_routed.rpt
# forward_utilization_routed.rpt
# solution1_data.json


def parse_using_standard_reports():
    all_reports = defaultdict(lambda: defaultdict(dict))

    for mod in modules:
        for fp in sorted(
            glob.glob(
                f"{mod}/**/solution1/impl/verilog/report/forward_design_analysis_synth.rpt",
                recursive=True,
            )
        ):
            _mod_name, unroll_factor = name.findall(fp)[0]
            unroll_factor = int(unroll_factor)
            if unroll_factor <= 1024:
                req, slack = parse_design_analysis_routed(fp)
                all_reports[mod][unroll_factor]["target_clock_period"] = req
                all_reports[mod][unroll_factor]["slack"] = slack
                all_reports[mod][unroll_factor]["clock_period_minus_wns"] = req - slack

    for mod in modules:
        for fp in sorted(
            glob.glob(
                f"{mod}/**/solution1/impl/verilog/report/forward_utilization_synth.rpt",
                recursive=True,
            )
        ):
            _mod_name, unroll_factor = name.findall(fp)[0]
            unroll_factor = int(unroll_factor)
            if unroll_factor == 1:
                unroll_factor = 0
            if unroll_factor <= 1024:
                resources = parse_utilization_routed(fp)
                if resources:
                    all_reports[mod][unroll_factor].update(resources)

        all_reports[mod] = dict(sorted(all_reports[mod].items()))

    return all_reports


def parse_json(all_reports):
    for mod in modules:
        for fp in sorted(
            glob.glob(f"reports/{mod}/**/*.json", recursive=True)
        ):
            unroll_factor = name.findall(fp)[0]
            unroll_factor = int(unroll_factor)
            if unroll_factor == 1:
                unroll_factor = 0
            if unroll_factor <= 1024:
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
    all_reports["avails"] = {
        k.replace("AVAIL_", ""): int(v)
        for k, v in sol_json["ModuleInfo"]["Metrics"]["forward"]["Area"].items()
        if "AVAIL" in k
    }
    return all_reports


def parse_using_export_rpts():
    all_reports = defaultdict(lambda: defaultdict(dict))

    for mod in modules:
        for fp in sorted(
            glob.glob(
                f"reports/{mod}/**/forward_export.rpt",
                recursive=True,
            )
        ):
            unroll_factor = name.findall(fp)[0]
            unroll_factor = int(unroll_factor)
            if unroll_factor == 1:
                unroll_factor = 0

            with open(fp) as f:
                lines = f.readlines()
                for line in lines:
                    splits = line.split(":")
                    if line.startswith("LUT:"):
                        all_reports[mod][unroll_factor]["LUT"] = int(splits[1].strip())
                    elif line.startswith("FF:"):
                        all_reports[mod][unroll_factor]["FF"] = int(splits[1].strip())
                    elif line.startswith("DSP:"):
                        all_reports[mod][unroll_factor]["DSP"] = int(splits[1].strip())
                    elif line.startswith("BRAM:"):
                        all_reports[mod][unroll_factor]["BRAM"] = int(splits[1].strip())
                    elif line.startswith("CP achieved post-synthesis:"):
                        all_reports[mod][unroll_factor][
                            "clock_period_minus_wns"
                        ] = float(splits[1].strip())
        all_reports[mod] = dict(sorted(all_reports[mod].items()))

    return all_reports


def parse_autopilot_logs(all_reports):
    # INFO: [HLS 200-111] Finished Command csynth_design CPU user time: 1822.87 seconds. CPU system time: 8.56 seconds. Elapsed time: 1860.74 seconds; current allocated memory: 2.120 GB.
    csynth_time = re.compile(
        r"INFO: \[HLS 200-111\] Finished Command csynth_design CPU user time: (.*) seconds. CPU system time: (.*) seconds. Elapsed time: (.*) seconds; current allocated memory: (.*)"
    )
    for mod in modules:
        for fp in sorted(
            pathlib.Path(".").glob(
                f"reports/{mod}/**/autopilot.flow.log"
            )
        ):
            fp = str(fp.resolve())
            unroll_factor = name.findall(fp)[0]
            unroll_factor = int(unroll_factor)
            if unroll_factor == 1:
                unroll_factor = 0
            log = open(fp).readlines()
            try:
                csynth_time_line = next(
                    line for line in log if "Finished Command csynth_design" in line
                )
                user_time, cpu_time, elapsed_time, memory = csynth_time.findall(
                    csynth_time_line
                )[0]
                elapsed_time = float(elapsed_time)
                if "elapsed_time" in all_reports[mod][unroll_factor]:
                    elapsed_time = max(
                        elapsed_time, all_reports[mod][unroll_factor]["elapsed_time"]
                    )
                all_reports[mod][unroll_factor]["elapsed_time"] = elapsed_time
            except:
                pass

    return all_reports


def bragghls():
    wns_re = re.compile(
        r"""\s* WNS\(ns\).*
\s* -------.*
\s* ([-]?\d+[.]\d+).*
""",
        flags=re.MULTILINE,
    )

    all_reports = {}
    sched_re = re.compile(r"return \{lpStartTime = (\d+) : i32}")
    for mod in modules:
        all_reports[mod] = {}
        fp = f"bragghls_artifacts/{mod}_16/reports/post_synth/utilization.rpt"
        if os.path.exists(fp):
            util = parse_utilization_routed(fp)
            all_reports[mod].update(util)

        fp = f"bragghls_artifacts/{mod}_16/reports/post_synth/timing_summary.rpt"
        if os.path.exists(fp):
            with open(fp) as f:
                wns = float(wns_re.findall(f.read())[0])

            fp = f"bragghls_artifacts/{mod}_16/{mod}.sched.mlir"
            if os.path.exists(fp):
                with open(fp) as f:
                    sched = int(sched_re.findall(f.read())[0])
                    all_reports[mod]["LatencyBest"] = sched
                    all_reports[mod]["clock_period_minus_wns"] = (10 - wns)

        all_reports[mod] = dict(sorted(all_reports[mod].items()))
    all_reports = dict(sorted(all_reports.items()))
    return all_reports


if __name__ == "__main__":
    # all_reports = parse_using_standard_reports()
    vitis_reports = parse_using_export_rpts()
    vitis_reports = parse_json(vitis_reports)
    avails = vitis_reports.pop("avails")
    for mod in vitis_reports:
        for unroll_factor in vitis_reports[mod]:
            for metric_name in vitis_reports[mod][unroll_factor]:
                if metric_name in avails:
                    vitis_reports[mod][unroll_factor][metric_name] /= avails[metric_name]
                    vitis_reports[mod][unroll_factor][metric_name] *= 100
    vitis_reports = parse_autopilot_logs(vitis_reports)

    bragghls_reports = bragghls()
    for mod in bragghls_reports:
        for metric_name in bragghls_reports[mod]:
            if metric_name in avails:
                bragghls_reports[mod][metric_name] /= avails[metric_name]
                bragghls_reports[mod][metric_name] *= 100

    json.dump(
        {"vitis": vitis_reports, "bragghls": bragghls_reports},
        open("all_reports.json", "w"),
        indent=2,
    )

    csv = open("vitis_reports.csv", "w")
    csv.write("module,unroll_factor,metric_name,metric_val\n")
    for mod in vitis_reports:
        if mod == "avails":
            continue
        for unroll_factor in vitis_reports[mod]:
            for metric_name in vitis_reports[mod][unroll_factor]:
                metric_val = vitis_reports[mod][unroll_factor][metric_name]
                print(f"{mod},{unroll_factor},{metric_name},{metric_val}\n")
                csv.write(f"{mod},{unroll_factor},{metric_name},{metric_val}\n")
                csv.flush()
    
    csv = open("bragghls_reports.csv", "w")
    csv.write("module,unroll_factor,metric_name,metric_val\n")
    for mod in bragghls_reports:
        for metric_name in bragghls_reports[mod]:
            metric_val = bragghls_reports[mod][metric_name]
            print(f"{mod},{metric_name},{metric_val}\n")
            csv.write(f"{mod},2048,{metric_name},{metric_val}\n")
            csv.flush()

