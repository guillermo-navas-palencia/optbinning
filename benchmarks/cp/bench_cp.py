"""Compare two clean checkouts with the same interpreter and benchmark code."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, default=lambda x: x.tolist()).encode()).hexdigest()


def save_report(path, report):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def worker(checkout):
    # Import exactly the requested checkout; imports are outside timings.
    sys.path.insert(0, str(checkout))
    import numpy as np
    import optbinning
    from sklearn.datasets import load_breast_cancer, load_diabetes
    from optbinning import OptimalBinning, ContinuousOptimalBinning
    from cases import canonical_hash, construction_cases

    if Path(optbinning.__file__).resolve().parent != checkout / "optbinning":
        raise RuntimeError("Imported a different OptBinning checkout")
    rows = []
    key, warmup, args = next(construction_cases())
    warmup.build_model(*args)
    for key, model, args in construction_cases():
        input_hash = digest(args)
        start = time.perf_counter()
        model.build_model(*args)
        seconds = time.perf_counter() - start
        rows.append(dict(case=key, stage="construction", seconds=seconds,
                         input_hash=input_hash,
                         model_hash=canonical_hash(model._model)))

    for target, loader, estimator in [
            ("binary", load_breast_cancer, OptimalBinning),
            ("continuous", load_diabetes, ContinuousOptimalBinning)]:
        data = loader()
        for dtype in ["numerical", "categorical"]:
            x = data.data[:, 0].copy()
            if dtype == "categorical":
                cuts = np.unique(np.quantile(x, np.linspace(0, 1, 13)[1:-1]))
                x = np.array([f"c{i}" for i in np.digitize(x, cuts)])
            for trend in [None, "auto"]:
                settings = dict(dtype=dtype, monotonic_trend=trend,
                                max_n_prebins=20, max_n_bins=8, time_limit=5)
                # Separate first-use warm-up for every public-fit case.
                estimator(**settings).fit(x, data.target)
                model = estimator(**settings)
                start = time.perf_counter()
                model.fit(x, data.target)
                seconds = time.perf_counter() - start
                metric = "event_rate" if target == "binary" else "mean"
                output_hash = None
                if model.status in ("OPTIMAL", "FEASIBLE"):
                    output_hash = digest(model.transform(x, metric=metric))
                rows.append(dict(case=[target, dtype, trend], stage="fit",
                                 seconds=seconds, status=model.status,
                                 input_hash=digest([x, data.target, settings]),
                                 output_hash=output_hash))
    versions = {name: importlib.metadata.version(name) for name in
                ["numpy", "scipy", "ortools", "scikit-learn", "pandas",
                 "matplotlib", "ropwr", "cvxpy"]}
    return dict(rows=rows, versions=versions, python=sys.version,
                platform=platform.platform())


def git(checkout, *args):
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args], text=True).strip()


def summarize(pairs, require_equivalent):
    """Keep all outcomes; only summarize compatible timings as speedups."""
    groups = {}
    for pair in pairs:
        reference, candidate = pair["reference"], pair["candidate"]
        if reference["versions"] != candidate["versions"]:
            raise ValueError("Different dependency versions")
        a = {(r["stage"], json.dumps(r["case"])): r
             for r in reference["rows"]}
        b = {(r["stage"], json.dumps(r["case"])): r
             for r in candidate["rows"]}
        if len(a) != len(reference["rows"]) or len(b) != len(candidate["rows"]):
            raise ValueError("Duplicate cases")
        if a.keys() != b.keys():
            raise ValueError("Incomplete paired coverage")
        for key, left in a.items():
            right = b[key]
            if left["input_hash"] != right["input_hash"]:
                raise ValueError("Different inputs")
            if key[0] == "construction":
                equal = left["model_hash"] == right["model_hash"]
            else:
                equal = (left["status"] == right["status"] and
                         left["output_hash"] == right["output_hash"])
            if require_equivalent and not equal:
                raise ValueError(f"Changed model or fit outcome: {key}")
            record = groups.setdefault(key, dict(
                stage=key[0], case=left["case"], equivalent=True,
                reference_seconds=[], candidate_seconds=[], speedups=[]))
            record["equivalent"] &= equal
            record["reference_seconds"].append(left["seconds"])
            record["candidate_seconds"].append(right["seconds"])
            # Time-limited/incumbent-only fits are not proven equal work.
            eligible = equal and (key[0] == "construction" or
                                 left["status"] == "OPTIMAL")
            if eligible:
                record["speedups"].append(left["seconds"] / right["seconds"])
    for record in groups.values():
        speeds = record.pop("speedups")
        record["eligible_pairs"] = len(speeds)
        record["speedup_median"] = statistics.median(speeds) if speeds else None
        record["speedup_range"] = [min(speeds), max(speeds)] if speeds else None
    return list(groups.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-equivalent", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        # Dependencies may print import diagnostics; stdout is the JSON protocol.
        with contextlib.redirect_stdout(sys.stderr):
            result = worker(args.worker.resolve())
        print(json.dumps(result))
        return
    if not args.reference or not args.candidate or not args.output:
        parser.error("--reference, --candidate and --output are required")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    checkouts = {name: getattr(args, name).resolve()
                 for name in ["reference", "candidate"]}
    revisions = {}
    for name, path in checkouts.items():
        if git(path, "status", "--porcelain", "--untracked-files=normal"):
            parser.error(f"Use a clean checkout: {path}")
        revisions[name] = git(path, "rev-parse", "HEAD")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", MPLBACKEND="Agg", PYTHONHASHSEED="0")
    rng = random.Random(args.seed)
    script = Path(__file__).resolve()
    report = dict(revisions=revisions, repeats=args.repeats, seed=args.seed,
                  benchmark_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [script, script.with_name("cases.py")]},
                  pairs=[])
    # Preserve completed workers even if a later worker fails; never call them
    # a complete comparison. Subprocess startup is excluded from measured time.
    report["complete"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_report(args.output, report)
    for repeat in range(args.repeats):
        order = list(checkouts)
        rng.shuffle(order)
        pair = dict(repeat=repeat, order=order)
        report["pairs"].append(pair)
        for name in order:
            completed = subprocess.run(
                [sys.executable, str(script), "--worker", str(checkouts[name])],
                env=env, cwd=checkouts[name], capture_output=True, text=True,
                check=True)
            pair[name] = json.loads(completed.stdout)
            pair[name]["stderr"] = completed.stderr
            save_report(args.output, report)
        print(f"Completed pair {repeat + 1}/{args.repeats}", flush=True)
    report["summary"] = summarize(report["pairs"], args.require_equivalent)
    report["complete"] = True
    save_report(args.output, report)


if __name__ == "__main__":
    main()
