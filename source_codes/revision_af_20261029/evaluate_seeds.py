"""当前 A/F 的训练种子评价；可核验复用旧记录的 A 两列，不加载联合校正。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import traceback
from unittest.mock import patch
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from study_full_baselines.common import SOURCE, require_host, sha256, write_json, now, atomic_npz, rng_for, verify_sources
from study_full_baselines.online_controller import OnlineController
from study_full_baselines.runtime_bundle import verify as verify_bundle
from study_full_baselines.confirmation_batch import guard_load
from study_full_baselines.confirmation_receiver import public_observation, score_decisions
from our_method_response_control.train import precision
from study_uniform64.controllers import Feedback64, audit as audit_trace
from study_full_baselines.paired_statistics import sufficient, metrics, FIELDS

NAMES = ["A_direct16", "A_feedback64"]
STATE = {}


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def original(row, fc):
    p = STATE["protocol"]
    source = Path(p["source"])
    folder = source / "records" / f"environment_{row['index']:05d}"
    marker = read(folder / "complete.json")
    assert marker["environment_id"] == row["environment_id"]
    assert sha256(folder / "environment.json") == marker["environment_sha256"]
    item = next(x for x in marker["carriers"] if x["carrier_ghz"] == fc)
    path = source / item["path"]
    assert sha256(path) == item["sha256"]
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k].copy() for k in f.files}
    return read(folder / "environment.json"), arrays, item["sha256"]


def audit(arrays, carrier):
    np.testing.assert_array_equal(arrays["metrics"][:, 10], [16, 64])
    np.testing.assert_array_equal(arrays["A_feedback64__proposal_control_code"], arrays["control_code"][0])
    subset = {k: arrays[k] for k in arrays if k.startswith("A_feedback64__")}
    subset.update(public_X=arrays["public_X"], control_code=arrays["control_code"][1:], metrics=arrays["metrics"][1:])
    audit_trace(subset, NAMES[1:], STATE["public"], carrier)


def initialize(project, output, protocol):
    require_host()
    precision()
    verify_sources(protocol["source_sha256"])
    assert torch.cuda.is_available()
    prior = protocol["prior"]
    bundle = Path(prior["runtime_bundle"])
    verify_bundle(bundle)
    assert sha256(bundle / "manifest.json") == prior["runtime_manifest_sha256"]
    for path, digest in protocol["artifact_sha256"].items():
        assert sha256(path) == digest, path
    with np.load(bundle / "public.npz") as f:
        public = {k: f[k].copy() for k in ("pilot_qpsk", "probe_controls", "catalog_controls")}
    with patch("numpy.load", guard_load(np.load)):
        base = OnlineController(bundle, "complex_response_cnn", public)
        if protocol["seed"]:
            base.model.load_state_dict(torch.load(protocol["weights"], map_location="cuda", weights_only=True))
            base.model.eval()
        models = [base, Feedback64(base, public)]
    STATE.update(project=Path(project), output=Path(output), protocol=protocol, models=models, public=public)
    write_json(Path(output) / "workers" / f"{os.getpid()}.json",
               dict(pid=os.getpid(), at=now(), seed=protocol["seed"], gpu=torch.cuda.get_device_name(),
                    joint_controller_instantiated=False))


def previous_A(row, carrier, old_input, digest):
    p = STATE["protocol"]
    if not p["reuse_source"]:
        return None
    path = Path(p["reuse_source"]) / "records" / f"environment_{row['index']:05d}" / f"carrier_{carrier:02d}.npz"
    marker = path.with_suffix(".json")
    if not marker.exists():
        return None
    meta = read(marker)
    assert meta["protocol_sha256"] == p["reuse_protocol_sha256"]
    assert meta["environment_id"] == row["environment_id"] and meta["carrier_ghz"] == carrier
    assert meta["original_sha256"] == digest and sha256(path) == meta["sha256"]
    assert meta["checks"]["public_input_bitwise_equal"]
    assert meta["checks"]["feedback_proposal_matches_direct"]
    with np.load(path, allow_pickle=False) as f:
        arrays = {"public_X": f["public_X"].copy(), "control_code": f["control_code"][[0, 2]].copy(),
                  "metrics": f["metrics"][[0, 2]].copy()}
        for field in ("trace_control_code", "trace_scores", "proposal_control_code"):
            arrays["A_feedback64__" + field] = f["A_feedback64__" + field].copy()
    np.testing.assert_array_equal(arrays["public_X"], old_input)
    audit(arrays, carrier)
    return arrays, dict(path=str(path), sha256=meta["sha256"], metadata_sha256=sha256(marker))


def compute(row, carrier, env, old):
    p, public = STATE["protocol"], STATE["public"]
    raw, engine = public_observation(env, carrier, public, p["prior"]["backend"])
    np.testing.assert_array_equal(raw, old["public_X"])
    arrays = {"public_X": raw}
    controls, costs = [], []
    with patch("numpy.load", guard_load(np.load)):
        for i, (name, model) in enumerate(zip(NAMES, STATE["models"])):
            def measure(u, call):
                return engine.measure_detailed(u, rng_for(row["seed"], carrier, 620, call))["score"]
            got = model.decide(raw.copy(), rng_for(0, row["seed"], carrier, 630), measure if i else None)
            controls.append(got["control_code"])
            costs.append([got["feedback_calls"], got["software_seconds"], got["feedback_simulator_seconds"]])
            if i:
                for field in ("trace_control_code", "trace_scores", "proposal_control_code"):
                    arrays[name + "__" + field] = got[field]
    scored = score_decisions(env, carrier, public,
        dict(methods=NAMES, control_code=np.asarray(controls), costs=np.asarray(costs),
             feedback={}, teacher_objective_evaluations=None), p["prior"]["backend"], include_mrc=False)
    arrays.update(control_code=scored["control_code"], metrics=scored["metrics"])
    audit(arrays, carrier)
    if p["seed"] == 0:
        for i, name in enumerate(("complex_response_cnn", "cnn_warm64")):
            j = p["prior"]["methods"].index(name)
            np.testing.assert_array_equal(arrays["control_code"][i], old["control_code"][j])
            np.testing.assert_array_equal(arrays["metrics"][i, [0,1,3,4,5,6]], old["metrics"][j, [0,1,3,4,5,6]])
            np.testing.assert_allclose(arrays["metrics"][i, :10], old["metrics"][j, :10], rtol=1e-12, atol=1e-28)
    return arrays


def one(row):
    p, output = STATE["protocol"], STATE["output"]
    folder = output / "records" / f"environment_{row['index']:05d}"
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for carrier in p["carriers"]:
        path = folder / f"carrier_{carrier:02d}.npz"
        marker = path.with_suffix(".json")
        if marker.exists():
            entry = read(marker)
            assert entry["protocol_sha256"] == fingerprint(p) and sha256(path) == entry["sha256"]
        else:
            if path.exists():
                raise RuntimeError("Uncommitted output requires inspection: " + str(path))
            env, old, digest = original(row, carrier)
            previous = previous_A(row, carrier, old["public_X"], digest)
            if previous is not None and not p["preflight"]:
                arrays, reused = previous
                origin = "reused_verified_A_columns"
            else:
                arrays = compute(row, carrier, env, old)
                reused, origin = None, "new_AF_only_evaluation"
                if previous is not None:
                    prior_arrays, reused = previous
                    for key in arrays:
                        if key == "metrics":
                            np.testing.assert_allclose(arrays[key][:, :10], prior_arrays[key][:, :10], rtol=1e-12, atol=1e-28)
                        else:
                            np.testing.assert_array_equal(arrays[key], prior_arrays[key])
            atomic_npz(path, **arrays)
            entry = dict(environment_id=row["environment_id"], carrier_ghz=carrier,
                path=str(path.relative_to(output)), sha256=sha256(path), protocol_sha256=fingerprint(p),
                original_sha256=digest, origin=origin, reuse_evidence=reused, at=now(),
                checks=dict(public_input_bitwise_equal=True, feedback_audit_passed=True,
                            feedback_proposal_matches_direct=True, joint_correction_used=False))
            write_json(marker, entry)
        entries.append(entry)
        write_json(output / "workers" / f"{os.getpid()}_progress.json",
                   dict(environment=row["index"], carrier=carrier, at=now()))
    record = dict(environment_id=row["environment_id"], index=row["index"], carriers=entries, at=now())
    write_json(folder / "complete.json", record)
    return record


def summarize(output, protocol, records):
    values = []
    for row in records:
        current = []
        for entry in row["carriers"]:
            path = output / entry["path"]
            assert sha256(path) == entry["sha256"]
            with np.load(path, allow_pickle=False) as f:
                audit(f, entry["carrier_ghz"])
                current.append(f["metrics"].copy())
        values.append(current)
    values = np.asarray(values)
    point = metrics(sufficient(values).sum(0))
    write_json(output / "summary.json", [
        dict(seed=protocol["seed"], method=name, **{f: float(point[i, j]) for j, f in enumerate(FIELDS)})
        for i, name in enumerate(NAMES)])
    atomic_npz(output / "environment_metrics.npz", values=values, methods=np.asarray(NAMES),
               environment_ids=np.asarray([r["environment_id"] for r in records]))


def run(project, output, seed, preflight, workers):
    require_host()
    if seed == 0 and not preflight:
        raise ValueError("Seed 0 already has a full audited evaluation; run preflight only")
    source = project / "dataset_simulation/baseline_results/20260926_final864_selected"
    prior = read(source / "protocol.json")
    af = project / "dataset_simulation/revision_rq2_af_20261029/full"
    sources = dict(read(af / "protocol.json")["source_sha256"])
    for path in (Path(__file__), Path(__file__).with_name("SEEDS_PROTOCOL.md")):
        sources[str(path.relative_to(SOURCE))] = sha256(path)
    verify_sources(sources)
    artifacts, weights = {}, None
    oldroot = project / "dataset_simulation/revision_20261029"
    if seed:
        folder = oldroot / "e2_training" / f"seed_{seed}"
        done, training = read(folder / "complete.json"), read(folder / "protocol.json")
        assert done["status"] == "complete" and done["epochs"] == 40 and done["seed"] == seed
        assert done["fingerprint"] == fingerprint(training) and not training["test_used"]
        artifacts = {str(folder / n): h for n, h in done["file_sha256"].items() if n in ("weights.pt", "history.json")}
        artifacts.update({str(folder / n): sha256(folder / n) for n in ("complete.json", "protocol.json")})
        weights = str(folder / "weights.pt")
    reuse, reuse_fingerprint = None, None
    if seed == 1:
        reuse = oldroot / "e2_seed_1_test"
        previous = read(reuse / "protocol.json")
        assert previous["methods"] == ["A_direct16", "AB_direct16", "A_feedback64", "AB_feedback64"]
        assert previous["seed"] == seed and previous["weights"] == weights and previous["prior"] == prior
        assert previous["code_sha256"] == sha256(SOURCE / "study_revision_20261029/evaluate_e2.py")
        assert previous["e1_code_sha256"] == sha256(SOURCE / "study_revision_20261029/run_e1.py")
        assert previous["rows"] == prior["rows"] and previous["carriers"] == prior["carriers"]
        for path, digest in previous["artifact_sha256"].items():
            assert sha256(path) == digest, path
        reuse_fingerprint = fingerprint(previous)
        artifacts[str(reuse / "protocol.json")] = sha256(reuse / "protocol.json")
    p = dict(schema="AF-training-seed-evaluation-v1", seed=seed, source=str(source), prior=prior,
        source_protocol_sha256=sha256(source / "protocol.json"), weights=weights, artifact_sha256=artifacts,
        source_sha256=sources, code_sha256=sha256(Path(__file__)), methods=NAMES,
        reuse_source=str(reuse) if reuse else None, reuse_protocol_sha256=reuse_fingerprint,
        rows=prior["rows"][:1] if preflight else prior["rows"],
        carriers=[4,12,20] if preflight else prior["carriers"], preflight=preflight,
        joint_correction=False, scope="author-authorized AF-only revision; unchanged checkpoints and existing test set")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "complete.json").exists():
            raise RuntimeError("Already complete; no duplicate evaluation")
        if (output / "protocol.json").exists():
            assert read(output / "protocol.json") == p, "Protocol changed"
        write_json(output / "protocol.json", p)
        for name, digest in sources.items():
            dest = output / "source_snapshot" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                assert sha256(dest) == digest
            else:
                shutil.copyfile(SOURCE / name, dest)
        (output / "workers").mkdir(exist_ok=True)
        if preflight:
            initialize(project, output, p)
            records = [one(p["rows"][0])]
            assert one(p["rows"][0])["carriers"] == records[0]["carriers"]
        else:
            for tested_seed in (0, seed):
                proof = read(output.parent / f"preflight_seed_{tested_seed}" / "complete.json")
                assert proof["preflight"] and proof["code_sha256"] == p["code_sha256"]
                assert proof["source_sha256"] == sources
            records = []
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                    initializer=initialize, initargs=(project, output, p)) as pool:
                for future in as_completed([pool.submit(one, row) for row in p["rows"]]):
                    records.append(future.result())
                    write_json(output / "progress.json", dict(status="running", pid=os.getpid(), seed=seed,
                        completed=len(records), total=len(p["rows"]), at=now()))
                    print(f"seed {seed}: {len(records)}/{len(p['rows'])}", flush=True)
        records.sort(key=lambda r: r["index"])
        with np.load(Path(prior["runtime_bundle"]) / "public.npz") as f:
            STATE["public"] = {k: f[k].copy() for k in ("pilot_qpsk", "probe_controls", "catalog_controls")}
        summarize(output, p, records)
        verify_sources(sources)
        for path, digest in artifacts.items():
            assert sha256(path) == digest
        write_json(output / "records.json", records)
        outputs = ("summary.json", "records.json", "environment_metrics.npz", "protocol.json")
        reused_count = sum(e["origin"] == "reused_verified_A_columns" for r in records for e in r["carriers"])
        write_json(output / "complete.json", dict(status="complete", seed=seed, preflight=preflight,
            environments=len(records), carriers=len(p["carriers"]), methods=NAMES, joint_correction=False,
            reused_conditions=reused_count, newly_evaluated_conditions=len(records)*len(p["carriers"])-reused_count,
            source_sha256=sources, code_sha256=p["code_sha256"], protocol_sha256=fingerprint(p),
            file_sha256={n: sha256(output/n) for n in outputs}, at=now()))
        write_json(output / "progress.json", dict(status="complete", seed=seed, completed=len(records),
                   total=len(p["rows"]), at=now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0,1,2), required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1,13), default=8)
    args = parser.parse_args()
    try:
        run(args.project.resolve(), args.output.resolve(), args.seed, args.preflight, args.workers)
    except BaseException:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / f"failure_{os.getpid()}.json", dict(at=now(), traceback=traceback.format_exc()))
        raise
