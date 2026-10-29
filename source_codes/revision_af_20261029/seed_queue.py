"""单实例 S 队列：预检后继续 A/F 评价，不恢复旧 E2 队列。"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main(project):
    root = project / "dataset_simulation/revision_af_20261029/S"
    root.mkdir(parents=True, exist_ok=True)
    script = project / "source_codes/revision_af_20261029/evaluate_seeds.py"
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    tasks = [(s, True) for s in (0,1,2)] + [(s, False) for s in (1,2)]
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for seed, preflight in tasks:
            name = f"preflight_seed_{seed}" if preflight else f"seed_{seed}"
            output = root / name
            proof = output / "complete.json"
            if proof.exists():
                item = json.loads(proof.read_text())
                if item["code_sha256"] != digest or item["preflight"] != preflight:
                    raise RuntimeError("Completed output uses a different executor")
                continue
            cmd = [sys.executable, str(script), "--project", str(project), "--output", str(output),
                   "--seed", str(seed), "--workers", "8"]
            if preflight:
                cmd.append("--preflight")
            with (root / f"{name}.log").open("a") as log:
                child = subprocess.Popen(cmd, cwd=project, stdout=log, stderr=subprocess.STDOUT)
                write(root / "queue_state.json", dict(status="running", queue_pid=os.getpid(),
                    child_pid=child.pid, task=name, at=now(), executor_sha256=digest))
                rc = child.wait()
            if rc:
                write(root / "queue_state.json", dict(status="failed", task=name, exit_code=rc,
                    at=now(), queue_pid=os.getpid()))
                raise RuntimeError(f"{name} failed with code {rc}")
        write(root / "queue_state.json", dict(status="evaluations_complete", at=now(),
            queue_pid=os.getpid(), executor_sha256=digest, analysis_complete=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    main(parser.parse_args().project.resolve())
