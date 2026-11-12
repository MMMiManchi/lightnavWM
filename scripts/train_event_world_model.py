"""Run a bounded eight-GPU event latent experiment with persistent logs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv/bin/python"
sys.path.insert(0, str(REPO))

from experiments.world_model.cache_writer import verify_bound_files  # noqa: E402
from experiments.world_model.event_guards import verify_prepared_references  # noqa: E402


def write_status(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    root = Path(config["run_dir"])
    command = [str(PYTHON), "-u", "-m", "experiments.world_model.event_train", "--config", str(config_path)]
    if not (root / "prepared.json").is_file():
        subprocess.run([*command, "--phase", "prepare"], cwd=REPO, check=True)
    if json.loads((root / "config.json").read_text()) != config:
        raise ValueError("prepared dataset config does not match the launch config")
    write_status(root / "causal_binding_check.json", verify_prepared_references(config))
    write_status(root / "encoder_binding_check.json", verify_bound_files(Path(config['encoder_snapshot'])/'model.safetensors'))
    source_files = [*list((REPO/"experiments/world_model").glob("event_*.py")), Path(__file__), config_path]
    import hashlib

    write_status(root/"training_source_manifest.json", {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    pid_path = root / "launcher.json"
    if pid_path.exists():
        pid = json.loads(pid_path.read_text())["pid"]
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f"another launcher may still be running with pid {pid}")
    write_status(pid_path, {"pid":os.getpid(), "config":str(config_path), "started_unix_s":time.time()})
    status_path = root / "status.json"
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    started = time.time()
    try:
        for phase in ("encode", "train"):
            if phase == "train" and not (root / "tensors.npz").exists():
                subprocess.run([*command, "--phase", "pack"], cwd=REPO, check=True)
            processes = []
            handles = []
            for worker in range(config["workers"]):
                if phase == "encode" and (root / f"encoded_{worker}.json").exists():
                    continue
                arm = config["experiments"][worker]
                if phase == "train" and (root / f"{arm['name']}_seed{arm['seed']}" / "result.json").exists():
                    continue
                environment = dict(os.environ)
                environment.update({"CUDA_VISIBLE_DEVICES":str(worker), "OMP_NUM_THREADS":"4", "MKL_NUM_THREADS":"4",
                                    "TOKENIZERS_PARALLELISM":"false", "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1"})
                handle = (logs / f"{phase}_gpu{worker}.log").open("a", buffering=1)
                handles.append(handle)
                process = subprocess.Popen([*command, "--phase", phase, "--worker", str(worker)],
                                           cwd=REPO, env=environment, stdout=handle, stderr=subprocess.STDOUT)
                processes.append((worker,process))
            while True:
                states = [{"gpu":worker, "pid":process.pid, "exit_code":process.poll()} for worker,process in processes]
                write_status(status_path, {"state":phase, "workers":states, "elapsed_s":time.time()-started,
                                           "launcher_pid":os.getpid(), "updated_unix_s":time.time()})
                failures = [r for r in states if r['exit_code'] not in (None,0)]
                if failures:
                    # Only children created by this launcher are stopped.
                    for _,process in processes:
                        if process.poll() is None:
                            process.terminate()
                    for _,process in processes:
                        process.wait()
                    raise RuntimeError(f"{phase} worker failed: {failures}; inspect logs under {logs}")
                if all(r['exit_code'] is not None for r in states):
                    break
                time.sleep(3)
            for handle in handles:
                handle.close()
        subprocess.run([*command, "--phase", "summarize"], cwd=REPO, check=True)
        write_status(status_path, {"state":"complete", "elapsed_s":time.time()-started, "updated_unix_s":time.time()})
    except Exception as exc:
        write_status(status_path, {"state":"failed", "error":repr(exc), "elapsed_s":time.time()-started})
        raise


if __name__ == "__main__":
    main()
