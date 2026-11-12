"""Prepare and train an event-level latent world model from admitted formal runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.world_model.event_dataset import construct_event_pairs
from experiments.world_model.event_latent import (
    EventLatentWorldModel,
    EventModelConfig,
    event_objective,
)
from experiments.world_model.feature_cache import (
    AUDITED_PREPROCESSOR,
    SIGLIP_REVISION,
    SIGLIP_WEIGHT_SHA256,
)
from experiments.world_model.latent_data import (
    load_accepted_run_set,
    read_admitted_rgb,
    scene_partition,
)
from experiments.world_model.precollection_protocol import assert_training_authorized

REPO = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def save(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("x") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prepare(config: dict) -> None:
    assert_training_authorized()
    root = Path(config["run_dir"])
    root.mkdir(parents=True, exist_ok=False)
    save(root / "config.json", config)
    save(root / "authorization.json", {
        "source": "Explicit user instruction in this Codex conversation on 2026-11-04: 那可以开始训练了吗，8卡均可用",
        "scope": "feature extraction and event latent world-model training on GPUs 0-7",
        "raw_records_modified": False, "navigation_evaluation_started": False,
    })
    annotation = Path(config["annotations"])
    hashes = {str(p):digest(p) for p in annotation.glob("*.json*")}
    for name in ["event_dataset.py", "event_latent.py", "event_train.py"]:
        p = REPO / "experiments/world_model" / name
        hashes[str(p)] = digest(p)
    frames = read_rows(annotation / "frames.jsonl")
    pairs, counts = construct_event_pairs(frames, read_rows(annotation / "places.jsonl"))
    print("event candidates", counts, flush=True)
    sources = []
    key = Path(config["approval_key"])
    for name in config["source_runs"]:
        run = REPO / "output/world_model_data" / name
        folder = "motion_supplement_20261030_01" if "radius010" in name else "first_v1_20261029"
        approval = REPO / "output/world_model_approvals" / folder / f"{name}.approval.json"
        sources.append((run, approval, key))
        hashes[str(approval)] = digest(approval)
    print("admitting immutable source snapshots", len(sources), flush=True)
    bundle = load_accepted_run_set(sources)
    print("admitted", len(bundle["observations"]), "observations", flush=True)
    observations = {r["observation_id"]:r for r in bundle["observations"]}
    episodes = {r["episode_uid"]:r for r in bundle["episodes"]}
    stage = root / "admitted_rgb"
    stage.mkdir()
    needed = {oid for p in pairs for oid in [*p["history_observation_ids"], p["target_observation_id"]] if oid is not None}
    image_rows = {}
    observation_images = {}
    weak = set()
    for oid in sorted(needed):
        row = observations[oid]
        pixels = read_admitted_rgb(row["rgb"])
        image_id = row["rgb"]["sha256"]
        observation_images[oid] = image_id
        if np.mean(np.max(pixels, axis=-1) <= 5) > 0.5:
            weak.add(image_id)
        if image_id in image_rows:
            continue
        destination = stage / f"{image_id}.npy"
        np.save(destination, pixels, allow_pickle=False)
        destination.chmod(0o444)
        image_rows[image_id] = {
            "image_id": image_id, "staged_path": str(destination),
            "staged_sha256": digest(destination), "source_observation_id": oid,
            "source_pixel_sha256": row["rgb"]["pixel_sha256"],
        }
    accepted = []
    for p in pairs:
        ep = episodes[p["episode_uid"]]
        if scene_partition(p["scene"]) != p["partition"] or ep["scene"] != p["scene"]:
            raise ValueError("event sample disagrees with the admitted scene split")
        image_ids = [observation_images[oid] if oid is not None else None for oid in p["history_observation_ids"]]
        target_id = observation_images[p["target_observation_id"]]
        if image_ids[-1] in weak or target_id in weak or target_id == image_ids[-1]:
            counts["weak_or_identical_endpoint"] = counts.get("weak_or_identical_endpoint", 0) + 1
            continue
        p["route_group_id"] = ep["route_group_id"]
        p["history_image_ids"] = image_ids
        p["history_valid"] = [i is not None and i not in weak for i in image_ids]
        p["target_image_id"] = target_id
        accepted.append(p)
    fit = [p for p in accepted if p["partition"] == "train_fit"]
    dev = [p for p in accepted if p["partition"] == "train_dev"]
    fit_scenes, dev_scenes = {p["scene"] for p in fit}, {p["scene"] for p in dev}
    if fit_scenes & dev_scenes or not fit or not dev:
        raise ValueError("event split is empty or leaks scenes")
    write_rows(root / "pairs.jsonl", accepted)
    write_rows(root / "images.jsonl", list(image_rows.values()))
    save(root / "dataset_report.json", {
        "construction": counts, "samples": {"fit":len(fit), "dev":len(dev)},
        "scenes": {"fit":sorted(fit_scenes), "dev":sorted(dev_scenes)},
        "route_groups": {"fit":len({p['route_group_id'] for p in fit}), "dev":len({p['route_group_id'] for p in dev})},
        "by_source": dict(Counter(p["source_run"] for p in accepted)),
        "unique_staged_images": len(image_rows), "formal_admission": bundle["admission_mode"],
        "target": "first RGB at next different event; normalized frozen SigLIP pooler, not pixels or node id",
        "branch_teacher": "current angular and depth-clear candidate matched to observed departure; geometry only labels the action",
        "model_inputs": ["past/current frozen visual features", "selected relative direction sin/cos"],
        "official_geometry_is_online_input": False, "future_trajectory_is_model_input": False,
        "data_kind": "event geometry weak supervision; not a demonstrated online navigation system",
    })
    for path, h in hashes.items():
        if digest(Path(path)) != h:
            raise RuntimeError(f"source changed while preparing: {path}")
    save(root / "input_manifest.json", hashes)
    save(root / "prepared.json", {"success":True, "at_unix_s":time.time()})
    print("prepared event data", len(fit), len(dev), "images", len(image_rows), flush=True)


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback for feature encoding or training")


def encode(config: dict, worker: int) -> None:
    from transformers import SiglipImageProcessor, SiglipVisionModel

    assert_training_authorized()
    require_cuda()
    root = Path(config["run_dir"])
    if not (root / "prepared.json").is_file():
        raise RuntimeError("dataset preparation incomplete")
    checkpoint = Path(config["encoder_snapshot"])
    weights = checkpoint / "model.safetensors"
    if digest(weights) != SIGLIP_WEIGHT_SHA256:
        raise RuntimeError("frozen SigLIP checkpoint does not match the audited hash")
    processor = SiglipImageProcessor.from_json_file(str(AUDITED_PREPROCESSOR))
    torch.set_float32_matmul_precision("highest")
    model = SiglipVisionModel.from_pretrained(checkpoint, local_files_only=True, dtype=torch.float32).to("cuda:0").eval()
    images = read_rows(root / "images.jsonl")[worker::config["workers"]]
    values = []
    batch_size = config["encoder_batch_size"]
    for start in range(0, len(images), batch_size):
        batch = images[start:start+batch_size]
        pixels = []
        for r in batch:
            path = Path(r["staged_path"])
            if digest(path) != r["staged_sha256"]:
                raise ValueError("admitted RGB stage changed")
            pixels.append(np.load(path, allow_pickle=False))
        tensor = processor(images=pixels, return_tensors="pt")["pixel_values"].to("cuda:0")
        with torch.inference_mode():
            features = model(tensor).pooler_output.float().cpu().numpy()
        if features.shape != (len(batch),1152) or not np.isfinite(features).all():
            raise ValueError("invalid frozen features")
        values.append(features)
        print("encoded", worker, start+len(batch), "/", len(images), flush=True)
    np.savez(root / f"features_{worker}.npz", ids=np.array([r["image_id"] for r in images]), features=np.concatenate(values))
    save(root / f"encoded_{worker}.json", {"images":len(images), "revision":SIGLIP_REVISION, "dtype":"float32"})


def pack(config: dict) -> None:
    root = Path(config["run_dir"])
    feature_map = {}
    for worker in range(config["workers"]):
        with np.load(root / f"features_{worker}.npz") as data:
            feature_map.update(zip(data["ids"].tolist(), data["features"], strict=True))
    pairs = read_rows(root / "pairs.jsonl")
    ids = sorted({p["target_image_id"] for p in pairs if p["partition"] == "train_fit"} | {
        i for p in pairs if p["partition"] == "train_fit" for i,v in zip(p["history_image_ids"], p["history_valid"], strict=True) if v
    })
    fit_features = np.stack([feature_map[i] for i in ids])
    mean, scale = fit_features.mean(0), fit_features.std(0).clip(.01)

    def transform(vector):
        x = (vector-mean)/scale
        return x / max(float(np.linalg.norm(x)), 1e-8)

    history = np.zeros((len(pairs),8,1152), dtype=np.float32)
    for j,p in enumerate(pairs):
        for k,(image_id, valid) in enumerate(zip(p["history_image_ids"], p["history_valid"], strict=True)):
            if valid:
                history[j,k] = transform(feature_map[image_id])
    labels = {key:i for i,key in enumerate(sorted({p['offline_labels']['target_place'] for p in pairs}))}
    multiplicity = Counter((p['route_group_id'],p['offline_labels']['origin_place'],p['offline_labels']['target_place']) for p in pairs)
    np.savez(root / "tensors.npz", history=history,
             valid=np.array([p['history_valid'] for p in pairs]),
             target=np.stack([transform(feature_map[p['target_image_id']]) for p in pairs]),
             branch=np.array([p['branch_direction'] for p in pairs],dtype=np.float32),
             target_labels=np.array([labels[p['offline_labels']['target_place']] for p in pairs]),
             fit=np.array([p['partition']=='train_fit' for p in pairs]),
             weights=np.array([1/multiplicity[(p['route_group_id'],p['offline_labels']['origin_place'],p['offline_labels']['target_place'])] for p in pairs],dtype=np.float32),
             feature_mean=mean, feature_scale=scale)
    save(root / "feature_contract.json", {"encoder":"google/siglip-so400m-patch14-384", "revision":SIGLIP_REVISION,
        "output":"pooler_output", "feature_dim":1152, "normalization":"fit-only channel standardization then L2",
        "fit_statistics_image_count":len(ids), "target_encoder_frozen":True,
        "memory_keys_require_same_normalization":True, "normalization_file":"tensors.npz"})


def metrics(prediction: np.ndarray, target: np.ndarray, labels: np.ndarray, scenes: list[str]) -> dict:
    cosine = (prediction*target).sum(-1)
    order = np.argsort(-(prediction @ target.T),axis=1)
    hit = labels[order] == labels[:,None]
    rank = hit.argmax(1)+1
    return {"cosine":float(cosine.mean()), "recall_at_1":float(hit[:,0].mean()),
            "recall_at_5":float(hit[:,:5].any(1).mean()), "mrr":float((1/rank).mean()),
            "scene_macro_cosine":float(np.mean([cosine[np.array(scenes)==s].mean() for s in sorted(set(scenes))])),
            "gallery":"offline held-out true future features, not an online accessible memory"}


def train(config: dict, worker: int) -> None:
    assert_training_authorized()
    require_cuda()
    root = Path(config['run_dir'])
    arm = config['experiments'][worker]
    destination = root / f"{arm['name']}_seed{arm['seed']}"
    destination.mkdir(exist_ok=False)
    seed = arm['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    arrays = np.load(root / 'tensors.npz')
    pair_meta = read_rows(root / 'pairs.jsonl')
    model_config = EventModelConfig(**config['model'],use_branch=arm['use_branch'],stochastic=arm['stochastic'])
    model = EventLatentWorldModel(model_config).to('cuda:0')
    tensors = {k:torch.from_numpy(arrays[k]).to('cuda:0') for k in ['history','valid','target','branch','target_labels']}
    fit_ids, dev_ids = np.flatnonzero(arrays['fit']), np.flatnonzero(~arrays['fit'])
    fit_tensor = torch.as_tensor(fit_ids,device='cuda:0')
    dev_tensor = torch.as_tensor(dev_ids,device='cuda:0')
    scenes = [pair_meta[i]['scene'] for i in dev_ids]
    target = arrays['target'][dev_ids]
    target_labels = arrays['target_labels'][dev_ids]
    weights = arrays['weights'][fit_ids].astype(np.float64)
    weights /= weights.sum()
    generator = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=.01)
    baseline_current = arrays['history'][dev_ids,-1]
    fit_current = arrays['history'][fit_ids,-1]
    nearest = (baseline_current @ fit_current.T).argmax(1)
    baseline_nn = arrays['target'][fit_ids][nearest]
    save(destination / 'baselines.json', {'current':metrics(baseline_current,target,target_labels,scenes),
                                         'nearest_training_successor':metrics(baseline_nn,target,target_labels,scenes)})
    save(destination / 'model_config.json', {**model.configuration(), 'parameters':sum(p.numel() for p in model.parameters()), **arm})
    best, stale = -math.inf, 0
    started = time.time()
    log = (destination / 'history.jsonl').open('x', buffering=1)
    for step in range(1,config['updates']+1):
        model.train()
        chosen = generator.choice(len(fit_ids),size=config['batch_size'],replace=True,p=weights)
        index = fit_tensor[torch.as_tensor(chosen,device='cuda:0')]
        optimizer.param_groups[0]['lr'] = config['learning_rate']*min(1.,step/100)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            outputs = model.training_outputs(tensors['history'][index],tensors['valid'][index],tensors['branch'][index],tensors['target'][index])
            loss, parts = event_objective(outputs,tensors['target'][index],tensors['target_labels'][index])
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step == 1 or step % config['evaluate_every'] == 0:
            model.eval()
            with torch.inference_mode():
                prediction = model.predict(tensors['history'][dev_tensor],tensors['valid'][dev_tensor],tensors['branch'][dev_tensor]).cpu().numpy()
                shuffled = model.predict(tensors['history'][dev_tensor],tensors['valid'][dev_tensor],tensors['branch'][dev_tensor].roll(1,0)).cpu().numpy()
            result = metrics(prediction,target,target_labels,scenes)
            row = {'step':step,'loss':float(loss.detach()),'train_parts':{k:float(v.detach()) for k,v in parts.items()},
                   'prior_dev':result,'shuffled_branch_dev':metrics(shuffled,target,target_labels,scenes),
                   'branch_response_l2':float(np.linalg.norm(prediction-shuffled,axis=1).mean()),'elapsed_s':time.time()-started}
            log.write(json.dumps(row)+'\n')
            print(json.dumps({'worker':worker,'arm':arm['name'],'seed':seed,**row}),flush=True)
            if result['scene_macro_cosine'] > best:
                best, stale = result['scene_macro_cosine'], 0
                torch.save({'model_state_dict':model.state_dict(),'model_config':model.configuration(),
                            'step':step,'evaluation':row,'feature_contract':json.loads((root/'feature_contract.json').read_text())},destination/'best.pt')
                np.save(destination/'best_prior_predictions.npy',prediction)
            else:
                stale += 1
            if stale >= config['patience_evaluations']:
                break
    log.close()
    checkpoint = torch.load(destination/'best.pt',map_location='cpu',weights_only=False)
    save(destination/'result.json',{'arm':arm,'best_step':checkpoint['step'],'best':checkpoint['evaluation'],
                                  'updates_completed':step,'elapsed_s':time.time()-started,'checkpoint':str(destination/'best.pt'),
                                  'blind_test':False,'navigation_connected':False})


def summarize(config: dict) -> None:
    root = Path(config['run_dir'])
    results = []
    for arm in config['experiments']:
        results.append(json.loads((root/f"{arm['name']}_seed{arm['seed']}"/'result.json').read_text()))
    save(root/'results.json',{'experiments':results,'evaluation':'development set used for checkpoint selection; not a blind navigation test'})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','encode','pack','train','summarize'],required=True)
    parser.add_argument('--worker',type=int,default=0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.phase in ('encode','train'):
        globals()[args.phase](config,args.worker)
    else:
        globals()[args.phase](config)


if __name__ == '__main__':
    main()
