# lightnavWM

Personal research workspace for a lightweight latent world model on top of LightNav-style navigation.

Author: **Liu Bingyu ([@MMMiManchi](https://github.com/MMMiManchi))**

This repository is **not** the public LightNav-0 release. It keeps a research snapshot of world-model collection, event / branch supervision, scene memory, and closed-loop harness work. Upstream LightNav-0 docs, model cards, and team branding do not apply here.

## Focus

The working goal is:

**lightweight latent world model → predict future state as a query → retrieve related history from episode / scene memory → feed retrieved context into navigation decisions.**

Current emphasis in this tree:

- train-only transition collection and admission gates
- event / branch / viewpoint supervision
- scene familiarity and cross-episode memory probes
- predictive / continual harness evaluation in Habitat

## Layout

| Path | Role |
|---|---|
| `experiments/world_model/` | Collection, contracts, event/branch/memory code |
| `scripts/` | Launchers for collect / train / eval harnesses |
| `docs/world_model_*.md` | Design notes, audits, and handoff records |
| `tests/world_model/` | CPU-side contract and unit tests |
| `src/lightnav/` | Serving / inference stack used by the harness |
| `habitat_server/` | Habitat env server (separate Python 3.9 env) |

Large run outputs (`output/`), local datasets (`data/`), and checkpoints stay on the machine and are not the point of this README.

## Status

This is an active research snapshot. APIs, directory names, and evaluation protocols change as experiments land. Prefer the dated notes under `docs/` and the entry scripts under `scripts/` over any assumption copied from LightNav-0 marketing docs.

## License

Apache License 2.0. See `LICENSE` and `THIRD_PARTY_NOTICES.md` for third-party code that remains in the tree.
