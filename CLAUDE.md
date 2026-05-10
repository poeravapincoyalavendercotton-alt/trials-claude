# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

Two unrelated pieces of code live here:

- `slack_reader.py` — standalone Slack `conversations.history` reader.
- `oracle/` + `solution/` — an inverse-problem pair. `oracle/oracle.py` holds hidden physical parameters and emits noisy S-parameter `.s2p` data; `solution/main.py` queries the oracle and fits the parameters back. They are coupled only through `oracle.handle_query`, which `solution/main.py` imports via a `sys.path` hack.

`requirements.txt` only lists `slack_sdk`. The oracle/solution code additionally requires `scikit-rf` (`skrf`), `numpy`, and `scipy` — install these manually when working on that side.

## Common commands

```bash
# Slack reader (needs SLACK_TOKEN exported, scopes per docstring in slack_reader.py)
python slack_reader.py <channel_id> [limit]

# Run the full oracle→solution fit (writes progress to stdout; long-running)
python solution/main.py
```

There is no test suite, linter config, or build system. Single-test invocation does not apply.

## Oracle/solution architecture

The oracle simulates a 100-segment cascaded transmission line on a Havriliak–Negami dielectric substrate and returns a Touchstone `.s2p` string per call. Four `mode` values expose different views of the same underlying model:

- `measurement1`: segments `n = 0..99`.
- `measurement2`: segments `n = 1..99` (drops the first segment — used to disambiguate the indexing convention).
- `measurement3`: same as `measurement2` but prepended with a fixed `Z = 75 Ω` "tag" segment.
- `measurement4`: segments `n = 0..99` with an extra exponential taper `exp(eta * n)` applied to each `Z_n`.

The per-segment impedance is `Z_n = 50 · exp(m · cos(2π·b·n + φ))`. Hidden ground-truth constants live at the top of `handle_query` in `oracle/oracle.py` (`m_true`, `b_true`, `phi_true`, `alpha_true`). All four measurements share the same `(m, b, φ, α)`; `measurement3` adds `Z_tag`, `measurement4` adds `eta`. Output is corrupted by `add_noise_polar(mag_dev=0.002, phase_dev=0.5)`.

`solution/main.py` mirrors the oracle's physics in a vectorised ABCD-cascade (`cascade_chain`) rather than calling `skrf` per segment, then fits parameters by:

1. `solve_first_three_modes` — joint DE + L-BFGS-B fit of `(m, b, φ, α, log Z_tag)` against measurements 1–3, run once with `n_start=0` and once with `n_start=1`. A coarse frequency subsample (`coarse_freq_idx`, every 8th point) is used for the global stage; the local polish uses the full grid.
2. `fit_eta_from_measurement4` — bounded scalar fit of `eta` only, holding the other five fixed. The `n_start` whose `eta` fit yields lower residual on measurement4 is taken as the correct indexing convention.
3. `solve_all_four_modes` — final joint 6-parameter refinement against all four measurements at the chosen `n_start`.

When changing the model, keep the substrate constants (`epsilon_inf`, `delta_epsilon`, `tau`, `beta_frac`, `length_m`) in `oracle/oracle.py` and the mirrored constants in `solution/main.py` (`eps_inf`, `delta_eps`, `tau_s`, `beta_frac`, `seg_len`, `z_ref`) in sync — the fitter assumes they match the oracle exactly and only the four hidden parameters are unknown.

Calls to `oracle.handle_query` are non-deterministic (noise is added each call) and not cached; rerunning `solution/main.py` re-samples all four measurements at import time in the module-level `meas_db` dict.
