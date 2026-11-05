# Implementation verification

Local verification on Windows, Python 3.11, 2026-11-05:

- 57 automated tests passed, including both `DummyVecEnv` and Windows `SubprocVecEnv` training/save/load tests.
- Ruff lint and formatting checks passed.
- A real CLI run trained MaskablePPO for 1,024 transitions with the requested 100-round session horizon, saved checkpoints and completed 100 deterministic held-out test sessions.
- Separate CLI checks exercised sampled evaluation, continuation from `latest.zip`, baseline simulation, and cross-run comparison with uncertainty estimates.
- A generated PPO report contained 151 interactive charts and 151 exports in each of PNG, SVG, and PDF, with underlying CSV, Parquet, and Plotly JSON artifacts.
- Public-information isolation, card exposure deduplication, exact cash/escrow settlement, reward telescoping, and monetary scale equivalence have automated coverage.

Generated artifacts are under ignored `runs/`. Example verification run: `runs/20261105T153021Z-ppo-seed-11-f002a366c2b8`; comparison: `runs/verification-comparison`. The run manifest records the configuration and source state at the time of execution. These are short verification artifacts, not a trained profitable strategy: the small PPO run performed poorly and went bankrupt in every test session. Full pilot and multi-seed experiment configurations are provided but were not executed as part of implementation verification.

The local `.venv` reused the machine's existing scientific packages through `--system-site-packages`, with missing pinned RL/testing dependencies installed into `.venv`. The project dependency closure was recorded in `requirements-lock.txt`. Whole-environment `pip check` also sees unrelated existing Anaconda packages and reports their existing missing/conflicting dependencies (conda, conda-repo-cli, gensim, and s3fs). Those packages are outside the project dependency closure. Use the clean virtual-environment installation in README for an isolated project setup. Hosted CI is configured for Windows/Linux but has not been run here.

The only warning in the passing automated suite came from Plotly 5.9's use of NumPy's deprecated `np.bool8` alias. Figures and exports completed successfully. No test requires positive blackjack profit or treats a short smoke budget as evidence of convergence.

Architecture decisions, added guards, scope restrictions, objective interpretation, and extension guidance are documented in README and the original implementation plan. The optional Monte Carlo hidden-state action-value probe is a future research extension; the implemented policy charts query probabilities on observed reachable public states.
