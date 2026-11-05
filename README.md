# Blackjack RL

A Python/Stable Baselines3 research environment where one agent chooses wagers and plays finite-shoe blackjack against a dealer. Sessions start at $100, require a $10 minimum opening bet, and last up to 100 settled rounds. These settings are configurable.

## Install

Use Python 3.11 and a virtual environment. The pinned stack uses SB3 2.3.2, sb3-contrib 2.3.0, Gymnasium 0.29.1 and Torch 2.2.1. These versions were chosen as a mutually compatible set and verified locally; the project does not require the latest releases.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
```

`requirements-lock.txt` records the verified Windows/Python 3.11 dependency closure. CPU/CUDA Torch wheel selection can be changed through the appropriate official PyTorch index. Other Python versions/platforms may require regenerating the lock. To regenerate after intentional dependency changes, run `python scripts/lock_dependencies.py` in the environment you verified.

## Quick start

Activate the environment or replace `blackjack-rl` with `.\.venv\Scripts\python.exe -m blackjack_rl`.

```powershell
.\.venv\Scripts\Activate.ps1
blackjack-rl validate-config --config configs/train.yaml --check-env
blackjack-rl simulate --config configs/evaluation.yaml --agent basic --sessions 100
blackjack-rl train --config configs/experiments/smoke.yaml
```

Each command prints its run directory. Use that path for subsequent commands:

```powershell
blackjack-rl report --run runs/<run>
blackjack-rl evaluate --run runs/<run> --sessions 100 --sampled
blackjack-rl compare --runs runs/<basic-run> runs/<ppo-run>
tensorboard --logdir runs
```

Open the generated `report/index.html` locally. Charts support zoom, hover, trace selection, and chart-name search. Each chart has CSV/Parquet aggregates, Plotly JSON, and configured PNG/SVG/PDF exports. Exporting many static figures can take substantially longer than generating HTML. Disable static exports with `--set 'plots.static_formats=[]'`. Reports are created explicitly so large export work does not interrupt training.

The smoke config trains briefly on shorter sessions and uses a small evaluation sample. It checks the pipeline and is not a trained blackjack strategy. To run a single-seed pilot with the requested 100-round game:

```powershell
blackjack-rl train --config configs/experiments/pilot.yaml
```

`configs/train.yaml` schedules the full configured budget for each independent seed and runs a final held-out evaluation for each selected policy. Training budget is measured in policy decisions, not rounds. PPO may finish a rollout beyond the requested budget. Progress bars, console diagnostics and TensorBoard show transitions, rounds, completed sessions, wealth, ruin, rewards, optimizer diagnostics and throughput.

## Game and information rules

- Six decks, shuffled without replacement, with the cut at 75% penetration. A shuffle occurs before the next bet, never after a bet is accepted. A conservative card reserve can cause an earlier shuffle; its reason and actual penetration are logged.
- Dealer stands on soft 17 by default and peeks for blackjack with an ace or ten-valued upcard.
- Naturals pay 3:2; ordinary wins pay 1:1; ties push.
- Aces are usable when counting an ace high does not bust the hand.
- Double on any eligible initial two-card hand if the extra stake is affordable: commit the same stake, draw one card, and finish the hand.
- No splitting, surrender, insurance, sitting out or voluntary session cash-out. Stand ends the current hand's player decisions.
- The bet ladder combines configured minimum-bet multiples and bankroll fractions, rounded down to betting increments. Duplicate economic wagers are masked. No artificial bankroll cap is imposed.
- All wagers are funded and held in escrow. All-in cash during a pending hand is not bankruptcy. Bankruptcy is settled bankroll below the minimum bet, including any small residual balance.
- The agent sees exposed rank counts, Hi-Lo running/true counts, shoe depth, its hand and dealer upcard. It never sees shoe order or the hidden dealer rank. Unrevealed cards discarded after player bust remain unknown. True count uses an exact dealt-card depth estimate in this simulator; this is more precise than real visual deck estimation.

This is an explicitly restricted ruleset. The configured reference strategy is a multi-deck S17 hit/stand/double table with no splitting. Changing to H17 does not automatically replace that reference table; label such comparisons appropriately. No strategy here is claimed optimal or profitable.

## Reward

At settlement, reward is the bankroll change divided by starting bankroll. At termination, subtract the configured ruin penalty and a proportional shortfall penalty for finishing below starting bankroll:

```text
session_return = final_bankroll / initial_bankroll - 1
                 - ruin_weight * I(final_bankroll < minimum_bet)
                 - shortfall_weight * max(0, 1 - final_bankroll / initial_bankroll)
```

Defaults are a ruin weight of 10 and a shortfall weight of 2. With no penalties, $100 -> $150 and $1000 -> $1500 both return +0.5. Fully scale the minimum and betting increment too when testing equivalent games. Keeping the minimum fixed changes risk even though reward remains relative.

Funded wagers prohibit negative cash. The request to penalize ending losses is implemented as a penalty for negative **net profit**. A negative-cash coefficient is reserved in the reward formula, but negative cash is an accounting failure in this no-credit game. A busted hand and a bankrupt session are distinct events. Ruin is checked after the final round too. Gamma is 1 so rewards sum to the stated session objective without favoring hands with fewer decisions. Risk penalties change the money-only objective and do not guarantee a hard ruin-probability constraint.

## Configuration and experiments

All numeric behavior settings live in `configs/`; modules derive dimensions and action IDs from those settings. Composition uses relative `include` files. Unknown fields, unsupported rule combinations, nonfinite numeric values, invalid payout increments, unaffordable initial bankrolls and incompatible PPO batch sizes fail validation. Config values are validated when loaded; use overrides or reload a changed configuration rather than mutating nested settings in application code.

```powershell
blackjack-rl train --config configs/experiments/pilot.yaml --set reward.ruin_weight=20 --set shoe.penetration_fraction=0.6
blackjack-rl train --config configs/experiments/no_count.yaml --seed 11
blackjack-rl train --config configs/experiments/hilo.yaml --seed 11
blackjack-rl train --config configs/experiments/fixed_bet.yaml --seed 11
blackjack-rl train --config configs/experiments/betting_only.yaml --seed 11
blackjack-rl train --config configs/experiments/shuffle_every_round.yaml --seed 11
```

`counting=none` removes all historical rank/count/depth features, preserving current hand/upcard information. `hilo` includes counts and depth; `full` additionally includes exposed rank frequencies. Retrain for feature ablations. `control=play` forces the minimum opening bet, and `control=betting` forces the configured reference playing action. Both retain the same observation/action dimensions and expose their actual legal masks to the policy.

Rules and reward changes during evaluation require explicit scenario labeling:

```powershell
blackjack-rl evaluate --run runs/<run> --allow-changed-scenario --set rules.dealer_hits_soft_17=true
```

The override is recorded with the training config hash and full evaluation configuration. Structural changes such as a different action-ladder length or rank vocabulary require a compatible model; the flag does not adapt a neural network. Report a separate evaluation using `--data runs/<run>/evaluations/<evaluation>`; its saved evaluation configuration is selected automatically. Report filters use JSON values, for example `--filter soft=true`, `--filter 'phase="player"'`, or `--filter 'true_count={"min":0,"max":3}'`. Range endpoints are inclusive. Filtering applies to tables containing that field; the report records the filters. Filtered reports omit whole-session metric summaries because those summaries refer to the unfiltered population.

Resume into a new run directory:

```powershell
blackjack-rl train --config runs/<run>/config.json --seed 11 --resume runs/<run>/latest.zip --set training.total_timesteps=100000
```

The seed must match the saved run. Resume permits a new additional training budget, output root, progress setting and final-test switch; other experiment settings must match. Learning resumes with saved optimizer/policy state, but worker games, RNG streams and partial rollout state restart. This is not bit-for-bit continuation. Ctrl+C during learning saves `latest.zip` and completed telemetry, marks the run interrupted, and prevents remaining seed runs from starting. Incomplete sessions are not classified as bankrupt.

## Research outputs and evaluation

Every run saves resolved config/hash, software versions, source revision/dirty state, hardware, seed manifests and schema version. Training saves complete session summaries plus sampled decision/round/card traces. Evaluation saves all session summaries and a configurable number of full traces. Parquet writes are buffered and atomic per chunk; existing evaluation directories are rejected to prevent mixing populations.

The chart registry generates the applicable catalog from available data, including:

- Reward components and financial metrics against decisions and wall time; validation intervals; PPO losses, KL, entropy, clipping, explained variance and throughput.
- Wealth distributions/ECDFs, drawdown, survival/ruin timing, population bankroll quantiles, and separately labeled survivor curves.
- Hard/soft and initial/later action-probability maps, visitation, reference disagreements, action frequency, legal-double rates, confidence, entropy and action transitions.
- Bet distributions, count/bankroll/horizon/depth relationships, variance and coefficient of variation, wager changes, lag dependence, all-in/minimum rates, double affordability, conditional profit and stake-normalized return.
- Count trajectories and distributions, shuffle depth/reasons, first-card frequencies by pre-bet count, and count/depth outcome maps.
- Across-run risk/growth and reward comparisons, hierarchical model/session uncertainty and matched-seed session differences.

Chart availability depends on observed events and sample thresholds. Constant-series correlations and unsupported cells remain missing. Outcomes conditional on chosen actions are observational and are not causal action values. The policy value network is never presented as a per-action Q-function. Optional hidden-state Monte Carlo action-value probes from the plan remain a future extension; this implementation queries action probabilities on actual reachable public states.

Validation and test seed manifests are separate. Best checkpoints are selected by validation reward, including a validation after the final PPO update; the test split is not used for selection. Repeatedly inspecting a test split during tuning still compromises that experimental separation, so use validation for development. Deterministic and sampled policies can be evaluated independently. `random` remains random in either mode. Confidence intervals resample whole sessions; comparison groups with independent models use a hierarchical model/session bootstrap. Wilson ruin intervals remain nonzero above zero even when no ruin was observed. Matched shoe seeds reduce variance but do not produce identical hands when policies consume different numbers of cards.

## Architecture and extension points

`game/` owns cards, shoe, public ledger, funded money and the state machine, without Gym/Torch dependencies. `env/` maps public snapshots into Gymnasium observations, legal action masks and rewards. `agents/` contains baselines and MaskablePPO integration. `training/` coordinates workers, telemetry, validation and saves. `evaluation/` produces held-out data and statistics. `reporting/` consumes saved tables and registers chart generators; it does not interact with training state.

The feed-forward agent receives a counting ledger rather than learning memory from raw card sequences. This works with MaskablePPO, which does not provide recurrent policies. Hidden discards still make the game partially observable. Add a feature through the public snapshot/encoder, a baseline through `probabilities(...)`, or a chart with a registered generator yielding `Chart` objects. Adding a rule requires engine, mask, config-validation and settlement-test changes together. Invalid actions raise errors; the legal-action adapter exists only for the random-action Gym/SB3 interface checker.

## Verify

```powershell
python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

Tests cover scripted natural/normal/double settlements, multiple aces, peek and soft-17 behavior, masks, cash conservation, hidden-card isolation, finite shoes and shuffles, reward telescoping and scaling, replay, configuration failures, reports and both dummy/subprocess SB3 save/load workflows. Artifact IDs use a configured compact length to keep nested Windows paths manageable; each run retains the full configuration hash. Statistical tests do not assume positive profit. A short verification run is not evidence of convergence; use the pilot/full multi-seed configs and held-out baselines for research claims.

The original design rationale and full proposed experiment matrix are in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Library integration references: [MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html), [Gymnasium Env](https://gymnasium.farama.org/api/env/), and [SB3 TensorBoard](https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html).
