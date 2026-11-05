# Blackjack RL implementation handoff

## 1. Goal and deliverables

Build a reproducible Python research project using Stable Baselines3 (SB3) and `sb3-contrib`. Train one agent to choose both wagers and playing actions against a rule-based dealer. Deliver a tested simulator, training CLI with progress reporting, checkpoints, reproducible evaluation, TensorBoard logs, and an extensible research report generator.

The repository currently contains only a short README and `.gitignore`. This document is a plan, not an implemented or trained system.

### User requirements

- Initial bankroll: $100, configurable as `n`.
- Session length: 100 blackjack rounds, configurable as `x`.
- Minimum opening bet: $10.
- Usable aces and actions hit, stand, and double down.
- Optimize bankroll growth relative to starting bankroll, rather than raw dollars.
- Strongly penalize inability to afford the minimum bet and negative ending balance.
- Finite cards, observable card history, card counting, and periodic realistic shuffles.
- Training progress and extensive research plots of performance, decisions, betting, and variance.
- Modular implementation; numeric behavior settings belong in configuration files; ASCII hyphens in code and comments.

### Explicit assumptions and design decisions

1. **Negative balance ambiguity:** Cash below zero is impossible when wagers are funded in advance and borrowing is prohibited. Enforce that accounting rule. Interpret the additional ending-loss request as penalizing **negative net profit**, meaning final bankroll below initial bankroll. Keep this interpretation visible in the README and report metadata. Provide independently configurable terminal-shortfall and negative-cash penalty settings, but explain that the negative-cash component is unreachable in the normal environment. Do not introduce credit merely to make that penalty observable.
2. **Bankruptcy:** Settled bankroll below the minimum bet ends the session, including a residual balance such as $5. A busted hand is a different event and does not itself imply bankruptcy.
3. **Forced participation:** The agent must bet in every round it can afford. No sit-out, early cash-out, or voluntary stop-session action. Here, "stop" means stand on the current hand. This prevents solving the task by refusing to play, but it limits realistic counting strategies that avoid unfavorable shoes.
4. **Ruleset:** Use the explicit configurable rules below. "Standard blackjack" alone is not a sufficiently precise simulator specification.
5. **Bet representation:** Start with a configurable ladder of minimum-bet multiples and bankroll fractions. This keeps the action space fixed as wealth changes. It restricts the exact available wager amounts; publish the ladder in every experiment. Include an all-in option and do not silently impose a maximum bankroll or bet cap.
6. **Memory:** Supply a public card ledger and count features directly to a feed-forward policy. This gives the agent counting capability without requiring it to discover long-term memory. Learning counting from raw history is a later experiment, not a claim of the initial implementation.
7. **Risk versus money:** A ruin penalty deliberately changes the objective from pure expected money maximization to risk-penalized expected relative wealth. Report both raw wealth and penalized reward so success cannot be claimed from reward alone.

## 2. Dependencies and repository structure

Use `pyproject.toml`, a dependency lock file, a `src` layout, and a Python version supported by the selected dependency versions. Pin a mutually compatible set at implementation time and record it in run metadata; do not use floating dependency versions in published experiments.

Dependencies: `numpy`, `gymnasium`, `torch`, `stable-baselines3`, `sb3-contrib`, `tensorboard`, `tqdm`, `rich`, a validated YAML configuration library, `pandas`, `pyarrow`, `plotly`, and `matplotlib`. Development dependencies: `pytest`, `hypothesis`, and `ruff`. Use Plotly for interactive local HTML and Matplotlib for publication exports. A hosted website is unnecessary.

```text
configs/
  game.yaml                 # Rules, money, cards, shoe, bet ladder
  reward.yaml               # Utility and penalty coefficients
  train.yaml                # PPO, seeds, budget, evaluation cadence
  evaluation.yaml           # Test suite, confidence intervals, baselines
  plots.yaml                # Bins, panels, smoothing, sampling, exports
  experiments/              # Small explicit overrides for ablations
src/blackjack_rl/
  config.py                 # Typed loading, validation, resolved config
  game/
    cards.py                # Rank definitions and ace-aware hand values
    rules.py                # Dealer policy and settlement rules
    shoe.py                 # Finite shoe and shuffle lifecycle
    ledger.py               # Public exposure events and card counts
    bankroll.py             # Cash, escrow, settlements
    engine.py               # Game state machine independent of Gym/SB3
    events.py               # Typed immutable event payloads
  env/
    blackjack_env.py        # Gymnasium adapter and action_masks()
    actions.py              # Stable action IDs and bet decoding
    observations.py         # Public-information feature encoding
    rewards.py              # Wealth deltas and terminal components
  agents/
    baselines.py            # Random, fixed-bet strategy, count-based
    factory.py              # MaskablePPO construction and loading
  training/
    runner.py               # Vector environments and learning lifecycle
    callbacks.py            # Progress, diagnostics, evaluation, saves
  evaluation/
    runner.py               # Independent held-out session execution
    metrics.py              # Financial and behavioral aggregates
    probes.py               # Policy probabilities on reachable states
    statistics.py           # Intervals, paired comparisons, uncertainty
  telemetry/
    schemas.py              # Versioned run/episode/round/decision schemas
    writer.py               # Buffered tables and atomic manifests
  reporting/
    registry.py             # Register plots by name and required fields
    training.py
    bankroll.py
    decisions.py
    betting.py
    counting.py
    comparisons.py
    report.py               # HTML index and static exports
  cli.py
tests/
  unit/
  integration/
  statistical/
```

Keep responsibilities narrow. The game engine must not import Torch, SB3, plotting code, or a logging backend. Observation encoding consumes a public snapshot, not unrestricted engine state. Reward calculation consumes settlement events. Plot functions consume saved tables, not a live training process.

Avoid a generic plugin framework or custom RL algorithm initially. A small plot registry and typed component interfaces are sufficient extension points.

## 3. Configuration contract

All numeric values that control behavior must be in the appropriate configuration, including card values, payouts, betting steps, model widths, seed lists, logging intervals, histogram bins, confidence levels, and numerical tolerances. Derive dimensions and action counts from those settings. Ordinary indexing and mathematical identities need not become arbitrary settings.

Illustrative initial `game.yaml`:

```yaml
money:
  minor_units_per_dollar: 100
  initial_bankroll_minor: 10000
  minimum_bet_minor: 1000
  bet_increment_minor: 1000
session:
  rounds: 100
rules:
  blackjack_total: 21
  dealer_stand_total: 17
  dealer_hits_soft_17: false
  natural_card_count: 2
  blackjack_profit_numerator: 3
  blackjack_profit_denominator: 2
  ordinary_win_profit_multiplier: 1
  double_stake_multiplier: 2
  double_draw_count: 1
  double_on_initial_hand_only: true
  dealer_peeks_for_blackjack: true
  allow_split: false
  allow_surrender: false
  allow_insurance: false
  reveal_hole_after_player_bust: false
shoe:
  decks: 6
  suits_per_rank: 4
  ranks: [A, '2', '3', '4', '5', '6', '7', '8', '9', '10', J, Q, K]
  low_values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
  ace_high_value: 11
  penetration_fraction: 0.75
  hilo_weights: [-1, 1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1]
  true_count_denominator_floor_decks: 0.25
betting:
  minimum_bet_multiples: [1, 2, 3, 4, 5, 10, 20]
  bankroll_fractions: [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
```

Use integer minor units for all cash arithmetic. Validate payout divisibility for allowed wager increments so a natural payout is exact. Do not round money using binary floating point. Fractional bet decoding should use decimal/rational arithmetic followed by floor-to-increment.

Validate positive affordable initial bankroll, positive horizon, matching rank/value/count arrays, valid penetration, supported rule combinations, and model batch compatibility. Unsupported rule switches must fail clearly instead of pretending to implement splitting or insurance. These checks are implementation decisions to catch invalid experiments before training; they do not add game penalties.

Save the complete resolved configuration, config hash, schema versions, source commit/dirty state, package versions, hardware, start time, and seed manifest in each run directory. Reject unknown config keys to catch misspelled hyperparameters.

## 4. Blackjack engine and accounting

### Rules and settlement

- One player hand versus one dealer hand. No other seated players initially.
- Aces count as low values, then upgrade an ace if the result does not exceed the blackjack total. Return both best total and `usable_ace`.
- A natural requires the configured initial card count and blackjack total. A multi-card 21 is an ordinary hand.
- Dealer stands on soft 17 by default. Make the soft-17 variation configurable and label it in reports.
- Opening deal order: player, dealer upcard, player, dealer hidden card.
- With ace or ten-valued upcard, dealer checks for a natural before player actions. Expose the result of a negative peek as public information without revealing the hole rank.
- Dealer natural beats non-natural player hands; simultaneous naturals push; player natural otherwise pays 3:2 profit.
- Double is available on an initial nonterminal two-card hand of any total only when another opening stake can be funded. Draw exactly one card, then automatically stand or bust.
- Hit at 21 is not offered; settle or advance to dealer play automatically. Stand is otherwise always legal in the player phase.
- Player bust loses immediately. With the default visibility setting, the unused dealer hole card is discarded unseen and the dealer does not draw additional cards.
- Reveal the dealer hole when dealer play or a natural comparison makes it public. Every physical card can be added to the public ledger only once.

### Money and escrow

Track `available_cash`, `committed_stake`, and pre-round settled bankroll separately. Deduct a wager from available cash and put it into escrow. Doubling moves an additional opening stake to escrow. Settlement transfers the applicable return and clears escrow.

For opening stake $10: loss returns $0; push returns $10; normal win returns $20; natural win returns $25. For a doubled total stake of $20: loss returns $0; push returns $20; normal win returns $40. No additional natural bonus after doubling.

The reward uses **settled bankroll changes**, never the temporary cash reduction caused by placing a bet. Having no available cash while an all-in bet is pending is not bankruptcy. Check bankruptcy after settlement. Session length counts settled rounds, including natural resolutions and pushes, not policy actions.

### Phase transitions

```text
reset -> prepare shoe -> BET
BET action -> escrow -> deal and peek
  -> immediate natural settlement, or PLAYER
PLAYER hit -> PLAYER, or settlement/dealer on bust/21
PLAYER stand -> dealer -> settlement
PLAYER double -> one draw -> dealer/settlement
settlement -> increment completed rounds -> reward and terminal checks
  -> terminal, or prepare shoe -> next BET
```

Dealer play is automatic inside the triggering step. There is no agent dealer-phase action. A bet step may settle a natural and return the next bet observation or a terminal observation.

## 5. Finite shoe and legitimate card counting

Create the configured number of physical decks, shuffle with a dedicated seeded RNG, and draw without replacement. Keep the shoe across rounds within a session. Start each independent session with a fresh shuffle and reset the public ledger.

Place a deterministic cut threshold at the configured penetration. If crossed during a round, finish that round and shuffle before the next betting decision. The next bet observation must already reflect the new shoe and a reset count. Never take a bet against an old count and then silently shuffle before dealing.

Add a pre-round card-sufficiency check using a conservative bound derived from the supported hand rules and minimum card value. Shuffle early if the remaining shoe cannot safely complete a round. This is an added guard against deck exhaustion; it can reduce effective penetration, so log `shuffle_reason` and actual penetration. Do not silently reshuffle mid-hand. Unexpected exhaustion is an engine error and fails the run. Reject configurations where the chosen reserve cannot fit in a fresh shoe.

Maintain distinct information:

- Private engine: true shoe sequence, hidden cards, exact internal hands.
- Public snapshot: exposed card counts by rank, player cards, dealer upcard, public negative-peek information, dealt-card count, shoe identifier, and shuffle event.
- Unseen-card pool: initial rank counts minus exposed rank counts. It includes the dealer hole card and any previously discarded unseen cards. Never label it "exact cards remaining in the shoe."

Compute Hi-Lo running count from public exposure events only. True count is running count divided by estimated decks physically remaining, with its configured positive denominator floor. In this simulator, assume the player can track the exact number of dealt cards, including hidden ones, but not their ranks; this is a more precise depth estimate than a real visual estimate. Add coarse/noisy depth estimates as an ablation later.

Unseen discards mean rank exposure counts are useful information, but not a fully observed Markov state. Do not claim feed-forward features eliminate all partial observability. Expose whether the current dealer peek ruled out blackjack, because that changes the player's information about the hidden card.

For version one, aggregate ten-valued cards in policy features if desired while retaining original rank in audit events. Card order within the exposed history is unnecessary for the initial counting features, but retain events in sampled/full evaluation logs for future sequence models.

## 6. Gymnasium interface and policy actions

Implement `BlackjackSessionEnv(gymnasium.Env)` with `reset(seed, options)`, `step(action)`, `action_masks()`, and a lightweight text render for debugging. Use Gymnasium's reset and step signatures and run SB3's environment checker. [Gymnasium API](https://gymnasium.farama.org/api/env/), [SB3 custom environments](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html).

Use a single `Discrete` action space: configured bet entries followed by HIT, STAND, and DOUBLE. Action meaning is stable across a run. At BET, mask all play actions; at PLAYER, mask all bets and illegal play actions.

Bet decoding:

1. A multiple entry maps to that multiple of the minimum bet; mask if unaffordable.
2. A fraction entry maps to that fraction of settled bankroll, rounded down to the configured bet increment; mask if below minimum or unaffordable.
3. If entries decode to the same amount, retain the first in the documented action ordering and mask duplicates. This prevents arbitrary duplicate labels distorting betting entropy. Log the actual decoded amount as well as action ID.
4. The minimum-bet entry guarantees a legal choice for any active session. The full-bankroll fraction means all available betting increments, with any sub-increment cash left uncommitted.

Do not silently clamp an invalid action to another action. Raise a descriptive exception for illegal actions; invalid choices indicate caller/mask bugs. The SB3 checker can sample actions without consulting masks: run interface checks through a clearly labeled test-only legal-action adapter, and separately test the unwrapped environment with legal-action rollouts and explicit illegal-action assertions. Never use that adapter for training or scientific evaluation.

Use a single-level `Dict` observation for `MultiInputPolicy`:

| Group | Features |
|---|---|
| Phase | Bet/player indicators |
| Player | Total normalized by blackjack total, usable ace, initial-hand flag, player rank histogram |
| Dealer | Upcard encoding and negative-peek indicator |
| Bankroll | Settled wealth / initial wealth, available cash / initial wealth, committed stake / initial wealth, minimum bet / initial wealth |
| Horizon | Completed and remaining rounds / configured horizon |
| Shoe | Public exposed rank counts / initial counts, running and true count with configured feature scales, fraction dealt, unseen-card count |
| Bets | Decoded candidate wager amounts / initial wealth and legal-action indicators |

Bet observations contain no newly dealt hand. Use neutral values and phase indicators for absent player/dealer features. Define valid terminal observations without requiring another policy decision. Do not clip bankroll or true count to a small arbitrary range; use valid space bounds and explicit scaling. Any later clipping must be configured and documented as information loss.

The fixed horizon is part of the task, so completion at round 100 is `terminated=True`, not a bootstrap-eligible time-limit truncation. Bankruptcy is also termination. Reserve `truncated=True` for an actual external interruption of the modeled episode. Include remaining horizon in observations. Distinguish both flags from SB3 VecEnv's combined done behavior and preserve terminal info before auto-reset.

## 7. Reward objective

Let `B0` be starting bankroll and `Bt` the settled bankroll after round `t`. Define `zt = Bt / B0`.

For each settlement, pay:

```text
wealth_reward = (Bt - B_previous) / B0
```

Non-settlement steps receive zero reward. Once, at session termination, add:

```text
ruin_penalty = ruin_weight * I(final_bankroll < minimum_bet)
shortfall_penalty = shortfall_weight * max(0, 1 - final_bankroll / B0)
negative_cash_penalty = negative_cash_weight * max(0, -final_bankroll / B0)
terminal_adjustment = -ruin_penalty - shortfall_penalty - negative_cash_penalty
```

Suggested initial `reward.yaml`: `ruin_weight: 10.0`, `shortfall_weight: 2.0`, `negative_cash_weight: 10.0`, `utility: relative_linear`. The coefficients are proposed research defaults, not user-specified values. Negative cash remains an accounting failure in the default no-credit game; the configured formula documents how it would be penalized in a separately designed future credit variant, rather than hiding such a failure during training.

With undiscounted rewards, the episode sum is exactly:

```text
final_bankroll / initial_bankroll - 1 - terminal penalties
```

Thus $100 -> $150 and $1000 -> $1500 both earn +0.5 when neither incurs a penalty. Do not use a sum of per-round percentage changes, which depends on the path. Do not add bonuses for wins, survival, counting, standing, or larger wagers; those alter the objective and can create shortcuts.

Use `gamma: 1.0` initially. Episodes are finite, and discounting by decisions would otherwise penalize hands requiring more hits and change the intended wealth objective. GAE lambda can remain below one as an estimator setting; evaluate the undiscounted actual return separately.

Examples with the proposed weights:

| Final bankroll from $100 | Wealth component | Ruin | Shortfall | Total |
|---|---:|---:|---:|---:|
| $150 | +0.50 | 0 | 0 | +0.50 |
| $100 | 0 | 0 | 0 | 0 |
| $80 | -0.20 | 0 | -0.40 | -0.60 |
| $5 | -0.95 | -10 | -1.90 | -12.85 |
| $0 | -1 | -10 | -2 | -13 |

Check ruin after the final round too. Apply overlapping ruin and shortfall penalties once each and log their components separately. Stop a bankrupt session early; do not invent losses for unplayed rounds.

Relative reward is scale invariant, but the entire game is only scale equivalent if minimum bet and bet increment also scale. Starting at $1000 while keeping a $10 minimum produces a materially different risk problem. Include both fully scaled tests and fixed-minimum tests and label them separately.

No finite penalty guarantees a hard maximum bankruptcy probability when upside is unbounded. Sweep penalty coefficients and report the growth/ruin frontier. A hard risk constraint would require a separately specified constrained objective, not a claim about these defaults. Log utility may be an optional ablation, clearly labeled as a different objective.

## 8. Agent and training pipeline

Use `sb3_contrib.MaskablePPO` with `MultiInputPolicy`. It provides categorical action masking and multiprocessing support, but not recurrent policies. Use mask-aware evaluation utilities and pass masks explicitly for standalone prediction. Implement `action_masks()` inside the environment for subprocess workers. [MaskablePPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).

Do not combine `RecurrentPPO` and `MaskablePPO` by simply swapping a policy name. A masked recurrent agent would need additional algorithm work and tests. The public ledger is the initial practical solution.

Suggested configurable training defaults, subject to pilot results:

| Setting | Initial value |
|---|---:|
| Parallel environments | 8 |
| Rollout steps per environment | 2048 |
| Batch size | 512 |
| PPO epochs per update | 10 |
| Learning rate | 0.0003 |
| Gamma | 1.0 |
| GAE lambda | 0.95 |
| PPO clip range | 0.2 |
| Entropy coefficient | 0.01 |
| Value coefficient | 0.5 |
| Gradient norm limit | 0.5 |
| Policy/value hidden widths | [128, 128] each |
| Pilot budget | 1,000,000 transitions |
| Initial full-run budget | 10,000,000 transitions |
| Validation interval | 100,000 transitions |
| Checkpoint interval | 500,000 transitions |
| Sessions per validation | 1,000 |
| Independent training seeds | [11, 23, 37, 51, 71] |

These are starting settings, not an assertion that this many samples will converge. Benchmark `DummyVecEnv` versus `SubprocVecEnv`; this simulator may be too cheap for process overhead to pay off. Default to CPU for the small MLP unless measured throughput favors another device. On Windows, use import-safe top-level factories and a guarded main entry point for subprocess creation.

Training sequence:

1. Validate and freeze config; create a unique run directory.
2. Derive independent RNG streams for training workers, shoes, evaluation, baselines, and analysis. Never reseed every reset with the same seed.
3. Build vector environments, monitor session returns, and construct MaskablePPO.
4. Start `learn(..., progress_bar=True)` with custom metrics, checkpoint, and mask-aware validation callbacks.
5. Report transitions, settled rounds, completed sessions, throughput, wall time, ETA, and latest validation wealth/ruin alongside SB3 progress. Transitions and rounds are not interchangeable.
6. Save latest and best validation-reward checkpoints. Also record validation mean wealth and ruin for each checkpoint; do not quietly select using test data.
7. On clean interruption, save a recoverable checkpoint and completed telemetry chunks. Report unfinished sessions as interrupted, not bankrupt.
8. After the run, evaluate the selected checkpoint once on the held-out test manifest and build reports.

SB3 exposes TensorBoard logging and custom scalar callbacks; keep console progress and persistent metrics separate. [TensorBoard integration](https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html), [callback API](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html).

Track callback schedules in actual global transitions; vectorization changes callback frequency. Allow cadence overshoot by a vector step and record actual evaluation timesteps. Validation environments must not share training RNGs or mutable state.

Start with deterministic feature scaling and unnormalized rewards so reward components remain interpretable. If `VecNormalize` is later tested, save its state, freeze it during evaluation, and retain original financial metrics. Reward normalization is an experiment, not an invisible default.

Document that loading an SB3 checkpoint resumes learning but does not necessarily restore exact worker states, partial rollouts, and RNG positions. Exact continuation would require extra state serialization; do not promise bit-for-bit resume by default.

## 9. Telemetry and data contracts

Use versioned Parquet tables plus JSON manifests and TensorBoard scalars. Buffer writes; never synchronously write a file per step. Training stores complete episode summaries and sampled detailed traces. Evaluation stores full traces for the configured research sample, with aggregate summaries for every session. Sampling rates, flush sizes, and storage limits belong in config.

Required tables:

| Table | Essential fields |
|---|---|
| Runs | Run/config/model IDs, seeds, versions, hardware, rules, budgets |
| Episodes | Episode ID, split, seed, initial/final wealth, return components, rounds, ruin, profit, max drawdown, termination reason |
| Rounds | Episode/round/shoe IDs, pre/post bankroll, initial/total stake, outcome, natural/double/player-bust flags, pre-bet count/depth, dealer outcome, payout |
| Decisions | Phase, public feature snapshot, action ID, decoded stake, legal mask, chosen probability, entropy, model timestep |
| Card events | Physical card ID for deduplication, rank when public, deal/exposure/reveal/discard/shuffle event, visibility, round/shoe IDs |
| Evaluations | Checkpoint, training transitions, policy mode, scenario, sample size, metrics and confidence intervals |

For sampled policy probes, store the full masked action probability vector. Keep engine-only audit data separate from policy inputs and public replay data. In training, callbacks must never rebuild observations from privileged `info` fields.

Define outcome semantics: push has zero net profit; doubled wins/losses use total stake; hand bust is distinct from session ruin. Store normalized and monetary bet sizes. For ratios with zero denominators, output missing values plus eligibility counts, never fabricate a zero.

## 10. Evaluation and scientific comparisons

### Baselines

Implement all baselines through the same public observations, masks, bankroll constraints, and game engine:

1. Uniform random legal actions, labeled a weak sanity baseline.
2. Fixed minimum bet with a documented simple threshold playing policy.
3. Fixed minimum bet with a rules-matched hit/stand/double strategy table. Verify the table for the configured no-split rules; do not label an arbitrary heuristic "optimal basic strategy."
4. The same playing strategy with a configurable Hi-Lo bet spread mapped to available bets.
5. Learned play with fixed minimum bets, isolating hand strategy.
6. Learned betting with a fixed playing strategy, isolating wager selection.
7. Full joint learned policy.

Baselines must respect unaffordable doubling using an explicitly specified alternative action. All heuristic thresholds and spreads are config settings. Baseline quality matters: an incorrect strategy table can make a poor RL agent look successful.

### Protocol

- Separate training, validation, and test seed manifests. Select models on validation only.
- Initial final-test target: 10,000 sessions per trained seed and baseline; store this budget in evaluation config and increase if tail estimates remain noisy.
- Use at least the configured independent training seeds. Show variation across learned models as well as stochastic sessions within each model.
- Evaluate deterministic and sampled policies separately. Select and declare the intended deployment mode; stochastic training return is not the same as deterministic evaluation return.
- Reuse pre-generated per-session/per-shoe random streams across policies where practical. Different actions consume different cards and can shift shuffles, so matching seeds is variance reduction, not identical hand-by-hand counterfactuals.
- Bootstrap whole sessions for within-model metrics, not individual hands. Use a hierarchical bootstrap or per-training-seed summaries for uncertainty about the training method. Configure confidence level, resamples, tail thresholds, and minimum sample counts.
- Report uncertainty for ruin probabilities and low-frequency events. Zero observed ruin does not prove zero ruin probability.
- Report mean, median, quantiles, and tail outcomes together. A few exceptionally profitable sessions can dominate mean wealth.
- Keep total session profit, profit per played round, profit per scheduled round, and profit per unit wagered distinct. Early ruin makes their denominators different.

Required ablations: no count features; Hi-Lo only; full public rank counts; shuffle every round; penetration variants; ruin/shortfall weight sweeps; bet-ladder resolution; fixed versus variable betting; initial bankroll variants; soft-17 rule variants; training seed and budget. Removing counting must remove all historical count-derived inputs, including residual rank-history fields. Current player cards and dealer upcard remain visible.

For no-count training, train a separate policy rather than merely zeroing features at evaluation. Feature removal at evaluation is a distribution-shift intervention and must be labeled as such.

## 11. Research charts and reports

"All possible graphs" is open-ended. Implement the following concrete catalog and make additional charts easy to register. Every chart identifies run/checkpoint, rules, policy mode, population, axes/units, sample count, and uncertainty where applicable. Save underlying aggregates so figures are reproducible.

### Learning and optimization

- Training session reward, evaluation reward, wealth component, ruin penalty, and shortfall penalty versus transitions and wall time.
- Mean/median final bankroll multiplier and profit probability versus training progress.
- Ruin probability, completion rate, mean completed rounds, and drawdown versus progress.
- PPO policy loss, value loss, entropy, approximate KL, clip fraction, explained variance, and learning rate.
- Entropy separately for betting and playing, accounting for number of legal actions; aggregate entropy alone can hide collapse in one phase.
- Throughput, evaluation overhead, logging overhead, and resource use if collected.
- Across-seed curves with raw traces and configurable smoothing; show how smoothing was computed and retain unsmoothed exports.

### Bankroll and risk

- Final wealth histogram, ECDF, and box/violin plots by agent and seed.
- Bankroll trajectories with median and percentile bands by round.
- Carry terminal bankroll forward for bankrupt sessions in population trajectory plots; separately label survivor-only curves to avoid survivorship bias.
- Probability of surviving to each round and distribution of ruin round.
- Maximum drawdown distribution, wealth volatility, worst-tail final wealth, and expected shortfall at configured quantiles.
- Mean wealth versus ruin probability across penalties/bet policies, with uncertainty.
- Win/loss/push/natural/player-bust frequencies and payout contributions.
- Return distributions conditional on starting bankroll and remaining horizon.

### Hit, stand, and double decisions

- Policy action-probability heatmaps by player total and dealer upcard, split into soft/hard hands.
- Separate initial two-card and later-hand panels because double availability differs.
- Panels by true-count range, bankroll range, remaining rounds, and committed stake.
- State visitation heatmaps next to decision heatmaps, identifying unsupported or rare cells.
- Action frequency and double-use rate over training; double rate conditioned on double being legal.
- Disagreement maps versus the documented rules-matched baseline.
- Chosen-action entropy/confidence distributions and action transitions after hits.
- Outcome distributions conditional on action and state. Clearly label these observational comparisons: hit and stand are chosen in different situations, so unconditional win rates are not causal action values.

Do not call SB3's state value estimate a per-action Q-value. For optional action-EV maps, build a separate Monte Carlo probe evaluator that samples hidden cards consistent with public information and negative-peek constraints, branches legal actions, and follows a specified continuation policy. Privileged actual-hole counterfactuals must be labeled as oracle diagnostics and cannot drive the agent.

Generate policy-probability charts by querying sampled reachable public states. Arbitrarily stitching together incompatible player hands, counts, and shoe depths can create impossible states and misleading plots.

### Betting and variance

- Actual opening-bet distribution in dollars, starting-bankroll units, and fraction of current wealth.
- Bet mean, median, standard deviation, variance, coefficient of variation, and quantiles over training and across sessions; omit undefined coefficients.
- Bet size versus pre-bet true count, running count, penetration, current bankroll, drawdown, and rounds remaining.
- Conditional bet heatmaps: count by bankroll and count by horizon.
- Bet action probabilities/entropy, decoded wager values, minimum-bet rate, and all-in rate.
- Bet changes and lag autocorrelation within sessions, avoiding cross-session boundary artifacts.
- Opening stake versus total stake after doubling; rate at which an opening wager prevents an otherwise possible double.
- Profit and loss contributions by wager bin and count bin; report counts and both per-round and per-unit-stake returns.
- Bet-size/return covariance and correlations with confidence intervals, explicitly distinguishing association from causal benefit.
- Within-session versus between-session bet variance, and between-training-seed variation.

### Counting and shoe behavior

- Running/true-count distributions and trajectories within shoes.
- Empirical next-card frequencies versus public rank/count features, with enough samples per bin.
- Bet and action changes before/after shuffle, and count reset validation.
- Actual penetration and shuffle reasons.
- Conditional outcome/return versus count and penetration, with uncertainty.
- Count-feature ablation comparisons and contribution of positive/negative-count rounds to total profit.
- Calibration of sampled estimates of favorable card composition; avoid claiming the realized hidden shoe is known to the policy.

### Comparison and robustness

- Baseline metric tables with confidence intervals and paired session differences where justified.
- Seed distributions, learning sample efficiency, and compute cost to reach stated validation criteria.
- Risk/growth tradeoff plots over penalty coefficients and betting resolutions.
- Fully scaled-bankroll reward equivalence tests versus fixed-minimum bankroll experiments.
- Generalization to different deck counts, penetration, and soft-17 rules, labeled in-distribution or changed-rule tests.

Reports should support filtering by run, checkpoint, seed, baseline, phase, soft/hard hand, count range, and bankroll/horizon bins. Build a static interactive HTML index linking chart files, plus PNG/SVG/PDF exports and CSV/Parquet aggregate exports. Do not require a remote tracking service. Avoid embedding every full-resolution decision in HTML; configurable aggregation and sampled traces keep reports usable without changing metrics.

## 12. Verification and acceptance criteria

### Deterministic game tests

- Multiple aces, ace downgrade after a hit, soft/hard totals, natural versus multi-card 21.
- Dealer soft-17 variants, peek timing, simultaneous naturals, ties, player bust.
- Every normal/natural/double cash settlement and exact escrow conservation.
- Double only on the initial eligible hand, exactly one draw, sufficient cash, and automatic resolution.
- Bankroll exactly at minimum remains eligible; below minimum terminates; all-in escrow does not prematurely terminate.
- Natural and push rounds increment the horizon once; terminal reward fires once.
- Deck multiplicities, no replacement, cut-card timing, count reset, unseen discards, and early reserve shuffles.
- Hidden hole and future shoe permutations leave the current observation and action mask unchanged when public state is identical. Allow future transitions to differ.
- Exposure events are counted once; negative peek conveys only its allowed information.
- Bet deduplication, rounding, phase masks, and nonempty masks in every active state.

### Reward/property tests

- Sum of settlement wealth rewards equals final relative profit for generated legal trajectories.
- Exact expected penalty totals for the example cases and for early ruin.
- Scale all monetary settings and scripted outcomes together and obtain identical relative rewards.
- Cash never becomes negative, no card is duplicated, and illegal actions are rejected clearly.
- Seed replay produces identical legal-action traces under the same configuration.
- No NaN/Inf features, rewards, probabilities, or saved aggregates.

### Integration and statistical tests

- Gym/SB3 interface validation plus legal masked rollouts over complete sessions.
- Short single-process and Windows-compatible subprocess training smoke tests.
- Mask-aware deterministic and stochastic evaluation, checkpoint save/load, and report generation.
- Terminal metrics survive VecEnv auto-reset and are attributed to the correct episode.
- Independence of worker/evaluation RNG streams and fresh shuffle behavior.
- Independent reference hand enumeration or a small trusted settlement calculator for controlled states. Do not demand an unrestricted casino house-edge number from this restricted ruleset.
- Statistical checks use declared tolerances and adequate sample sizes; avoid flaky fixed-profit assertions.

A valid implementation need not beat blackjack or make a positive profit. Acceptance requires correct mechanics, objective fidelity, functioning learning, honest held-out comparisons, and reproducible artifacts. Investigate poor learning rather than changing test outcomes or quietly weakening the baseline.

## 13. Implementation order and handoff gates

1. **Configuration and engine:** Create package structure, typed configs, money/rules/shoe/ledger components, and deterministic tests. Gate: scripted sessions settle correctly without Gym or ML dependencies.
2. **Environment and rewards:** Add phase/action encoder, public observations, masks, termination, and reward accounting. Gate: long legal random rollouts satisfy invariants, hidden-state tests, and reward telescoping tests.
3. **Baselines and evaluator:** Implement verified baselines, seed manifests, session metrics, and elementary plots. Gate: reproducible baseline report exists before PPO training.
4. **SB3 training:** Add MaskablePPO, vectorization, progress, TensorBoard, evaluation, and checkpoints. Gate: short training saves/loads and produces legal actions and complete session summaries.
5. **Research telemetry and reports:** Add full plot catalog, policy probes, exports, and filters. Gate: every plot can be regenerated from saved artifacts; unavailable/rare data is explicitly marked.
6. **Pilot diagnostics:** Run the pilot, inspect mask coverage, reward scales, ruin, action entropy, betting distribution, and throughput. Fix engine/logging failures before tuning.
7. **Full experiments:** Run independent seeds, penalty/count/bet ablations, and untouched final evaluation. Gate: compare with baselines and uncertainty; retain failed and negative results.
8. **Documentation:** Record commands, rules, assumptions, reward math, hardware/budget, limitations, config overrides, and how to add a rule, feature, baseline, or plot.

Suggested CLI contract, to be implemented rather than assumed to exist:

```text
blackjack-rl validate-config --config configs/train.yaml
blackjack-rl simulate --agent basic --config configs/evaluation.yaml
blackjack-rl train --config configs/train.yaml
blackjack-rl train --config configs/train.yaml --resume runs/<run>/latest.zip
blackjack-rl evaluate --run runs/<run> --split test
blackjack-rl report --run runs/<run>
blackjack-rl compare --runs runs/<run-a> runs/<run-b>
tensorboard --logdir runs
```

Use a config composition mechanism so the training config references game, reward, evaluation, and plot defaults. CLI overrides must be typed, recorded, and included in the resolved config. Do not scatter fallback numeric defaults across modules.

## 14. Explicit added guards and restrictions

| Decision | Concrete reason | Behavioral effect or limitation |
|---|---|---|
| Funded wagers and escrow | Prevent accidental borrowing and false bankruptcy | Negative cash is unreachable; ending loss means net loss |
| Legal-action masks and rejection | Prevent illegal bets/doubles and expose integration bugs | Caller must supply legal actions; no silent correction |
| Fixed configurable bet ladder | Keep a fixed SB3 action space as bankroll changes | Not every affordable dollar wager is available |
| Duplicate bet masking | Avoid multiple labels for one economic action | Available label count varies with bankroll |
| Forced betting, no quitting | Preserve the requested fixed session task | Cannot sit out negative-count rounds |
| No split, surrender, insurance | Bound initial scope to requested actions | Results apply to this explicit restricted ruleset |
| Public-only card ledger | Prevent hidden-card and shoe-order leakage | Exact remaining rank composition is not known |
| Pre-round shoe reserve | Prevent mid-round exhaustion | Some shuffles may happen earlier than the cut threshold |
| True-count denominator floor | Avoid unstable division near an empty shoe | Count estimates saturate in that denominator regime |
| Fail-fast config and numerical validation | Prevent meaningless or corrupted experiments | Invalid runs stop instead of auto-repairing settings |
| Sampled detailed training logs | Bound disk and training overhead | Fine-grained training plots describe the sampled population |
| Separate validation/test manifests | Prevent model-selection leakage | Test data cannot be used to choose checkpoints |
| Confidence intervals and sample thresholds | Avoid overinterpreting rare events | Some chart cells remain missing or uncertain |

## 15. Technical references

- [Gymnasium Blackjack](https://gymnasium.farama.org/environments/toy_text/blackjack/): its stock game uses an infinite deck and hit/stick actions, so it is not the requested bankroll-and-shoe environment. Build a custom environment instead of wrapping it as if it already supports counting and double down.
- [Gymnasium Env API](https://gymnasium.farama.org/api/env/): reset, step, spaces, and termination interface.
- [SB3 custom environments](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html): compatible environment interfaces and checking.
- [MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html): masking, policy support, evaluation helpers, and subprocess integration.
- [SB3 TensorBoard](https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html) and [callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html): logging, training progress, evaluation hooks, and checkpoints.

These sources support library integration details. The ruleset, reward coefficients, architecture, and experimental defaults above are explicit project design proposals.
