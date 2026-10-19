# Methodology

BlackJackAI compares initial-wager policies in a seeded blackjack simulator.
The neural network chooses how much to bet. `BasicStrategy` chooses whether to
hit, stand, double or split for every participating player under every policy.

## Game and accounting

`blackjack/engine.py` coordinates dealing, turns and settlement. `Shoe` uses a
seeded six-deck shuffle and reshuffles between rounds at 75% penetration. The
dealer peeks for blackjack and stands on soft 17. Naturals pay 3:2; doubling
after splitting is allowed. Same-rank pairs can produce at most four hands.
Split aces receive one replacement card and cannot be resplit; split 21 is not
a natural. There is no insurance or surrender.

Money uses `Decimal`. Initial wagers are reserved from the bankroll; splits
and doubles reserve additional stakes. Each settled hand's net profit must
reconcile to the player's bankroll change. A compulsory draw that exhausts the
shoe voids the round and refunds all stakes. The engine supports multiple
players, while the research design uses a reference seat followed by an agent.

Strategies receive copied public observations. The dealer's hole card is
hidden until revealed, and policies cannot inspect future shoe order. Hi-Lo
counts visible reveal events once: 2–6 contribute +1, 7–9 contribute zero, and
ten-value cards and aces contribute −1. True count divides running count by
decks remaining as estimated from publicly seen cards. A pending reshuffle
resets the next betting count to zero.

## Research sessions and risk constraints

Each evaluation session has 500 betting opportunities and an agent bankroll of
1,000 units. The reference seat starts with 1,000,000,000 units and always bets
one unit, so the shoe advances when the agent sits out. All hands follow the
same six-deck S17/DAS BasicStrategy chart, including legal-action fallbacks.

The action grid is **sit out, 1, 2, 4, 6 or 8 units**. A positive initial wager
must fit table limits, be no more than 2% of current bankroll, and satisfy
`8 × wager < bankroll` for worst-case split/double exposure. Classical proposals
round down to a permitted wager. Abstention is never rounded up to a bet.

The round reward is `log(bankroll_after / bankroll_before)`. A refunded or
no-bet opportunity has zero reward. Rewards telescope to session log growth,
`log(final_bankroll / 1000)`. If no positive wager is legal, the evaluator holds
remaining opportunities at unchanged wealth and records functional ruin. This
is inability to place a permitted positive wager, not necessarily a zero balance.

## Calibration and independent validation

The original [study plan](../research/study_plan.json) allocates 120 calibration
sessions and 40 independent validation sessions, each with 5,000 rounds. Both
seats bet one unit with effectively unconstrained calibration bankrolls. A
round's return is total profit over all agent hands divided by the original
initial stake, including splits and doubles. Voids are recorded and excluded
from ordinary return estimates.

True-count bins have boundaries −2, −1, 0, 1, 2, 3 and 4, including the two open
tails. The model retains mean return, sample variance (`ddof=1`), empirical
outcome frequencies and uncertainty. Calibration uses 2,000 joint whole-session
bootstrap replicates, RNG seed 25001, and percentile 95% intervals. Each
replicate uses the same sampled sessions across bins. Support requires at least
500 observations, 10 contributing sessions, variance greater than 1e−12, at
least 95% bootstrap coverage and mean-interval width at most 0.10. Positive-edge
support additionally requires a strictly positive lower mean bound.

The frozen model is fitted only on calibration seeds. Validation compares
separate sessions without refitting. Metadata records plan, source and dataset
hashes. `owner_approved` is descriptive plan metadata; the validator does not
enforce approval. Calibration's count-bin approximation omits exact shoe
composition and does not establish a profitable wagering policy.

## Wager policies

| Label | Decision |
|---|---|
| `flat` | Propose one unit every opportunity. |
| `hi_lo` | Propose 1/2/4/6/8 units at true-count thresholds +1/+2/+3/+4. |
| `calibrated_kelly` | Propose half of `max(0, mean / variance)` times wealth, subject to evidence and risk caps. This is a Kelly approximation. |
| `model_based_log` | Maximise estimated one-round `E[log(1 + wager × return / bankroll)]` over legal wagers, including abstention. |
| `dqn` | Greedy frozen Extended DQN 3. |
| `dqn_runner_up` | Greedy frozen Original DQN 1, exploratory. |
| `never_bet` | Abstain at every opportunity. |

Both calibrated policies require a supported bin with a strictly positive lower
mean-return bound. Model-based log betting uses the full empirical distribution;
Kelly uses its first two moments. These are estimated, one-round decision rules,
not globally optimal finite-horizon solutions.

## DQN training and selection

The six inputs are clipped/scaled true count, running count, publicly estimated
shoe fraction remaining, relative bankroll, drawdown and fraction of opportunities
remaining. The network is **6 → 64 ReLU → 64 ReLU → 6**. Its outputs estimate
action values for the common wager grid; illegal actions are masked.

Training uses Adam at 0.001, replay capacity 10,000, batches of 64, discount
factor 1, Huber loss, gradient clipping at 10 and target updates every 250
optimiser steps. Epsilon falls from 1 to 0.05 over 10,000 environment steps.
Completed original runs used 100 episodes of 500 opportunities; extensions used
500 episodes. Initialisation seeds are 101, 202 and 303. The separate
reward-scale-100 experiment was exploratory and excluded from formal selection.
Its implementation and mathematical tests remain; its generated models/logs
are preserved in the external archive.

Six unscaled original/extended candidates were compared on 30 reserved selection
sessions. The selection rule maximises mean session log growth, with ties within
1e−12 resolved by smaller mean maximum drawdown and then fixed candidate order.
Extended DQN 3 won and remains the primary DQN. The immutable
[selection record](../models/selection.json) identifies historical paths and
hashes. Selected checkpoint tensors and their matching settings were relocated
byte-for-byte; the [asset mapping](../research/evidence/asset_mapping.json)
records their current locations.

## Confirmatory design and exploratory amendment

The reconstruction increased the earlier 40/20 calibration/validation session
counts to 120/40 while retaining the original statistical thresholds. The later
500-episode training extension and post-smoke exploratory comparator are
separate amendments, not changes to the original selected-model criterion.

The original six-policy design comprises Flat, Hi-Lo, Calibrated Kelly,
Model-based log, selected DQN and Never-bet. Its primary comparison is selected
DQN minus model-based log mean session log growth. Original DQN 1 was added
as a seventh exploratory comparator after reviewing three non-final smoke
sessions. It must not be presented as independently selected on an untouched
sample, and it does not replace the selected model.

The completed final experiment used 300 independent seed-paired sessions per
policy: 2,100 strategy-sessions and 1,050,000 opportunities. Calibration,
validation, each training run, selection and evaluation have disjoint seed roles
within each frozen plan. [The extended plan](../research/study_plan_dqn_extended.json)
increases training allocations while retaining the other roles. Common seeds
pair experimental sessions, but participation changes card consumption, so
policies need not encounter identical realised hands.

The final data are not used to retune models or selection. Source snapshots,
original manifests and numerical outputs remain identifiable by their original
checksums. No new final evaluation or model training was performed for the
publication analysis.

## Statistical analysis

For each final seed, the primary difference is DQN log growth minus model-based
log growth. The estimator averages these 300 paired differences. The recorded
protocol specifies **2,000 whole-session paired bootstrap resamples**. It does
not specify a final-analysis RNG seed, percentile interpolation convention or
practical-effect threshold. The analysis fixes `random.Random(25002)`, linear
percentile interpolation and a two-sided 95% percentile interval as disclosed
analysis choices. Seed 25002 is not a simulation seed or a prespecified decision
rule. No practical threshold is invented after seeing the results.

One resampled seed-index vector is shared across all seven policies and metrics.
Within-session opportunities are never treated as independent bootstrap units.
Individual-policy intervals and secondary pairwise comparisons are pointwise
and exploratory, without multiplicity adjustment. An interval containing zero
does not establish equivalence.

Risk summaries include session log-growth variance/standard deviation, drawdown
distributions, the 5th percentile of session log growth and the mean of the worst
5% of sessions. Bootstrap intervals quantify their sampling uncertainty.
Participation is total bets divided by opportunities; conditional wager means
weight each session by its bet count. A no-bet policy's conditional wager is
undefined. Positive-profit frequencies use Wilson 95% intervals. Zero observed
functional-ruin events use an exact two-sided binomial 95% bound, rather than
asserting that the underlying probability is zero.

The executed [analysis notebook](../notebooks/BlackJackAI_Analysis.ipynb) derives
the results from the retained session records and creates its five Matplotlib
figures directly in code. [research/analysis.py](../research/analysis.py) is the
shared, tested statistical implementation. No saved statistics cache or external
raw-round dataset is needed. Installation, notebook execution, verification and
archive recovery are in [reproducibility.md](reproducibility.md).
