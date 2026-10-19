# Factual notes for the README author

`README.md` is reserved for the owner. These notes collect the implementation
facts and a few results needed to explain the project. The executed
[analysis notebook](../notebooks/BlackJackAI_Analysis.ipynb) contains the full
statistical analysis; the README should introduce it and link to it.

## Motivation and research question

Can a Deep Q-Network learn blackjack wager sizing that outperforms classical
betting rules and a calibrated expected-log policy? Blackjack provides a
controlled setting with uncertain returns, a changing public information set
and decisions about risk exposure. The experiment separates playing a hand from
choosing how much capital to put at risk.

**BasicStrategy plays every hand; the DQN chooses only initial wagers.**
There is no learned hit/stand/split policy. The completed experiment found lower
mean session log growth for the selected DQN than for model-based expected-log
betting under this particular simulator and training design.

## Engine and information boundary

- [Engine](../blackjack/engine.py): American blackjack, dealer peek, dealer
  stands on soft 17, 3:2 naturals, doubling after splitting, at most four split
  hands. Split aces receive one card and cannot be resplit; split 21 is not a
  natural. No insurance or surrender.
- [Rules](../blackjack/rules.py) and [shoe](../blackjack/shoe.py): six decks,
  reshuffle between rounds at 75% penetration, seeded shuffles. Table limits are
  1–1,000 units; the study's shared wager grid and risk limits are tighter.
- Monetary accounting uses `Decimal`. Initial stakes, split/double exposure,
  settlement and bankroll changes reconcile. A draw that exhausts the shoe voids
  the round and refunds all stakes.
- Policies receive copied public observations. The dealer hole card is hidden
  until revealed; future shoe order is unavailable.
- A reference seat always bets one unit with a very large bankroll. The agent
  can sit out while the reference keeps the shoe moving. Both seats use
  [BasicStrategy](../strategies/basic.py), with legal-action fallbacks.

## Betting policies and mathematics

For a wager \(w\), bankroll \(B\) and realised net return per initial unit \(R\),
wealth becomes \(B+wR\). Splits and doubles are included in \(R\).
All seven evaluated policies share the action grid \(\{0,1,2,4,6,8\}\), with zero
meaning abstention. A positive initial wager must fit table limits, be at most
2% of current wealth and satisfy \(8w<B\). Classical proposals round down to a
legal grid wager; an abstention is never rounded up.

| Policy | Method and source |
| --- | --- |
| Flat | One unit every legal opportunity; [flat.py](../strategies/flat.py). |
| Hi-Lo | Running count: 2–6 = +1, 7–9 = 0, tens/aces = −1. True count divides by publicly estimated decks remaining. Wager ramp 1/2/4/6/8 at true-count thresholds +1/+2/+3/+4; [hi_lo.py](../strategies/hi_lo.py). |
| Calibrated Kelly | Half-Kelly approximation \(f=\tfrac12\max(0,\hat\mu/\hat\sigma^2)\), then risk caps and grid rounding; [calibrated_kelly.py](../strategies/calibrated_kelly.py). |
| Model-based expected-log | Choose the legal wager maximising \(\sum_r\hat p(r)\log(1+wr/B)\), including abstention; [model_based_log.py](../strategies/model_based_log.py). |
| Selected DQN | Greedy frozen Extended DQN 3; [policy.py](../rl/policy.py). |
| Exploratory runner-up DQN | Greedy frozen Original DQN 1; included after smoke review. |
| Never-bet control | Abstain; wealth remains 1,000 units. |

The calibrated policies use frozen true-count-bin estimates and require both
adequate bin support and a strictly positive lower confidence bound on mean
return. Kelly uses the first two moments; model-based log uses empirical outcome
frequencies. These are estimates under a fixed-unit calibration population,
not guarantees of an edge in subsequent play. Calibration and independent
validation are implemented in [research/](../research/).

## DQN implementation, training and selection

[environment.py](../rl/environment.py) builds six public inputs: clipped/scaled
true count, running count, estimated shoe fraction remaining, relative bankroll,
drawdown and fraction of the session remaining. Despite the saved feature name
`decks_remaining`, that input is a fraction of the six-deck shoe.

[dqn.py](../rl/dqn.py) implements a **6 → 64 ReLU → 64 ReLU → 6** network, replay
buffer, epsilon-greedy action choice, legal-action masks and target network.
Its one-opportunity reward is

\[
r_t=\log(B_{t+1}/B_t),\qquad
\sum_t r_t=\log(B_T/B_0).
\]

Abstention and refunded rounds have zero reward. With discount factor 1, the
learning target is \(y=r+\max_{a'\text{ legal}}Q_{\mathrm{target}}(s',a')\)
for nonterminal transitions, and \(y=r\) at termination.
[train.py](../rl/train.py) uses Adam (0.001), Huber loss, batch size 64, replay
capacity 10,000, gradient clipping at 10 and target updates every 250 optimiser
steps. Epsilon declines from 1 to 0.05 over 10,000 opportunities.

Three original runs used 100 episodes; three extensions used 500 episodes,
with 500 opportunities per episode. Network seeds are 101, 202 and 303. The
reward-scale-100 trials were exploratory and excluded from formal selection.
[select.py](../rl/select.py) compared the six unscaled candidates on 30 separate
selection sessions, maximising mean log growth, then smaller mean drawdown
within a 1e−12 tie, then fixed candidate order. **Extended DQN 3 remains the
formal selected model.** Original DQN 1 is an exploratory comparator, not a
replacement selected on the final data.

## Experiment and concise results

The original and extended [study plans](../research/study_plan.json) retain
separate seed roles for calibration, validation, training, selection and final
evaluation. The completed final experiment used 300 sessions per policy, 500
opportunities per session, a 1,000-unit agent bankroll and two seats: **2,100
strategy-sessions and 1,050,000 opportunities**. Common seeds pair sessions;
different participation can change card consumption and realised hands.

The original confirmatory design had six policies. Adding Original DQN 1 after
smoke-test review made the final comparison seven policies; its analysis remains
exploratory. No final seeds were used for selection. The final sample is now
observed and must not be reused to optimise another model.

For \(G_i=\log(B_{T,i}/1000)\), the primary statistic is the mean paired
\(G_{\mathrm{DQN},i}-G_{\mathrm{Model},i}\). Recomputing from saved sessions gives
**−0.017444627584**, with paired whole-session bootstrap 95% CI
**[−0.035090337022, −0.001425544573]**. The interval lies below zero. There was
no prespecified practical-effect threshold, and this is not a claim about
real-world casino profitability.

The selected DQN bet on 99.9253% of opportunities, versus 3.4473% for each
calibrated policy; its mean maximum drawdown was 16.0223%, versus 3.6054% for
model-based log. These are observed behavioural/risk differences, not proof of
why the DQN underperformed. No functional ruin was observed for any policy;
0/300 still gives a two-sided exact 95% upper probability bound of 1.2221%.
The notebook explains uncertainty, arithmetic wealth versus log growth and the
remaining policy comparisons.

## Installation and inspection

From the repository root, on a compatible Python installation:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -B -m research.verification verify
python -B -m research.analysis
python -B -m research.notebook
python -B -m research.notebook --execute
python -B -m pytest tests -q -p no:cacheprovider
```

The tested lock records Python 3.14.7 on macOS arm64; it is not a universal lock.
`requirements.txt` lists direct dependencies; `requirements-lock.txt` pins the
resolved tested environment. The notebook runner uses `nbclient` and the current
Python kernel. Its execute command refreshes notebook outputs from saved data;
it does not generate blackjack sessions. See [reproducibility.md](reproducibility.md).

## Layout, strengths and limitations

```text
blackjack/   engine, cards, hands, shoe, rules and observations
strategies/ fixed playing policy and traditional/statistical wager policies
rl/         environment, DQN, training, inference and model selection
research/   tested statistics, verification, original methods/plans and compact evidence
models/     two frozen DQNs/settings, calibration model and selection record
results/    original sessions.csv, summary.csv and experiment_metadata.json
notebooks/  executed statistical analysis with five Matplotlib figures
tests/      engine, policy, numerical, integrity and notebook checks
docs/       temporary factual material for README writing
```

Useful strengths to explain: explicit accounting and public-information checks,
shared risk constraints, isolated seed roles, a frozen primary model, paired
session inference and unchanged historical hashes. The notebook is the evidence
for results, rather than a training-curve success claim.

Limitations include a single game configuration and finite session horizon,
a count-bin approximation rather than exact shoe composition, limited training
and candidate search, an exploratory runner-up, pointwise secondary intervals
and simulation-to-casino differences. Frequent wagers and larger exposure are
possible explanations for poor DQN results; the experiment did not isolate
causes. Hybrid policies or richer states would require a new design and fresh
evaluation data; they are future work, not completed features.

Large raw data, unselected models, training logs and removed reporting files
remain in separately verified local archives. They are not needed to reproduce
the session analysis, and no public archive download is configured. The compact
[asset mapping](../research/evidence/asset_mapping.json) preserves historical
identities and checksums.

Before publication, resolve the fixture redistribution question and review
PyDealer compatibility as described in [attribution.md](attribution.md).
When the owner later removes `docs/`, transfer necessary attribution and notices
to a suitable retained location. No project licence has been selected.
