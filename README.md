# Blackjack Bankroll Reinforcement Learning and Risk Analysis

A Python blackjack simulator comparing classical betting strategies with deep reinforcement learning, focusing on bankroll management and risk.

> Personal Project<br>
> Date: 10/2026 <br>
> [**Live Streamlit app**]()

## Project overview

Can a neural network learn how much to bet in blackjack better than a set of traditional betting strategies?

This project aimed to explore that question using a blackjack simulator, several betting strategies and a Deep Q Network (DQN). We see whether reinforcement learning could make sensible decisions about risk, rather than just learn to place bigger bets.

The important distinction is that **the AI does not decide whether to hit, stand, double or split**. Every player uses the same fixed `BasicStrategy` to play their hands. The strategies we compare only decide **how much to wager before a round starts**, including whether to sit out.

The completed experiment compares seven policies across 2,100 simulated sessions. The full statistical analysis, graphs and interpretation are in [the analysis notebook](notebooks/Analysis.ipynb).

## Blackjack engine

The simulator uses six decks, reshuffles at 75% penetration and has a dealer who stands on soft 17. A natural blackjack pays 3:2, doubling after splitting is allowed, and a player can have up to four hands after splits. There is no insurance or surrender.

[BasicStrategy](strategies/basic.py) uses the player's cards and the dealer's visible card to choose how to play each hand. A separate reference player continues playing even when the strategy being tested sits out, so cards keep moving through the shoe.

The simulator tracks the bankroll, wagers, wins and losses, including the additional stakes needed for doubles and splits. Betting strategies can only use information that would be visible before the next round. They cannot see hidden cards or the future order of the deck.

## Betting strategies

Each strategy chooses from the same set of initial wagers:

$$
\mathcal{A}=\{0,1,2,4,6,8\}
$$

A wager of zero means sitting out. Positive wagers must satisfy the table limits and cannot exceed 2% of the current bankroll. There is also an exposure check to leave enough money for potential doubles and splits.

The strategies use this same set of choices, but arrive at their decisions in different ways.

<details>
<summary><strong>Flat betting</strong></summary>

The simplest benchmark. Bet one unit whenever a bet is allowed, regardless of the cards already played or the current bankroll.

There is no prediction or attempt to find an advantage. It gives us a useful reference for what happens when someone plays every round with a fixed wager.

[Implementation](strategies/flat.py)

</details>

<details>
<summary><strong>Hi Lo card counting</strong></summary>

Hi Lo tracks which cards have been revealed. Low cards increase the count, high cards reduce it, and cards worth 7, 8 or 9 leave it unchanged.

| Cards | Count change |
| :--- | ---: |
| 2 to 6 | +1 |
| 7 to 9 | 0 |
| 10 to Ace | −1 |

The running count is adjusted for the estimated number of decks still in the shoe:

$$
\text{True count}=\frac{\text{Running count}}{\text{Estimated decks remaining}}
$$

A higher true count suggests that more high cards remain. The strategy responds by increasing its wager.

| True count | Proposed wager |
| :--- | ---: |
| Below 1 | 1 unit |
| At least 1 | 2 units |
| At least 2 | 4 units |
| At least 3 | 6 units |
| At least 4 | 8 units |

The same bankroll restrictions still apply. Hi Lo does not use a fitted prediction model, just the observed cards and this betting rule.

[Implementation](strategies/hi_lo.py)

</details>

<details>
<summary><strong>Calibrated Kelly betting</strong></summary>

Kelly betting is about choosing how much of your bankroll to risk when you think you have an advantage.

We first use a separate set of blackjack simulations to estimate the average return and variance for different true count ranges. The strategy then uses a half Kelly approximation:

$$
 f=\frac{1}{2}\max\left(0,\frac{\widehat{\mu}}{\widehat{\sigma}^{2}}\right)
$$

Here, $\widehat{\mu}$ is the estimated return per unit wagered, $\widehat{\sigma}^{2}$ is its estimated variance, and $f$ is the fraction of the bankroll to bet before applying the shared limits.

The idea is simple. A stronger estimated advantage supports a larger wager, while more uncertain returns reduce it. Using half Kelly also makes the proposed wager more conservative.

The strategy only bets when the calibration data are sufficiently reliable and the lower confidence bound for the estimated return is positive. Otherwise, it sits out.

[Implementation](strategies/calibrated_kelly.py)

</details>

<details>
<summary><strong>Model based log betting</strong></summary>

This strategy also uses the calibration simulations, but instead of estimating a wager from the mean and variance, it looks at the observed distribution of possible returns.

For each legal wager, it estimates how much that wager would change the logarithm of the bankroll on average:

$$
 w^*=\arg\max_{w\in\mathcal{A}_{\mathrm{legal}}}
 \sum_r \widehat{p}(r)\log\left(1+\frac{wr}{B}\right)
$$

Here, $B$ is the current bankroll, $r$ is a possible return per unit wagered, and $\widehat{p}(r)$ is the estimated probability of that return for the current true count range.

It chooses the legal wager with the highest estimated log growth. Sitting out is always an option, with log growth of zero. It also requires sufficient calibration evidence for a positive advantage before considering a bet.

This is the main benchmark we want the DQN to beat.

[Implementation](strategies/model_based_log.py)

</details>

<details>
<summary><strong>Deep Q Network</strong></summary>

Unlike the previous strategies, the DQN learns a betting policy from experience rather than following a fixed formula.

Before each round, it receives six pieces of public information:

1. True count
2. Running count
3. Estimated fraction of the shoe remaining
4. Current bankroll relative to the starting bankroll
5. Current bankroll drawdown
6. Fraction of the session remaining

The neural network has two hidden layers with 64 neurons each. It produces a predicted value for every possible wager, and the policy chooses the legal action with the highest value:

$$
 a^*=\arg\max_{a\in\mathcal{A}_{\mathrm{legal}}}Q_{\theta}(s,a)
$$

The DQN learns from changes in its bankroll after each opportunity. Its reward is log growth, so it is trained to consider the effect of wins and losses on its remaining wealth.

The formal selected model is **Extended DQN 3**. Its final weights are saved in [`models/dqn_primary/`](models/dqn_primary/).

[Network](rl/dqn.py) · [Environment](rl/environment.py) · [Saved policy](rl/policy.py)

</details>

<details>
<summary><strong>Exploratory DQN</strong></summary>

I also included **Original DQN 1**, an earlier trained model, to see how it compared with the selected DQN and the other strategies.

This is an exploratory comparison. It was added after examining preliminary smoke tests, and it does not replace the model chosen through the formal selection procedure.

Its weights are saved in [`models/dqn_exploratory/`](models/dqn_exploratory/).

</details>

<details>
<summary><strong>Never bet</strong></summary>

Always choose zero and sit out.

The bankroll stays at 1,000 units throughout the session. This strategy is a control, showing what happens when no capital is put at risk.

</details>

## Training and evaluation

The experiment had three main stages.

**First, we calibrate the statistical strategies.** We run separate simulations to estimate blackjack returns in different true count ranges. This used 120 calibration sessions, each containing 5,000 rounds. Another 40 sessions were reserved for validation. The calibration model was then frozen before the final comparison.

**Next, we train the DQN.** The network learned through trial and error using experience replay and a target network. Its reward for each opportunity was the change in log bankroll:

$$
 r_t=\log\left(\frac{B_{t+1}}{B_t}\right)
$$

Three original training runs used 100 episodes each, while three extended runs used 500 episodes each. Each episode contained 500 betting opportunities. We compared the six candidates on 30 separate selection sessions and selected Extended DQN 3 using mean session log growth, with drawdown as a tie breaker.

**Finally, we evaluate the frozen strategies.** Each of the seven policies completed 300 sessions with 500 betting opportunities per session, starting with 1,000 simulated units. That gave me **2,100 strategy sessions and 1,050,000 betting opportunities** to analyse.

The strategies used matched random seeds for the final comparison, although different betting decisions could cause them to encounter different cards. The final evaluation data were not used to choose or retrain the selected DQN.

The session results are kept in [`results/sessions.csv`](results/sessions.csv), with a smaller overview in [`results/summary.csv`](results/summary.csv). The notebook works directly from these saved results, so there is no need to retrain the models or replay the full experiment to inspect the analysis.

## Results and analysis

The main question was whether the selected DQN could achieve better session log growth than the model based strategy.

The table below shows the mean final bankroll and mean log growth from the 300 sessions for each strategy.

| Strategy | Mean final bankroll | Mean log growth |
| :--- | ---: | ---: |
| Hi Lo | 1,008.65 | 0.00685 |
| Never bet | 1,000.00 | 0.00000 |
| Flat betting | 998.80 | −0.00153 |
| Calibrated Kelly | 998.75 | −0.00162 |
| Model based log | 998.40 | −0.00227 |
| Exploratory DQN | 994.04 | −0.01419 |
| Selected DQN | 990.99 | −0.01971 |

**The selected DQN did not outperform the model based strategy.** Its mean session log growth was lower, and a paired bootstrap comparison gave a mean difference of approximately **−0.01745**, with a 95% confidence interval of **[−0.03509, −0.00143]**.

The betting behaviour was also interesting. The selected DQN placed a wager on about **99.93%** of opportunities, compared with about **3.45%** for the calibrated strategies. Its mean maximum drawdown was around **16.0%**, against **3.6%** for model based log betting.

Hi Lo produced the highest average log growth in this particular simulation, but that does not mean it will always perform best. These results reflect the game rules, models and sample that was tested, not guaranteed real world profits.

The result was not what I hoped for, but it was useful. It showed me why a learned strategy still needs to be compared against simple, sensible benchmarks. A more complicated model is not necessarily a better one.

**[Read the full statistical analysis in the notebook](notebooks/Analysis.ipynb)** for the graphs, risk comparisons, confidence intervals and interpretation.

## Setup and verification

The project was developed and verified with Python 3.14 on macOS. From the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

To check the saved research files and run the tests:

```bash
python -m research.verification verify
python -m pytest tests -q
```

Open [`notebooks/Analysis.ipynb`](notebooks/Analysis.ipynb) in VS Code or your preferred Jupyter environment to view or run the analysis. The notebook already contains its executed graphs and results, so it can also be read directly on GitHub.

## Repository contents

The source is organised into [`blackjack/`](blackjack/) for the game engine, [`strategies/`](strategies/) for betting rules, [`rl/`](rl/) for reinforcement learning, and [`research/`](research/) for calibration, evaluation and statistical calculations.

```text
.
├── blackjack/            # Blackjack engine and rules
├── strategies/           # Playing strategy and betting policies
├── rl/                   # DQN environment, training and inference
├── research/             # Calibration, evaluation and statistics
├── models/               # Frozen DQN and calibration models
├── results/              # Final evaluation data
├── notebooks/
│   └── Analysis.ipynb    # Statistical analysis and figures
├── tests/                # Automated tests
├── requirements.txt      # Python dependencies
└── README.md
```

## Acknowledgements

The simulator uses [PyDealer](https://github.com/Trebek/pydealer) for card and deck objects. The BasicStrategy test reference was drawn from the [Wizard of Odds blackjack strategy calculator](https://wizardofodds.com/games/blackjack/strategy/calculator/).
