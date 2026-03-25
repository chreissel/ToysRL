# ToysRL — RL Noise Cancellation

A reinforcement learning environment for active noise cancellation, formulated as a **closed-loop control problem**. A PPO agent with an LSTM policy learns to track and subtract a nonlinear, time-varying coupling signal from a noisy main channel — benchmarked against classical adaptive filters (LMS, IIR).

---

## Problem Formulation

### Closed-loop block diagram

```
w_t ──→ [ Plant P(z) ] ──→ y_t = f(w_t, t) + n_t
  │                               │
  │                        ─── (+) ──→ e_t = y_t − a_t  (error / residual)
  │                       │
  └──→ [ Controller C ] ──→ a_t
        (RL policy π)
```

| Symbol | Meaning |
|--------|---------|
| `w_t` | Witness signal (reference input, 1 Hz sine + noise) |
| `f(w_t, t)` | Time-varying nonlinear coupling: `A(t)·w + B(t)·w² + C(t)·w³` |
| `n_t` | Sensor noise (Gaussian, unpredictable) |
| `y_t` | Main channel (observed: coupling + noise) |
| `a_t` | Agent's action — estimated coupling to subtract |
| `e_t = y_t − a_t` | Residual after subtraction (the "error signal") |

**Objective:** minimise `E[e_t²]` — drive the residual to the sensor noise floor.

### Why closed-loop?

The observation fed to the agent is:

```
obs_t = [ witness[t-W+1 .. t],  residual[t-W .. t-1] ]
          ↑ reference input       ↑ past error signal (feedback)
```

Feeding back past residuals `e_{t-W..t-1}` closes the loop: the agent can observe whether its previous actions over- or under-corrected and adapt accordingly — analogous to the FxLMS algorithm. Without this, the agent would be purely feedforward and unable to correct for model mismatch or drift.

### Coupling model

The coupling coefficients oscillate slowly with incommensurate periods, making the system continuously time-varying:

```
A(t) = 2.0 + 1.0·sin(2π t / 30)   # linear term
B(t) = 0.5 + 0.3·sin(2π t / 47)   # quadratic term
C(t) = 0.2 + 0.1·sin(2π t / 61)   # cubic term
```

---

## Repository Structure

```
ToysRL/
├── noise_removal/
│   ├── environment.py     # Gymnasium environment (closed-loop formulation)
│   └── signals.py         # Signal generator: witness, coupling, sensor noise
├── baselines/
│   ├── lms_filter.py      # FIR LMS adaptive filter (open-loop baseline)
│   └── iir_filter.py      # IIR adaptive filter (closed-loop baseline)
├── train.py               # RecurrentPPO training script
├── evaluate.py            # Evaluation and comparison plots
└── requirements.txt
```

### Key files

**`noise_removal/environment.py`** — Gymnasium-compatible environment.
- Observation: `Box(2·W,)` — witness window + residual window
- Action: `Box(1,)` in `[-15, +15]` — scalar coupling estimate
- Reward: `y_t² − e_t²` — improvement in instantaneous squared amplitude. Maximised in expectation when `a_t = f(w_t, t)` (exact coupling subtraction).

**`baselines/lms_filter.py`** — Standard Widrow-Hoff LMS filter. Adapts online using only the witness signal (feedforward, no error feedback).

**`baselines/iir_filter.py`** — Equation-error IIR LMS filter. Uses both the witness (feedforward) and past residuals (feedback), mirroring the RL observation structure exactly:
```
â_t = b^T · witness[t-M..t]  +  a^T · residual[t-N..t-1]
```

**`train.py`** — Trains a `RecurrentPPO` agent with `MlpLstmPolicy`. The LSTM hidden state allows the agent to adapt its behaviour within an episode, acting like an online adaptive filter rather than a fixed open-loop mapping.

**`evaluate.py`** — Runs all methods on the same held-out episode and produces comparison plots.

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** `gymnasium`, `stable-baselines3`, `sb3-contrib` (for RecurrentPPO), `numpy`, `matplotlib`, `scipy`.

---

## Usage

### Train

```bash
python train.py                          # 2M steps, default settings
python train.py --timesteps 5_000_000   # longer run
python train.py --save-path models/my_model
```

Saves model to `models/ppo_noise_cancellation.zip` and VecNormalize stats to `models/ppo_noise_cancellation_vecnorm.pkl`.

### Evaluate

```bash
python evaluate.py                   # compare all methods
python evaluate.py --no-model        # baselines only (no RL model needed)
python evaluate.py --duration 120    # longer evaluation episode
```

Saves plots to `results/noise_cancellation_overview.png`.

---

## Results

Performance measured as RMS of the output signal, normalised to the oracle (sensor noise floor — the best achievable by any causal filter).

| Method | RMS | vs Oracle |
|--------|-----|-----------|
| Raw main channel | 1.68 | 5.6× |
| LMS filter (FIR) | 0.54 | 1.8× |
| IIR adaptive filter | 0.45 | 1.5× |
| RL agent (RecurrentPPO) | TBD | TBD |
| **Oracle** (sensor noise floor) | 0.30 | **1.0×** |

The **oracle** subtracts the exact coupling `f(w_t, t)` at every step, leaving only unpredictable sensor noise `n_t`. It is not achievable in practice but defines the performance ceiling.

The IIR filter outperforms LMS by exploiting residual feedback — the same closed-loop structure used by the RL agent. The RecurrentPPO agent (LSTM policy) is designed to close the remaining gap by learning a nonlinear adaptive strategy that classical linear filters cannot represent.

---

## Why RL?

Classical adaptive filters (LMS, IIR) are well-suited to smooth, slowly-varying polynomial couplings. RL with a recurrent policy becomes advantageous when:

1. **Sudden regime changes** — coupling jumps discontinuously between modes; the LSTM can detect the transition and re-adapt faster than gradient descent.
2. **High-order nonlinearity** — couplings beyond cubic that require many filter taps to approximate linearly.
3. **Multi-source coupling with cross-terms** — e.g. `f(w1, w2) = w1·w2`, which is not separable by additive adaptive filters.
4. **Hysteresis / state-dependent coupling** — coupling that depends on signal direction or recent history, which the LSTM hidden state can encode.
