#!/usr/bin/env python3
"""A reservoir, built from nothing but numpy, with every design choice
explained inline. Task: recall the input from a few steps ago (the
simplest possible test that the reservoir has USABLE memory) -- if this
doesn't work, nothing built on top of it will either.
"""
import numpy as np
rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 1. THE COUPLING MATRIX -- this is "the reservoir" itself: N nodes, each
#    one's next state depends on a weighted sum of every other node's
#    current state. W[i, j] = how much node j pushes on node i.
# ---------------------------------------------------------------------------
N = 100                       # number of reservoir nodes -- start modest
sparsity = 0.1                # fraction of possible connections that exist

W = rng.uniform(-1, 1, size=(N, N))
mask = rng.uniform(0, 1, size=(N, N)) < sparsity
W = W * mask                  # most nodes are NOT connected -- this matters:
                               # dense coupling makes every node's state a
                               # blurred average of every other node, which
                               # destroys the diversity you need for a
                               # readout to tell different pasts apart.

# ---------------------------------------------------------------------------
# 2. SPECTRAL RADIUS RESCALING -- the single most important step, and the
#    one people skip and then wonder why their reservoir is useless.
#    rho(W) = largest |eigenvalue| of W. It controls whether repeatedly
#    applying W (a) shrinks any perturbation toward zero (rho<1, the
#    reservoir "forgets," which is what you want -- a fading trace of
#    recent inputs, not permanent blow-up) or (b) amplifies it without
#    bound (rho>1, unstable, the state saturates into a fixed corner and
#    stops reacting to new input at all). This is the "echo state
#    property": the CURRENT state should be a function of RECENT inputs
#    only, echoing out old ones. Target rho is a tunable hyperparameter,
#    usually 0.8-0.99 -- close to 1 for long memory, safely under 1 for
#    guaranteed stability once tanh's own saturation is added on top.
# ---------------------------------------------------------------------------
target_rho = 0.9
current_rho = np.max(np.abs(np.linalg.eigvals(W)))
W = W * (target_rho / current_rho)
print(f"[build] N={N}, sparsity={sparsity}, rescaled spectral radius: "
      f"{np.max(np.abs(np.linalg.eigvals(W))):.3f} (target {target_rho})")

# ---------------------------------------------------------------------------
# 3. INPUT WEIGHTS -- how the outside world enters the reservoir. Also
#    random and fixed (never trained). input_scaling controls the balance
#    between "driven by the input" and "running its own internal
#    dynamics" -- too small and the reservoir barely notices new input,
#    too large and it ignores its own recurrent structure and just
#    tracks the input directly (no useful nonlinear mixing).
# ---------------------------------------------------------------------------
input_scaling = 0.5
Win = rng.uniform(-1, 1, size=N) * input_scaling

# ---------------------------------------------------------------------------
# 4. THE UPDATE RULE -- run the reservoir forward. x[t] depends on x[t-1]
#    (via W, the coupling) and u[t] (via Win, the input), squashed
#    through tanh (the nonlinearity -- without this, a reservoir is just
#    a linear filter and can't do anything a straight regression on the
#    raw input couldn't already do).
# ---------------------------------------------------------------------------
def run_reservoir(W, Win, u):
    T = len(u)
    X = np.zeros((T, N))
    x = np.zeros(N)
    for t in range(T):
        x = np.tanh(W @ x + Win * u[t])
        X[t] = x
    return X

# ---------------------------------------------------------------------------
# 5. THE TASK -- delay-recall: y[t] = u[t - DELAY]. Trivial for a system
#    with memory, impossible for one without -- the simplest honest test
#    of whether step 1-4 actually built something with usable memory.
# ---------------------------------------------------------------------------
DELAY = 5
T = 2000
u = rng.standard_normal(T)
y = np.zeros(T)
y[DELAY:] = u[:-DELAY]

X = run_reservoir(W, Win, u)

# ---------------------------------------------------------------------------
# 6. TRAIN THE READOUT -- the ONLY part of a reservoir computer that
#    learns. Everything above (W, Win) is fixed and random. Just linear
#    regression (ridge, for numerical stability) from reservoir state to
#    target. This is the entire point of the reservoir-computing idea:
#    all the hard nonlinear-memory work happens for free in a randomly
#    wired dynamical system; training cost is one linear solve.
# ---------------------------------------------------------------------------
BURN = 100                    # discard early states -- they depend on the
                               # arbitrary x[0]=0 start, not yet "warmed up"
split = T - 400
X_train, y_train = X[BURN:split], y[BURN:split]
X_test,  y_test  = X[split:],     y[split:]

lam = 1e-6
Wout = np.linalg.solve(X_train.T @ X_train + lam * np.eye(N), X_train.T @ y_train)

pred = X_test @ Wout
nmse = np.mean((pred - y_test) ** 2) / np.var(y_test)
print(f"[test]  delay-{DELAY} recall NMSE = {nmse:.4f}  (0 = perfect, 1 = no better than guessing the mean)")

# ---------------------------------------------------------------------------
# 7. SANITY CHECK -- would a system with NO memory (e.g. bad coupling, or
#    rho too low) still look like it worked? Compare against a reservoir
#    with rho forced near 0 (no memory at all) to prove the delay task is
#    actually discriminating, not vacuously easy.
# ---------------------------------------------------------------------------
W_nomem = W * 0.02 / target_rho   # crush the recurrent coupling to near-zero
X_nomem = run_reservoir(W_nomem, Win, u)
Wout_nomem = np.linalg.solve(X_nomem[BURN:split].T @ X_nomem[BURN:split] + lam * np.eye(N),
                              X_nomem[BURN:split].T @ y_train)
nmse_nomem = np.mean((X_nomem[split:] @ Wout_nomem - y_test) ** 2) / np.var(y_test)
print(f"[control] same task, coupling crushed toward zero (no memory): NMSE = {nmse_nomem:.4f} "
      f"(should be near 1.0 -- proves the real reservoir's low NMSE comes from actual memory, not a fluke)")
