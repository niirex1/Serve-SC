"""Central configuration for SERVE-SC.

The five vulnerability classes and the five exploit-precondition axes match the
paper (Section IV). The matrix ``A`` is the fixed, parameter-free class-to-
precondition map used by the exploitability gate in Eq. (2). It is a design
specification, not a learned object, so changing it changes the gate's meaning.
"""
from __future__ import annotations

# Stage-1 vulnerability classes (order is fixed and used everywhere).
CLASSES = [
    "reentrancy",
    "access_control",
    "unchecked_calls",
    "oracle_misuse",
    "logic_inconsistency",
]
K = len(CLASSES)

# Exploit-precondition axes (phi), order fixed.
PRECONDITIONS = [
    "public_reachability",      # j0: public function reachable without an access guard
    "oracle_exposure",          # j1: oracle call on a vulnerable execution path
    "cross_contract_trigger",   # j2: triggerable from an untrusted external caller
    "flashloan_feasible",       # j3: single-tx borrow-and-return feasible
    "tx_order_sensitive",       # j4: front-running / ordering sensitivity
]
P = len(PRECONDITIONS)

# Fixed class-to-precondition map A in {0,1}^{K x P} (Eq. 2).
# pi_k(phi) = max_j A[k, j] * phi[j]. A class contributes to exploitability only
# when at least one of its enabling preconditions is present.
A = [
    # reach oracle xtrig flash order
    [0,    0,     1,    0,    0],   # reentrancy        <- cross-contract triggerability
    [1,    0,     0,    0,    0],   # access_control    <- public reachability
    [0,    0,     1,    0,    0],   # unchecked_calls   <- cross-contract triggerability
    [0,    1,     0,    0,    0],   # oracle_misuse     <- oracle exposure
    [0,    0,     0,    1,    1],   # logic_inconsist.  <- flash-loan OR order sensitivity
]

# Stage-1 detection model.
NODE_FEAT_DIM = 128      # opcode n-gram / AST node feature width
HGT_HIDDEN = 256         # graph representation width h_c
HGT_LAYERS = 2
HGT_HEADS = 4
HGT_DROPOUT = 0.1
DENSE_CUE_DIM = 12       # m_c
NODE_TYPES = ["func", "state", "extcall", "oracle"]
EDGE_TYPES = ["calls", "reads", "writes", "ctrl_dep",
              "data_dep", "oracle_dep", "tx_interact"]

# Optimisation (Stage-1).
LR = 3e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
MAX_EPOCHS = 80
PATIENCE = 10

# Stage-2 risk model (Eq. 3).
RISK_L2_LAMBDA = 0.01
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_CI = 0.90      # central interval reported in the coefficient table

# Reproducibility.
SEEDS = [0, 1, 2, 3, 4]
DEFAULT_SEED = 0

# Risk-model design-matrix column names (intercept first), Eq. (3).
RISK_TERMS = ["intercept", "u", "e", "chi", "q", "p", "u*e", "u*chi"]
