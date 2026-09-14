"""
Optional cross-check: the same reservoir step expressed as an Amazon Braket
circuit, run on Braket's own local simulator (no AWS account or credentials
needed -- `LocalSimulator` runs entirely on this machine via the Braket SDK).

This is the Braket-native sibling of `qrc_pennylane.py`: that file already
confirms the fast NumPy density-matrix simulator in `qrc.py` agrees with
PennyLane's `default.qubit` on a single reservoir step; this file confirms
the SAME circuit also reproduces the SAME numbers on Braket's SDK/simulator
stack -- the piece that makes "quantum ... using Amazon Braket" literally
true for this project, not just "portable in principle."

To later point this at a real AWS on-demand simulator or QPU instead of the
local one, swap `LocalSimulator()` for
`AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")` (needs
an AWS account with Braket enabled) -- the circuit-building code is
unchanged either way.

Run:  python -m src.qrc_braket
"""
from __future__ import annotations

import numpy as np

I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)


def _numpy_single_step(cfg_qrc, seed, feats):
    from .qrc import QuantumReservoir
    res = QuantumReservoir({**cfg_qrc, "virtual_nodes": 1, "reupload_rounds": 1,
                            "input_qubits": len(feats)}, seed)
    rho = res._init_state()
    rho = res._reset_inputs(rho, feats)
    rho = res._evolve(rho)
    z, _ = res._measure(rho)
    return z


def _braket_single_step(cfg_qrc, seed, feats):
    from braket.circuits import Circuit, Observable
    from braket.devices import LocalSimulator

    N = int(cfg_qrc["n_qubits"])
    n_in = len(feats)
    dt = float(cfg_qrc["dt"])
    rng = np.random.default_rng(seed)
    J = float(cfg_qrc["J"])
    h = float(cfg_qrc["h"])
    Jij = rng.uniform(-J / 2, J / 2, size=(N, N))

    def op_on(pauli, q):
        mats = [I2] * N
        mats[q] = pauli
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        H += h * op_on(PZ, i)
        for j in range(i + 1, N):
            H += Jij[i, j] * (op_on(PX, i) @ op_on(PX, j))
    # exact matrix exponential via diagonalisation -- same as qrc.py / the
    # PennyLane cross-check, so this is a like-for-like numerical comparison,
    # not a Trotterised approximation of the same physical model.
    E, W = np.linalg.eigh(H)
    U = (W * np.exp(-1j * E * dt)) @ W.conj().T

    circ = Circuit()
    for q in range(n_in):
        theta = 2 * np.arcsin(np.sqrt(np.clip(feats[q], 0, 1)))
        circ.ry(q, theta)
    circ.unitary(matrix=U, targets=list(range(N)))
    for i in range(N):
        circ.expectation(Observable.Z(), target=i)

    device = LocalSimulator()
    result = device.run(circ, shots=0).result()
    return np.array(result.values)


def verify(cfg_qrc=None, seed=7):
    cfg_qrc = cfg_qrc or dict(n_qubits=4, input_qubits=2, reupload_rounds=1,
                              virtual_nodes=1, dt=1.0, J=1.0, h=0.5,
                              observables=["single_z", "zz"], shots=None,
                              spatial_multiplex=1)
    feats = np.array([0.3, 0.8, 0.55, 0.1][: cfg_qrc["input_qubits"]])
    a = _numpy_single_step(cfg_qrc, seed, feats)
    b = _braket_single_step(cfg_qrc, seed, feats)
    err = float(np.max(np.abs(a - b)))
    print("numpy         <Z_i> =", np.round(a, 5))
    print("amazon braket <Z_i> =", np.round(b, 5))
    print("max abs diff        =", err, "->", "OK" if err < 1e-6 else "MISMATCH")
    return err


if __name__ == "__main__":
    verify()
