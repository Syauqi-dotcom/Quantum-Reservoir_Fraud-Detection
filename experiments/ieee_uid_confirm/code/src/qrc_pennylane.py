"""
Optional cross-check: the same reservoir step expressed as a PennyLane circuit
(gate-level, Amazon Braket-portable).  Used only to verify that the fast NumPy
density-matrix simulator in qrc.py agrees with a standard circuit simulator on
a single step.  Not on the critical path.

Run:  python -m src.qrc_pennylane
"""
from __future__ import annotations

import numpy as np


def _numpy_single_step(cfg_qrc, seed, feats):
    from .qrc import QuantumReservoir
    res = QuantumReservoir({**cfg_qrc, "virtual_nodes": 1, "reupload_rounds": 1,
                            "input_qubits": len(feats)}, seed)
    rho = res._init_state()
    rho = res._reset_inputs(rho, feats)
    rho = res._evolve(rho)
    z, _ = res._measure(rho)
    return z


def _pennylane_single_step(cfg_qrc, seed, feats):
    import pennylane as qml

    N = int(cfg_qrc["n_qubits"])
    n_in = len(feats)
    dt = float(cfg_qrc["dt"])
    rng = np.random.default_rng(seed)
    J = float(cfg_qrc["J"])
    h = float(cfg_qrc["h"])
    Jij = rng.uniform(-J / 2, J / 2, size=(N, N))

    coeffs, ops = [], []
    for i in range(N):
        coeffs.append(h); ops.append(qml.PauliZ(i))
        for j in range(i + 1, N):
            coeffs.append(Jij[i, j]); ops.append(qml.PauliX(i) @ qml.PauliX(j))
    H = qml.Hamiltonian(coeffs, ops)

    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev)
    def circuit():
        for q in range(n_in):
            theta = 2 * np.arcsin(np.sqrt(np.clip(feats[q], 0, 1)))
            qml.RY(theta, wires=q)
        qml.evolve(H, coeff=dt)                        # exact matrix exponential
        return [qml.expval(qml.PauliZ(i)) for i in range(N)]

    return np.array(circuit())


def verify(cfg_qrc=None, seed=7):
    cfg_qrc = cfg_qrc or dict(n_qubits=4, input_qubits=2, reupload_rounds=1,
                              virtual_nodes=1, dt=1.0, J=1.0, h=0.5,
                              observables=["single_z", "zz"], shots=None,
                              spatial_multiplex=1)
    feats = np.array([0.3, 0.8, 0.55, 0.1][: cfg_qrc["input_qubits"]])
    a = _numpy_single_step(cfg_qrc, seed, feats)
    b = _pennylane_single_step(cfg_qrc, seed, feats)
    err = float(np.max(np.abs(a - b)))
    print("numpy   <Z_i> =", np.round(a, 5))
    print("pennylane<Z_i>=", np.round(b, 5))
    print("max abs diff  =", err, "->", "OK" if err < 1e-6 else "MISMATCH")
    return err


if __name__ == "__main__":
    verify()
