"""
Stage D - Quantum Reservoir.

Physical model (Fujii & Nakajima, arXiv:1602.08159):

  * N-qubit register, fully-connected transverse-field Ising Hamiltonian

        H = sum_{i<j} J_ij X_i X_j  +  h sum_i Z_i ,     J_ij ~ U(-J/2, J/2)

    coupling matrix drawn once from `reservoir_seed` and then FROZEN.
  * Input injection is a CPTP map: the `input_qubits` are traced out and
    replaced by the product state  prod_q ( sqrt(1-f_q)|0> + sqrt(f_q)|1> ),
    where f_q are the (angle-scaled) step features.  This reset is the source
    of fading memory  ->  || T(rho) - T(sigma) ||_1 <= eta || rho - sigma ||_1.
  * Data re-uploading: `reupload_rounds` resets per time step let a small
    input register carry more than `input_qubits` features per step.
  * Between resets the register evolves under exp(-i H dt); observables are
    read at `virtual_nodes` equally spaced sub-times (time multiplexing).
  * Observables: <Z_i> and <Z_i Z_j>.  All commute, so a finite-shot readout
    is simulated correctly by sampling computational-basis bitstrings from
    diag(rho).

Only the linear readout / fusion downstream is trained - never this circuit.

Ablation switches (set by run_experiment):
  _entangle=False     -> J_ij = 0            (product dynamics, no correlations)
  _random_reservoir   -> fixed Haar unitary instead of Ising evolution
"""
from __future__ import annotations

from functools import reduce

import numpy as np

I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)


def _op_on(pauli, q, N):
    mats = [I2] * N
    mats[q] = pauli
    return reduce(np.kron, mats)


def _haar_unitary(dim, rng):
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    return q * (d / np.abs(d))


def _single_qubit_dm(f):
    f = float(np.clip(f, 0.0, 1.0))
    off = np.sqrt(max(f - f * f, 0.0))
    return np.array([[1 - f, off], [off, f]], dtype=complex)


class QuantumReservoir:
    def __init__(self, cfg_qrc: dict, reservoir_seed: int):
        self.N = int(cfg_qrc["n_qubits"])
        self.n_in = int(cfg_qrc["input_qubits"])
        self.R = int(cfg_qrc["reupload_rounds"])
        self.V = int(cfg_qrc["virtual_nodes"])
        self.dt = float(cfg_qrc["dt"])
        self.shots = cfg_qrc.get("shots", None)
        self.dim = 2 ** self.N
        self.dmem = 2 ** (self.N - self.n_in)
        self.din = 2 ** self.n_in
        self._entangle = cfg_qrc.get("_entangle", True)
        self._random = cfg_qrc.get("_random_reservoir", False)
        rng = np.random.default_rng(reservoir_seed)

        J = float(cfg_qrc["J"]) if self._entangle else 0.0
        h = float(cfg_qrc["h"])
        H = np.zeros((self.dim, self.dim), dtype=complex)
        Jij = rng.uniform(-J / 2, J / 2, size=(self.N, self.N))
        Xops = [_op_on(PX, i, self.N) for i in range(self.N)]
        for i in range(self.N):
            H += h * _op_on(PZ, i, self.N)
            for j in range(i + 1, self.N):
                H += Jij[i, j] * (Xops[i] @ Xops[j])
        self.H = H
        E, W = np.linalg.eigh(H)
        self.U_sub = (W * np.exp(-1j * E * self.dt / self.V)) @ W.conj().T
        if self._random:
            self.U_sub = _haar_unitary(self.dim, rng)
        self.U_sub_dag = self.U_sub.conj().T

        # observable bookkeeping (Z basis, so all simultaneously diagonal)
        bits = ((np.arange(self.dim)[:, None] >> np.arange(self.N - 1, -1, -1)) & 1)
        self._signs = (1 - 2 * bits).astype(float)          # (dim, N)  +-1
        if self.N <= 6:
            self.pairs = [(i, j) for i in range(self.N) for j in range(i + 1, self.N)]
        else:
            self.pairs = [(i, (i + 1) % self.N) for i in range(self.N)] + \
                         [(i, (i + 2) % self.N) for i in range(self.N)]
        self.n_obs = self.N + len(self.pairs)
        self.K_traj = int(cfg_qrc.get("_K_traj", 8))
        # final-step virtual nodes  +  per-step <Z_i> trajectory  +  mean + delta
        self.embed_dim = self.n_obs * self.V + self.K_traj * self.N + self.N + self.N
        self._rng = np.random.default_rng(reservoir_seed + 999)

    # ---- state ops ----------------------------------------------------
    def _init_state(self):
        rho = np.zeros((self.dim, self.dim), dtype=complex)
        rho[0, 0] = 1.0
        return rho

    def _reset_inputs(self, rho, fr):
        r = rho.reshape(self.din, self.dmem, self.din, self.dmem)
        rho_mem = np.einsum("akal->kl", r)
        in_dm = reduce(np.kron, [_single_qubit_dm(x) for x in fr])
        return np.kron(in_dm, rho_mem)

    def _evolve(self, rho):
        return self.U_sub @ rho @ self.U_sub_dag

    def _measure(self, rho):
        diag = np.clip(np.real(np.diag(rho)), 0.0, None)
        s = diag.sum()
        diag = diag / s if s > 0 else np.full(self.dim, 1.0 / self.dim)
        if self.shots is None:
            z = diag @ self._signs
            zz = np.array([(diag * self._signs[:, i] * self._signs[:, j]).sum()
                           for i, j in self.pairs])
            return z, zz
        idx = self._rng.choice(self.dim, size=int(self.shots), p=diag)
        sg = self._signs[idx]
        z = sg.mean(axis=0)
        zz = np.array([(sg[:, i] * sg[:, j]).mean() for i, j in self.pairs])
        return z, zz

    # ---- public -----------------------------------------------------
    def embed(self, seq_scaled: np.ndarray, mask: np.ndarray, progress=None):
        n, K, F = seq_scaled.shape
        slots = self.n_in * self.R
        Kt = self.K_traj
        out = np.zeros((n, self.embed_dim), dtype=np.float32)
        for a in range(n):
            rho = self._init_state()
            step_singles = []
            traj = np.zeros((Kt, self.N))              # right-aligned <Z_i> per step
            final_virtual = np.zeros(self.n_obs * self.V)
            active = int(mask[a].sum())
            done = 0
            for t in range(K):
                if mask[a, t] == 0.0:
                    continue
                feats = seq_scaled[a, t]
                padded = np.full(slots, 0.5)
                m = min(F, slots)
                padded[:m] = feats[:m]
                virt = []
                for r in range(self.R):
                    fr = padded[r * self.n_in:(r + 1) * self.n_in]
                    rho = self._reset_inputs(rho, fr)
                    for _ in range(self.V):
                        rho = self._evolve(rho)
                        if r == self.R - 1:
                            z, zz = self._measure(rho)
                            virt.append(np.concatenate([z, zz]))
                final_virtual = np.concatenate(virt)
                s_now = self._measure(rho)[0]
                step_singles.append(s_now)
                slot = Kt - active + done
                if 0 <= slot < Kt:
                    traj[slot] = s_now
                done += 1
            ss = np.array(step_singles) if step_singles else np.zeros((1, self.N))
            delta = ss[-1] - ss[0] if len(ss) > 1 else np.zeros(self.N)
            out[a] = np.concatenate([final_virtual, traj.ravel(),
                                     ss.mean(axis=0), delta])
            if progress is not None and (a + 1) % max(1, n // 10) == 0:
                progress(a + 1, n)
        return out


def make_embeddings(cfg_qrc, seeds, seq_scaled, mask, progress=None):
    """Spatial multiplexing: concatenate embeddings from M reservoirs (one per seed)."""
    M = int(cfg_qrc.get("spatial_multiplex", 1))
    seeds = list(seeds)[:M] if M <= len(seeds) else list(seeds) + [seeds[-1] + k for k in range(M - len(seeds))]
    mats = []
    for s in seeds[:M]:
        res = QuantumReservoir(cfg_qrc, s)
        mats.append(res.embed(seq_scaled, mask, progress=progress))
    return np.concatenate(mats, axis=1)
