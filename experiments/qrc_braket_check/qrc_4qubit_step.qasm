OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;
bit[4] c;

// Stage C->D: encode 3 transaction features into 3 qubits (RY angle encoding)
ry(1.159279) q[0];  // feature f0 = 0.3
ry(2.214297) q[1];  // feature f1 = 0.8
ry(1.670964) q[2];  // feature f2 = 0.55
// q[3] stays |0> -- carries memory from the previous step in the full pipeline

// Stage D: one reservoir step exp(-iH dt), 1st-order Trotter (H = sum J_ij XiXj + h sum Zi)
// local transverse field h*dt on every qubit
rz(0.05) q[0];
rz(0.05) q[1];
rz(0.05) q[2];
rz(0.05) q[3];

// pairwise Ising coupling J_ij*dt : exp(-i*theta*Xi*Xj) = (H H) CX RZ(2*theta) CX (H H)
h q[0]; h q[1];
cx q[0], q[1];
rz(0.099303) q[1];
cx q[0], q[1];
h q[0]; h q[1];

h q[0]; h q[2];
cx q[0], q[2];
rz(0.068921) q[2];
cx q[0], q[2];
h q[0]; h q[2];

h q[0]; h q[3];
cx q[0], q[3];
rz(-0.068698) q[3];
cx q[0], q[3];
h q[0]; h q[3];

h q[1]; h q[2];
cx q[1], q[2];
rz(-0.123684) q[2];
cx q[1], q[2];
h q[1]; h q[2];

h q[1]; h q[3];
cx q[1], q[3];
rz(0.080307) q[3];
cx q[1], q[3];
h q[1]; h q[3];

h q[2]; h q[3];
cx q[2], q[3];
rz(-0.055394) q[3];
cx q[2], q[3];
h q[2]; h q[3];

// Stage D: measurement (Pauli-Z basis) -> reservoir observables z_qrc
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
