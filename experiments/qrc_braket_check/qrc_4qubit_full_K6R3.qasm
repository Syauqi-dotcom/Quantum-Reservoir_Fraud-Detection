OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;   // q0,q1,q2 = input/reset each round; q3 = persistent memory qubit
bit[4] c;

// ================================================================
// Full 4-qubit QRC embedding circuit for ONE transaction + its K=6
// most recent transactions of the same entity (tuned config:
// n_qubits=4, input_qubits=3, reupload_rounds=3, dt=0.25, J=0.5, h=0.1)
// Feature values below are ILLUSTRATIVE (synthetic), not real transactions.
// ================================================================

// ---------- timestep t=0 (transaction 5 steps back) ----------
// -- round r=0 : reset q0,q1,q2, encode 3 new features, evolve exp(-iH dt) --
reset q[0]; reset q[1]; reset q[2];
ry(1.905229) q[0];   // feature=0.6641
ry(0.638279) q[1];   // feature=0.0984
ry(1.043322) q[2];   // feature=0.2483
rz(0.05) q[0]; rz(0.05) q[1]; rz(0.05) q[2]; rz(0.05) q[3];
h q[0]; h q[1]; cx q[0], q[1]; rz(0.099303) q[1]; cx q[0], q[1]; h q[0]; h q[1];
h q[0]; h q[2]; cx q[0], q[2]; rz(0.068921) q[2]; cx q[0], q[2]; h q[0]; h q[2];
h q[0]; h q[3]; cx q[0], q[3]; rz(-0.068698) q[3]; cx q[0], q[3]; h q[0]; h q[3];
h q[1]; h q[2]; cx q[1], q[2]; rz(-0.123684) q[2]; cx q[1], q[2]; h q[1]; h q[2];
h q[1]; h q[3]; cx q[1], q[3]; rz(0.080307) q[3]; cx q[1], q[3]; h q[1]; h q[3];
h q[2]; h q[3]; cx q[2], q[3]; rz(-0.055394) q[3]; cx q[2], q[3]; h q[2]; h q[3];

// -- round r=1 : reset q0,q1,q2, encode 3 new features, evolve exp(-iH dt) --
reset q[0]; reset q[1]; reset q[2];
ry(0.966564) q[0];   // feature=0.2159
ry(0.947925) q[1];   // feature=0.2083
ry(2.16732) q[2];   // feature=0.7809
rz(0.05) q[0]; rz(0.05) q[1]; rz(0.05) q[2]; rz(0.05) q[3];
h q[0]; h q[1]; cx q[0], q[1]; rz(0.099303) q[1]; cx q[0], q[1]; h q[0]; h q[1];
h q[0]; h q[2]; cx q[0], q[2]; rz(0.068921) q[2]; cx q[0], q[2]; h q[0]; h q[2];
h q[0]; h q[3]; cx q[0], q[3]; rz(-0.068698) q[3]; cx q[0], q[3]; h q[0]; h q[3];
h q[1]; h q[2]; cx q[1], q[2]; rz(-0.123684) q[2]; cx q[1], q[2]; h q[1]; h q[2];
h q[1]; h q[3]; cx q[1], q[3]; rz(0.080307) q[3]; cx q[1], q[3]; h q[1]; h q[3];
h q[2]; h q[3]; cx q[2], q[3]; rz(-0.055394) q[3]; cx q[2], q[3]; h q[2]; h q[3];

// -- round r=2 : reset q0,q1,q2, encode 3 new features, evolve exp(-iH dt) --
reset q[0]; reset q[1]; reset q[2];
ry(2.437225) q[0];   // feature=0.881
ry(1.156915) q[1];   // feature=0.2989
ry(2.184083) q[2];   // feature=0.7878
rz(0.05) q[0]; rz(0.05) q[1]; rz(0.05) q[2]; rz(0.05) q[3];
h q[0]; h q[1]; cx q[0], q[1]; rz(0.099303) q[1]; cx q[0], q[1]; h q[0]; h q[1];
h q[0]; h q[2]; cx q[0], q[2]; rz(0.068921) q[2]; cx q[0], q[2]; h q[0]; h q[2];
h q[0]; h q[3]; cx q[0], q[3]; rz(-0.068698) q[3]; cx q[0], q[3]; h q[0]; h q[3];
h q[1]; h q[2]; cx q[1], q[2]; rz(-0.123684) q[2]; cx q[1], q[2]; h q[1]; h q[2];
h q[1]; h q[3]; cx q[1], q[3]; rz(0.080307) q[3]; cx q[1], q[3]; h q[1]; h q[3];
h q[2]; h q[3]; cx q[2], q[3]; rz(-0.055394) q[3]; cx q[2], q[3]; h q[2]; h q[3];

// ---------- timestep t=1 (transaction 4 steps back) ----------
// -- round r=0 : reset q0,q1,q2, encode 3 new features, evolve exp(-iH dt) --
reset q[0]; reset q[1]; reset q[2];
ry(2.348727) q[0];   // feature=0.8509
ry(1.594145) q[1];   // feature=0.5117
ry(1.093855) q[2];   // feature=0.2705
rz(0.05) q[0]; rz(0.05) q[1]; rz(0.05) q[2]; rz(0.05) q[3];
h q[0]; h q[1]; cx q[0], q[1]; rz(0.099303) q[1]; cx q[0], q[1]; h q[0]; h q[1];
h q[0]; h q[2]; cx q[0], q[2]; rz(0.068921) q[2]; cx q[0], q[2]; h q[0]; h q[2];
h q[0]; h q[3]; cx q[0], q[3]; rz(-0.068698) q[3]; cx q[0], q[3]; h q[0]; h q[3];
h q[1]; h q[2]; cx q[1], q[2]; rz(-0.123684) q[2]; cx q[1], q[2]; h q[1]; h q[2];
h q[1]; h q[3]; cx q[1], q[3]; rz(0.080307) q[3]; cx q[1], q[3]; h q[1]; h q[3];
h q[2]; h q[3]; cx q[2], q[3]; rz(-0.055394) q[3]; cx q[2], q[3]; h q[2]; h q[3];

// -- round r=1 : reset q0,q1,q2, encode 3 new features, evolve exp(-iH dt) --
reset q[0]; reset q[1]; reset q[2];
ry(2.193994) q[0];   // feature=0.7918
ry(1.029524) q[1];   // feature=0.2424
ry(2.020436) q[2];   // feature=0.7173
rz(0.05) q[0]; rz(0.05) q[1]; rz(0.05) q[2]; rz(0.05) q[3];
h q[0]; h q[1]; cx q[0], q[1]; rz(0.099303) q[1]; cx q[0], q[1]; h q[0]; h q[1];
h q[0]; h q[2]; cx q[0], q[2]; rz(0.068921) q[2]; cx q[0], q[2]; h q[0]; h q[2];
h q[0]; h q[3]; cx q[0], q[3]; rz(-0.068698) q[3]; cx q[0], q[3]; h q[0]; h q[3];
h q[1]; h q[2]; cx q[1], q[2]; rz(-0.123684) q[2]; cx q[1], q[2]; h q[1]; h q[2];
h q[1]; h q[3]; cx q[1], q[3]; rz(0.080307) q[3]; cx q[1], q[3]; h q[1]; h q[3];
h q[2]; h q[3]; cx q[2], q[3]; rz(-0.055394) q[3]; cx q[2], q[3]; h q[2]; h q[3];

// -- round r=2 : reset q0,q1,q2, encode 3 new features, evolve exp(-iH dt) --
reset q[0]; reset q[1]; reset q[2];
ry(1.806876) q[0];   // feature=0.6169
ry(2.448593) q[1];   // feature=0.8847
ry(1.067215) q[2];   // feature=0.2587
rz(0.05) q[0]; rz(0.05) q[1]; rz(0.05) q[2]; rz(0.05) q[3];
h q[0]; h q[1]; cx q[0], q[1]; rz(0.099303) q[1]; cx q[0], q[1]; h q[0]; h q[1];
h q[0]; h q[2]; cx q[0], q[2]; rz(0.068921) q[2]; cx q[0], q[2]; h q[0]; h q[2];
h q[0]; h q[3]; cx q[0], q[3]; rz(-0.068698) q[3]; cx q[0], q[3]; h q[0]; h q[3];
h q[1]; h q[2]; cx q[1], q[2]; rz(-0.123684) q[2]; cx q[1], q[2]; h q[1]; h q[2];
h q[1]; h q[3]; cx q[1], q[3]; rz(0.080307) q[3]; cx q[1], q[3]; h q[1]; h q[3];
h q[2]; h q[3]; cx q[2], q[3]; rz(-0.055394) q[3]; cx q[2], q[3]; h q[2]; h q[3];

// ---------- timestep t=2 (transaction 3 steps back) ----------
// -- round r=0 : reset q0,q1,q2, encode 3 new features, evolve exp(-iH dt) --
reset q[0]; reset q[1]; reset q[2];
ry(2.139364) q[0];   // feature=0.7692
ry(1.603499) q[1];   // feature=0.5163
ry(1.06649) q[2];   // feature=0.2584
rz(0.05) q[0]; rz(0.05) q[1]; rz(0.05) q[2]; rz(0.05) q[3];
h q[0]; h q[1]; cx q[0], q[1]; rz(0.099303) q[1]; cx q[0], q[1]; h q[0]; h q[1];
h q[0]; h q[2]; cx q[0], q[2]; rz(0.068921) q[2]; cx q[0], q[2]; h q[0]; h q[2];
h q[0]; h q[3]; cx q[0], q[3]; rz(-0.068698) q[3]; cx q[0], q[3]; h q[0]; h q[3];
h q[1]; h q[2]; cx q[1], q[2]; rz(-0.123684) q[2]; cx q[1], q[2]; h q[1]; h q[2];
h q[1]; h q[3]; cx q[1], q[3]; rz(0.080307) q[3]; cx q[1], q[3]; h q[1]; h q[3];
h q[2]; h q[3]; cx q[2], q[3]; rz(-0.055394) q[3]; cx q[2], q[3]; h q[2]; h q[3];

// -- round r=1 : reset q0,q1,q2, encode 3 new features, evolve exp(-iH dt) --
reset q[0]; reset q[1]; reset q[2];
ry(0.925578) q[0];   // feature=0.1993
ry(1.566816) q[1];   // feature=0.498
ry(1.720257) q[2];   // feature=0.5745
rz(0.05) q[0]; rz(0.05) q[1]; rz(0.05) q[2]; rz(0.05) q[3];
h q[0]; h q[1]; cx q[0], q[1]; rz(0.099303) q[1]; cx q[0], q[1]; h q[0]; h q[1];
h q[0]; h q[2]; cx q[0], q[2]; rz(0.068921) q[2]; cx q[0], q[2]; h q[0]; h q[2];
h q[0]; h q[3]; cx q[0], q[3]; rz(-0.068698) q[3]; cx q[0], q[3]; h q[0]; h q[3];
h q[1]; h q[2]; cx q[1], q[2]; rz(-0.123684) q[2]; cx q[1], q[2]; h q[1]; h q[2];
h q[1]; h q[3]; cx q[1], q[3]; rz(0.080307) q[3]; cx q[1], q[3]; h q[1]; h q[3];
h q[2]; h q[3]; cx q[2], q[3]; rz(-0.055394) q[3]; cx q[2], q[3]; h q[2]; h q[3];

// -- round r=2 : reset q0,q1,q2, encode 3 new features, evolve exp(-iH dt) --
reset q[0]; reset q[1]; reset q[2];
ry(0.96649) q[0];   // feature=0.2159
ry(0.509088) q[1];   // feature=0.0634
ry(1.518813) q[2];   // feature=0.474
rz(0.05) q[0]; rz(0.05) q[1]; rz(0.05) q[2]; rz(0.05) q[3];
h q[0]; h q[1]; cx q[0], q[1]; rz(0.099303) q[1]; cx q[0], q[1]; h q[0]; h q[1];
h q[0]; h q[2]; cx q[0], q[2]; rz(0.068921) q[2]; cx q[0], q[2]; h q[0]; h q[2];
h q[0]; h q[3]; cx q[0], q[3]; rz(-0.068698) q[3]; cx q[0], q[3]; h q[0]; h q[3];
h q[1]; h q[2]; cx q[1], q[2]; rz(-0.123684) q[2]; cx q[1], q[2]; h q[1]; h q[2];
h q[1]; h q[3]; cx q[1], q[3]; rz(0.080307) q[3]; cx q[1], q[3]; h q[1]; h q[3];
h q[2]; h q[3]; cx q[2], q[3]; rz(-0.055394) q[3]; cx q[2], q[3]; h q[2]; h q[3];

// ---------- timestep t=3 (transaction 2 steps back) ----------
// -- round r=0 : reset q0,q1,q2, encode 3 new features, evolve exp(-iH dt) --
reset q[0]; reset q[1]; reset q[2];
ry(1.994169) q[0];   // feature=0.7054
ry(2.424137) q[1];   // feature=0.8767
ry(1.798726) q[2];   // feature=0.613
rz(0.05) q[0]; rz(0.05) q[1]; rz(0.05) q[2]; rz(0.05) q[3];
h q[0]; h q[1]; cx q[0], q[1]; rz(0.099303) q[1]; cx q[0], q[1]; h q[0]; h q[1];
h q[0]; h q[2]; cx q[0], q[2]; rz(0.068921) q[2]; cx q[0], q[2]; h q[0]; h q[2];
h q[0]; h q[3]; cx q[0], q[3]; rz(-0.068698) q[3]; cx q[0], q[3]; h q[0]; h q[3];
h q[1]; h q[2]; cx q[1], q[2]; rz(-0.123684) q[2]; cx q[1], q[2]; h q[1]; h q[2];
h q[1]; h q[3]; cx q[1], q[3]; rz(0.080307) q[3]; cx q[1], q[3]; h q[1]; h q[3];
h q[2]; h q[3]; cx q[2], q[3]; rz(-0.055394) q[3]; cx q[2], q[3]; h q[2]; h q[3];

// -- round r=1 : reset q0,q1,q2, encode 3 new features, evolve exp(-iH dt) --
reset q[0]; reset q[1]; reset q[2];
ry(2.4201) q[0];   // feature=0.8754
ry(2.286889) q[1];   // feature=0.8282
ry(1.038698) q[2];   // feature=0.2463
rz(0.05) q[0]; rz(0.05) q[1]; rz(0.05) q[2]; rz(0.05) q[3];
h q[0]; h q[1]; cx q[0], q[1]; rz(0.099303) q[1]; cx q[0], q[1]; h q[0]; h q[1];
h q[0]; h q[2]; cx q[0], q[2]; rz(0.068921) q[2]; cx q[0], q[2]; h q[0]; h q[2];
h q[0]; h q[3]; cx q[0], q[3]; rz(-0.068698) q[3]; cx q[0], q[3]; h q[0]; h q[3];
h q[1]; h q[2]; cx q[1], q[2]; rz(-0.123684) q[2]; cx q[1], q[2]; h q[1]; h q[2];
h q[1]; h q[3]; cx q[1], q[3]; rz(0.080307) q[3]; cx q[1], q[3]; h q[1]; h q[3];
h q[2]; h q[3]; cx q[2], q[3]; rz(-0.055394) q[3]; cx q[2], q[3]; h q[2]; h q[3];

// -- round r=2 : reset q0,q1,q2, encode 3 new features, evolve exp(-iH dt) --
reset q[0]; reset q[1]; reset q[2];
ry(2.290324) q[0];   // feature=0.8295
ry(1.999128) q[1];   // feature=0.7077
ry(1.159451) q[2];   // feature=0.3001
rz(0.05) q[0]; rz(0.05) q[1]; rz(0.05) q[2]; rz(0.05) q[3];
h q[0]; h q[1]; cx q[0], q[1]; rz(0.099303) q[1]; cx q[0], q[1]; h q[0]; h q[1];
h q[0]; h q[2]; cx q[0], q[2]; rz(0.068921) q[2]; cx q[0], q[2]; h q[0]; h q[2];
h q[0]; h q[3]; cx q[0], q[3]; rz(-0.068698) q[3]; cx q[0], q[3]; h q[0]; h q[3];
h q[1]; h q[2]; cx q[1], q[2]; rz(-0.123684) q[2]; cx q[1], q[2]; h q[1]; h q[2];
h q[1]; h q[3]; cx q[1], q[3]; rz(0.080307) q[3]; cx q[1], q[3]; h q[1]; h q[3];
h q[2]; h q[3]; cx q[2], q[3]; rz(-0.055394) q[3]; cx q[2], q[3]; h q[2]; h q[3];

// ---------- timestep t=4 (transaction 1 steps back) ----------
// -- round r=0 : reset q0,q1,q2, encode 3 new features, evolve exp(-iH dt) --
reset q[0]; reset q[1]; reset q[2];
ry(2.134923) q[0];   // feature=0.7673
ry(2.288158) q[1];   // feature=0.8287
ry(1.201444) q[2];   // feature=0.3195
rz(0.05) q[0]; rz(0.05) q[1]; rz(0.05) q[2]; rz(0.05) q[3];
h q[0]; h q[1]; cx q[0], q[1]; rz(0.099303) q[1]; cx q[0], q[1]; h q[0]; h q[1];
h q[0]; h q[2]; cx q[0], q[2]; rz(0.068921) q[2]; cx q[0], q[2]; h q[0]; h q[2];
h q[0]; h q[3]; cx q[0], q[3]; rz(-0.068698) q[3]; cx q[0], q[3]; h q[0]; h q[3];
h q[1]; h q[2]; cx q[1], q[2]; rz(-0.123684) q[2]; cx q[1], q[2]; h q[1]; h q[2];
h q[1]; h q[3]; cx q[1], q[3]; rz(0.080307) q[3]; cx q[1], q[3]; h q[1]; h q[3];
h q[2]; h q[3]; cx q[2], q[3]; rz(-0.055394) q[3]; cx q[2], q[3]; h q[2]; h q[3];

// -- round r=1 : reset q0,q1,q2, encode 3 new features, evolve exp(-iH dt) --
reset q[0]; reset q[1]; reset q[2];
ry(1.619491) q[0];   // feature=0.5243
ry(0.689878) q[1];   // feature=0.1143
ry(1.721192) q[2];   // feature=0.5749
rz(0.05) q[0]; rz(0.05) q[1]; rz(0.05) q[2]; rz(0.05) q[3];
h q[0]; h q[1]; cx q[0], q[1]; rz(0.099303) q[1]; cx q[0], q[1]; h q[0]; h q[1];
h q[0]; h q[2]; cx q[0], q[2]; rz(0.068921) q[2]; cx q[0], q[2]; h q[0]; h q[2];
h q[0]; h q[3]; cx q[0], q[3]; rz(-0.068698) q[3]; cx q[0], q[3]; h q[0]; h q[3];
h q[1]; h q[2]; cx q[1], q[2]; rz(-0.123684) q[2]; cx q[1], q[2]; h q[1]; h q[2];
h q[1]; h q[3]; cx q[1], q[3]; rz(0.080307) q[3]; cx q[1], q[3]; h q[1]; h q[3];
h q[2]; h q[3]; cx q[2], q[3]; rz(-0.055394) q[3]; cx q[2], q[3]; h q[2]; h q[3];

// -- round r=2 : reset q0,q1,q2, encode 3 new features, evolve exp(-iH dt) --
reset q[0]; reset q[1]; reset q[2];
ry(1.079501) q[0];   // feature=0.2641
ry(2.06796) q[1];   // feature=0.7385
ry(0.942876) q[2];   // feature=0.2063
rz(0.05) q[0]; rz(0.05) q[1]; rz(0.05) q[2]; rz(0.05) q[3];
h q[0]; h q[1]; cx q[0], q[1]; rz(0.099303) q[1]; cx q[0], q[1]; h q[0]; h q[1];
h q[0]; h q[2]; cx q[0], q[2]; rz(0.068921) q[2]; cx q[0], q[2]; h q[0]; h q[2];
h q[0]; h q[3]; cx q[0], q[3]; rz(-0.068698) q[3]; cx q[0], q[3]; h q[0]; h q[3];
h q[1]; h q[2]; cx q[1], q[2]; rz(-0.123684) q[2]; cx q[1], q[2]; h q[1]; h q[2];
h q[1]; h q[3]; cx q[1], q[3]; rz(0.080307) q[3]; cx q[1], q[3]; h q[1]; h q[3];
h q[2]; h q[3]; cx q[2], q[3]; rz(-0.055394) q[3]; cx q[2], q[3]; h q[2]; h q[3];

// ---------- timestep t=5 (transaction 0 steps back) ----------
// -- round r=0 : reset q0,q1,q2, encode 3 new features, evolve exp(-iH dt) --
reset q[0]; reset q[1]; reset q[2];
ry(1.227) q[0];   // feature=0.3315
ry(0.507533) q[1];   // feature=0.063
ry(0.570916) q[2];   // feature=0.0793
rz(0.05) q[0]; rz(0.05) q[1]; rz(0.05) q[2]; rz(0.05) q[3];
h q[0]; h q[1]; cx q[0], q[1]; rz(0.099303) q[1]; cx q[0], q[1]; h q[0]; h q[1];
h q[0]; h q[2]; cx q[0], q[2]; rz(0.068921) q[2]; cx q[0], q[2]; h q[0]; h q[2];
h q[0]; h q[3]; cx q[0], q[3]; rz(-0.068698) q[3]; cx q[0], q[3]; h q[0]; h q[3];
h q[1]; h q[2]; cx q[1], q[2]; rz(-0.123684) q[2]; cx q[1], q[2]; h q[1]; h q[2];
h q[1]; h q[3]; cx q[1], q[3]; rz(0.080307) q[3]; cx q[1], q[3]; h q[1]; h q[3];
h q[2]; h q[3]; cx q[2], q[3]; rz(-0.055394) q[3]; cx q[2], q[3]; h q[2]; h q[3];

// -- round r=1 : reset q0,q1,q2, encode 3 new features, evolve exp(-iH dt) --
reset q[0]; reset q[1]; reset q[2];
ry(1.56486) q[0];   // feature=0.497
ry(1.513728) q[1];   // feature=0.4715
ry(0.836375) q[2];   // feature=0.1649
rz(0.05) q[0]; rz(0.05) q[1]; rz(0.05) q[2]; rz(0.05) q[3];
h q[0]; h q[1]; cx q[0], q[1]; rz(0.099303) q[1]; cx q[0], q[1]; h q[0]; h q[1];
h q[0]; h q[2]; cx q[0], q[2]; rz(0.068921) q[2]; cx q[0], q[2]; h q[0]; h q[2];
h q[0]; h q[3]; cx q[0], q[3]; rz(-0.068698) q[3]; cx q[0], q[3]; h q[0]; h q[3];
h q[1]; h q[2]; cx q[1], q[2]; rz(-0.123684) q[2]; cx q[1], q[2]; h q[1]; h q[2];
h q[1]; h q[3]; cx q[1], q[3]; rz(0.080307) q[3]; cx q[1], q[3]; h q[1]; h q[3];
h q[2]; h q[3]; cx q[2], q[3]; rz(-0.055394) q[3]; cx q[2], q[3]; h q[2]; h q[3];

// -- round r=2 : reset q0,q1,q2, encode 3 new features, evolve exp(-iH dt) --
reset q[0]; reset q[1]; reset q[2];
ry(1.119217) q[0];   // feature=0.2818
ry(0.46399) q[1];   // feature=0.0529
ry(1.355048) q[2];   // feature=0.393
rz(0.05) q[0]; rz(0.05) q[1]; rz(0.05) q[2]; rz(0.05) q[3];
h q[0]; h q[1]; cx q[0], q[1]; rz(0.099303) q[1]; cx q[0], q[1]; h q[0]; h q[1];
h q[0]; h q[2]; cx q[0], q[2]; rz(0.068921) q[2]; cx q[0], q[2]; h q[0]; h q[2];
h q[0]; h q[3]; cx q[0], q[3]; rz(-0.068698) q[3]; cx q[0], q[3]; h q[0]; h q[3];
h q[1]; h q[2]; cx q[1], q[2]; rz(-0.123684) q[2]; cx q[1], q[2]; h q[1]; h q[2];
h q[1]; h q[3]; cx q[1], q[3]; rz(0.080307) q[3]; cx q[1], q[3]; h q[1]; h q[3];
h q[2]; h q[3]; cx q[2], q[3]; rz(-0.055394) q[3]; cx q[2], q[3]; h q[2]; h q[3];

// Final readout (only point where measurement is physically non-destructive-safe)
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
