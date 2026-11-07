import sys, json, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from qubios.memory import build_memory_circuit, DemDecoder, NoiseModel
P_DEC = 3e-3
spec = {}
for flags in (False, True):
    for one in (False, True):
        name = f"{'flag' if flags else 'basic'}_L{int(one)}"
        sc = build_memory_circuit(1, NoiseModel(P_DEC), "Z", flags, logical_one=one)
        nm, nd = sc.num_measurements, sc.num_detectors
        conv = sc.compile_m2d_converter()
        Z = np.zeros((1, nm), dtype=bool)
        d0, o0 = conv.convert(measurements=Z, separate_observables=True)
        A = []; Lrow = []
        for j in range(nm):
            e = Z.copy(); e[0, j] = True
            d, o = conv.convert(measurements=e, separate_observables=True)
            A.append(int(sum(1 << i for i in range(nd) if d[0, i] ^ d0[0, i])))
            Lrow.append(int(o[0, 0] ^ o0[0, 0]))
        dec = DemDecoder(sc)
        table = "".join(str(int(dec.decode(np.array([(s >> i) & 1 for i in range(nd)], dtype=np.uint8))[0])) for s in range(1 << nd))
        d0i = int(sum(1 << i for i in range(nd) if d0[0, i])); 
        spec[name] = dict(nm=nm, nd=nd, colDet=A, colObs=Lrow, d0=d0i, o0=int(o0[0, 0]), table=table, one=one)
        # self-check against direct decode on random noisy samples
        dets, obs, meas = None, None, sc.compile_sampler(seed=5).sample(3000)
        D, O = conv.convert(measurements=meas, separate_observables=True)
        direct = np.array([dec.decode(d)[0] for d in D]) ^ O[:, 0]
        via = []
        for row in meas:
            s = d0i; o = int(o0[0, 0])
            for j in np.nonzero(row)[0]: s ^= A[j]; o ^= Lrow[j]
            via.append(int(table[s]) ^ o)
        assert (np.array(via) == direct).all(), name
        print(name, "nm", nm, "nd", nd, "table OK; sim logical err", direct.mean())
open("spec.json", "w").write(json.dumps(spec))
print(len(json.dumps(spec)), "bytes")
