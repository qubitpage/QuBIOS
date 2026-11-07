import sys, os, json
sys.path.insert(0, sys.argv[1])
from qubios.memory import build_memory_circuit
import stim
def to_qasm(sc, one):
    c = sc.without_noise().flattened(); nq=c.num_qubits; nm=c.num_measurements
    bq0, bq1 = nq, nq+1
    L=['OPENQASM 2.0;','include "qelib1.inc";',f'qreg q[{nq+2}];',f'creg c[{nm+2}];']
    L+= [f'reset q[{bq0}];', f'reset q[{bq1}];', f'x q[{bq1}];', 'barrier q;']
    k=0
    for inst in c:
        g=inst.name; ts=[t.value for t in inst.targets_copy()]
        if g in ("TICK","DETECTOR","OBSERVABLE_INCLUDE"): continue
        if g=="R": L+=[f"reset q[{q}];" for q in ts]
        elif g=="RX": L+=[x for q in ts for x in (f"reset q[{q}];",f"h q[{q}];")]
        elif g in ("H","X","Z","S"): L+=[f"{g.lower()} q[{q}];" for q in ts]
        elif g=="CX": L+=[f"cx q[{a}],q[{b}];" for a,b in zip(ts[::2],ts[1::2])]
        elif g=="M":
            if k+len(ts)==nm: L.append('barrier q;')   # final data readout together with bare qubits
            for q in ts: L.append(f"measure q[{q}] -> c[{k}];"); k+=1
        else: raise ValueError(g)
    L+= [f"measure q[{bq0}] -> c[{nm}];", f"measure q[{bq1}] -> c[{nm+1}];"]
    return "\n".join(L)+"\n"
out={}
for flags in (False,True):
    for one in (False,True):
        sc=build_memory_circuit(1,None,"Z",flags,logical_one=one)
        name=f"qubios_{'flag' if flags else 'basic'}_R1_L{int(one)}"
        q=to_qasm(sc,one); out[name]=q
        open(os.path.join(os.path.dirname(__file__), f"{name}.qasm"),"w").write(q)
        print(name, "qubits", sc.num_qubits+2, "clbits", sc.num_measurements+2, "lines", q.count("\n"), "chars", len(q))
