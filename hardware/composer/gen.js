function genMem(d){const P=[[0,[2,4,6]],[1,[2,5,6]],[3,[4,5,6]]];const L=['OPENQASM 3.0;','include "stdgates.inc";','qubit[64] q;','bit[64] c;'];
const lg=[];for(let i=0;i<8;i++)lg.push([i<4?'L1':'Lp',7*i]);
for(const[k,b]of lg){for(const[p]of P)L.push(`h q[${b+p}];`);for(const[p,ts]of P)for(const t of ts)L.push(`cx q[${b+p}],q[${b+t}];`);for(let i=0;i<7;i++)L.push(k==='L1'?`x q[${b+i}];`:`h q[${b+i}];`);}
for(let i=56;i<60;i++)L.push(`x q[${i}];`);for(let i=60;i<64;i++)L.push(`h q[${i}];`);L.push('barrier q;');
if(d>0){L.push(`delay[${d}us] q;`);L.push('barrier q;');}
for(const[k,b]of lg)if(k==='Lp')for(let i=0;i<7;i++)L.push(`h q[${b+i}];`);for(let i=60;i<64;i++)L.push(`h q[${i}];`);
for(let i=0;i<64;i++)L.push(`c[${i}] = measure q[${i}];`);return L.join('\n')+'\n';}
if(typeof module!=='undefined'){const fs=require('fs');for(const d of [0,10,30,60,100]){console.log(d, genMem(d)===fs.readFileSync(`mem_d${d}.qasm3`,'utf8'));}}
