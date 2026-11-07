function genBig(nL1,nLp,nB){const P=[[0,[2,4,6]],[1,[2,5,6]],[3,[4,5,6]]];const nq=7*(nL1+nLp)+nB;
const L=['OPENQASM 3.0;','include "stdgates.inc";',`qubit[${nq}] q;`,`bit[${nq}] c;`];
for(let j=0;j<nL1+nLp;j++){const b=7*j;
 if(j<nL1){for(const[p]of P)L.push(`h q[${b+p}];`);for(const[p,ts]of P)for(const t of ts)L.push(`cx q[${b+p}],q[${b+t}];`);for(let i=0;i<7;i++)L.push(`x q[${b+i}];`);}
 else{for(const i of[2,4,5,6])L.push(`h q[${b+i}];`);for(const[p,ts]of P)for(const t of ts)L.push(`cx q[${b+t}],q[${b+p}];`);for(let i=0;i<7;i++)L.push(`h q[${b+i}];`);}}
const B0=7*(nL1+nLp);for(let i=0;i<nB;i++)L.push(`x q[${B0+i}];`);
for(let i=0;i<nq;i++)L.push(`c[${i}] = measure q[${i}];`);return L.join('\n')+'\n';}
function anaBig(rows,nL1,nLp,nB){const S=[[3,4,5,6],[1,2,5,6],[0,2,4,6]];const nL=nL1+nLp,nq=7*nL+nB,nh=Math.ceil(nq/4);
 const R=[];for(let j=0;j<nL;j++)R.push({raw:0,dec:0,keep:0,ps:0});const B=new Array(nB).fill(0);let N=0;
 for(const[h,c]of rows){const hx=h.replace('0x','').padStart(nh,'0');const bit=k=>(parseInt(hx[nh-1-(k>>2)],16)>>(k&3))&1;N+=c;
  for(let j=0;j<nL;j++){const b=7*j,exp=j<nL1?1:0;const v=[0,1,2,3,4,5,6].map(i=>bit(b+i));const s=S.map(sp=>sp.reduce((a,i)=>a^v[i],0));const pos=s[0]*4+s[1]*2+s[2];const par=v.reduce((a,x)=>a^x,0);
   const r=R[j];if(par!==exp)r.raw+=c;if((par^(pos>0?1:0))!==exp)r.dec+=c;if(pos===0){r.keep+=c;if(par!==exp)r.ps+=c;}}
  for(let i=0;i<nB;i++)if(bit(7*nL+i)!==1)B[i]+=c;}
 const f=x=>+(x/N).toFixed(4);return{N,dec:R.map(r=>f(r.dec)),keep:R.map(r=>f(r.keep)),ps:R.map(r=>r.keep?+(r.ps/r.keep).toFixed(4):null),bare:B.map(f)};}
if(typeof module!=='undefined'){const fs=require('fs');fs.writeFileSync('big.qasm3',genBig(9,9,12));
 if(fs.existsSync('big_rows.json'))console.log(JSON.stringify(anaBig(JSON.parse(fs.readFileSync('big_rows.json')),9,9,12)));}
