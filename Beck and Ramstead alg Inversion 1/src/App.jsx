/**
 * TORUS EMERGENCE via Inverted DMBD
 * ════════════════════════════════════════════════════════════════════════════
 * Beck & Ramstead (2025) arXiv:2502.21217
 *
 * KEY VISUAL PRINCIPLE:
 *   Color encodes CERTAINTY of q(ωᵢ), not just role.
 *   At t=0: all points grey — q=(⅓,⅓,⅓), maximum entropy, no blanket.
 *   Over time: color saturation = cert = (max_k q(ωᵢ=k) − ⅓) / (⅔)
 *   The blanket APPEARS as certainty concentrates on the torus surface.
 *
 * THREE DESIGN LEVERS (manuscript §4):
 *
 *   PRECISION CRAFTING — prior concentration γ
 *     Controls sharpness of p*(ω). High γ = strong role signals.
 *     Acts on KL[q‖p*] in G(π). Shell band σ_B shrinks with precision.
 *     Visual: blanket forms faster but risks freezing early.
 *
 *   CURIOSITY SCULPTING — epistemic weight β in EFE
 *     G(π) = KL[q‖p*] − β·H[q]
 *     High β = epistemic value dominates = elements stay uncertain longer.
 *     Visual: points stay grey longer, explore more space before committing.
 *
 *   PREDICTION EMBEDDING — EFE/noise ratio (annealing)
 *     Controls σ_Langevin. High embedding = low noise = trust predictions.
 *     σ(t) = σ₀·(1−embedding·0.7)·exp(−t·0.002·embedding)
 *     Visual: trajectories become smoother, less jitter, sharper convergence.
 *
 * VBEM ATTEND [Eq. 26]:
 *   log q(ωᵢ=k) ∝ log p(yᵢ|ωᵢ=k, macro) + λ·log p*(ωᵢ=k)
 *
 * EFE per element:
 *   G(π) = KL[q(ωᵢ)‖p*(ωᵢ)] − β·H[q(ωᵢ)]
 *   F = −λ(t)·q_k·G(π)·∇d_torus
 *
 * Euler-Maruyama:
 *   v ← α·v + F·dt + σ·√dt·η,   η~N(0,1)
 *   x ← x + v·dt
 * ════════════════════════════════════════════════════════════════════════════
 */

import { useState, useEffect, useRef, useCallback } from "react";

// ── PARAMETERS ────────────────────────────────────────────────────────────────
const N        = 2000;
const R_MAJOR  = 2.6;    // torus: distance from Y-axis to tube centre
const R_MINOR  = 0.85;   // torus: tube radius
const DT       = 0.018;
const DAMPING  = 0.80;
const R_EXCL   = 0.22;
const K_EXCL   = 0.55;
const LAMBDA0  = 0.52;
const ETA_LAM  = 0.004;  // λ growth rate
const GAMMA0   = 2.2;

// ── COLOURS ───────────────────────────────────────────────────────────────────
const C = {
  bg:"#04040e", surface:"#07071a", border:"#10102a",
  accent:"#c8a45a", dimAcc:"#4a3810",
  Z:"#3a7fd4", B:"#c8a45a", S:"#b03030",
  grey:"#303040",
  green:"#42b060", teal:"#38aaa0", strain:"#e05020",
  text:"#ccc4b8", dim:"#383830", mid:"#686050",
  lev1:"#a06ad4", lev2:"#38b0a0", lev3:"#e08040",
};
const MONO = "'Fira Code','JetBrains Mono','Courier New',monospace";

// ── MATH ─────────────────────────────────────────────────────────────────────
function randn(){
  let u,v;
  do{u=Math.random();}while(!u);
  do{v=Math.random();}while(!v);
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
}
function clamp(x,a,b){return x<a?a:x>b?b:x;}

// Signed distance to torus surface (axis = Y)
function distTorus(x,y,z){
  const rXZ=Math.sqrt(x*x+z*z);
  return Math.sqrt((rXZ-R_MAJOR)**2+y*y)-R_MINOR;
}

// Gradient of signed distance (unit normal toward surface)
function gradTorus(x,y,z){
  const rXZ=Math.sqrt(x*x+z*z)+1e-9;
  const nxz_x=x/rXZ, nxz_z=z/rXZ;
  const dx=rXZ-R_MAJOR;
  const d=Math.sqrt(dx*dx+y*y)+1e-9;
  return {gx:(dx/d)*nxz_x, gy:y/d, gz:(dx/d)*nxz_z};
}

// ── INIT ──────────────────────────────────────────────────────────────────────
function initParticles(){
  const px=new Float32Array(N),py=new Float32Array(N),pz=new Float32Array(N);
  const vx=new Float32Array(N),vy=new Float32Array(N),vz=new Float32Array(N);
  const q=new Float32Array(N*3);
  for(let i=0;i<N;i++){
    // Uniform cube — pure chaos, no structure
    px[i]=(Math.random()-0.5)*10;
    py[i]=(Math.random()-0.5)*10;
    pz[i]=(Math.random()-0.5)*10;
    vx[i]=randn()*0.03; vy[i]=randn()*0.03; vz[i]=randn()*0.03;
    // q = Uniform(⅓,⅓,⅓) — maximum entropy, H = log3 ≈ 1.099
    q[i*3]=1/3; q[i*3+1]=1/3; q[i*3+2]=1/3;
  }
  return {px,py,pz,vx,vy,vz,q};
}

// ── DESIGNER PRIOR ────────────────────────────────────────────────────────────
// p*(ωᵢ=k) ∝ exp(γ · prop_k)  where prop_k ∈ [0,1]
// Based on signed distance d = distTorus(xᵢ):
//   B (blanket/shell): d ≈ 0
//   Z (internal):      d < 0
//   S (external):      d > 0
function computePrior(px,py,pz,conv,precision){
  const prior=new Float32Array(N*3);
  const gamma=GAMMA0+4.0*precision;
  // Shell band tightens with convergence and precision
  const sigB=clamp((1.05-conv*0.60-precision*0.25)*R_MINOR, 0.12, R_MINOR*1.1);

  for(let i=0;i<N;i++){
    const d=distTorus(px[i],py[i],pz[i]);
    const base=i*3;
    const propB=Math.exp(-0.5*(d/sigB)**2);
    const propZ=1/(1+Math.exp( d/0.50));   // sigmoid: 1 inside, 0 outside
    const propS=1/(1+Math.exp(-d/0.50));   // sigmoid: 0 inside, 1 outside
    const eB=Math.exp(gamma*propB);
    const eZ=Math.exp(gamma*propZ*0.72);
    const eS=Math.exp(gamma*propS*0.55);
    const Z=eB+eZ+eS+1e-12;
    prior[base]=eS/Z; prior[base+1]=eB/Z; prior[base+2]=eZ/Z;
  }
  return prior;
}

// ── ATTEND + INFER [Eq. 26–27] ───────────────────────────────────────────────
function attendStep(px,py,pz,q,prior,lambda){
  // INFER: macroscopic latents — mean signed distance per role
  let dB=0,wB=0,dZ=0,wZ=0,dS=0,wS=0;
  for(let i=0;i<N;i++){
    const d=distTorus(px[i],py[i],pz[i]);
    dB+=q[i*3+1]*d; wB+=q[i*3+1];
    dZ+=q[i*3+2]*d; wZ+=q[i*3+2];
    dS+=q[i*3  ]*d; wS+=q[i*3  ];
  }
  const muB=dB/(wB+1e-9),muZ=dZ/(wZ+1e-9),muS=dS/(wS+1e-9);
  let vB=0,vZ=0,vS=0;
  for(let i=0;i<N;i++){
    const d=distTorus(px[i],py[i],pz[i]);
    vB+=q[i*3+1]*(d-muB)**2;
    vZ+=q[i*3+2]*(d-muZ)**2;
    vS+=q[i*3  ]*(d-muS)**2;
  }
  const sigB=Math.sqrt(vB/(wB+1e-9))+0.18;
  const sigZ=Math.sqrt(vZ/(wZ+1e-9))+0.28;
  const sigS=Math.sqrt(vS/(wS+1e-9))+0.28;

  // ATTEND [Eq. 26]: log q(ωᵢ=k) ∝ llk + λ·log p*(ωᵢ=k)
  for(let i=0;i<N;i++){
    const d=distTorus(px[i],py[i],pz[i]);
    const base=i*3;
    const llB=-0.5*((d-muB)/sigB)**2;
    const llZ=-0.5*((d-muZ)/sigZ)**2;
    const llS=-0.5*((d-muS)/sigS)**2;
    const lB=llB+lambda*Math.log(Math.max(prior[base+1],1e-10));
    const lZ=llZ+lambda*Math.log(Math.max(prior[base+2],1e-10));
    const lS=llS+lambda*Math.log(Math.max(prior[base  ],1e-10));
    const mx=Math.max(lB,lZ,lS);
    const eB=Math.exp(lB-mx),eZ=Math.exp(lZ-mx),eS=Math.exp(lS-mx);
    const sm=eB+eZ+eS+1e-12;
    q[base]=eS/sm; q[base+1]=eB/sm; q[base+2]=eZ/sm;
  }
  return {muB,muZ,muS,sigB,wB,wZ,wS};
}

// ── EFE FORCES + PHYSICS ─────────────────────────────────────────────────────
function physicsStep(particles,prior,q,lambda,levers,iter){
  const {px,py,pz,vx,vy,vz}=particles;
  const {precision,curiosity,embedding}=levers;

  // CURIOSITY SCULPTING: β controls epistemic weight in G(π)
  const beta=0.18+curiosity*0.60;

  // PREDICTION EMBEDDING: annealing of Langevin noise
  const sigmaBase=0.040-precision*0.010;
  const anneal=Math.exp(-iter*0.0018*embedding);
  const sigma=clamp(sigmaBase*(1-embedding*0.68)*anneal+0.006, 0.005, 0.06);

  // PRECISION CRAFTING: amplifies EFE forces
  const forceScale=0.20+precision*0.35;

  const fx=new Float32Array(N),fy=new Float32Array(N),fz=new Float32Array(N);

  // Stochastic contact repulsion
  const PAIRS=4500;
  for(let s=0;s<PAIRS;s++){
    const i=Math.floor(Math.random()*N),j=Math.floor(Math.random()*N);
    if(i===j)continue;
    const dx=px[i]-px[j],dy=py[i]-py[j],dz=pz[i]-pz[j];
    const d2=dx*dx+dy*dy+dz*dz;
    if(d2<R_EXCL*R_EXCL&&d2>1e-8){
      const d=Math.sqrt(d2),f=K_EXCL*(R_EXCL-d)/d;
      fx[i]+=f*dx;fy[i]+=f*dy;fz[i]+=f*dz;
      fx[j]-=f*dx;fy[j]-=f*dy;fz[j]-=f*dz;
    }
  }

  // EFE forces — push each element toward its designated zone
  for(let i=0;i<N;i++){
    const base=i*3;
    const qS=q[base],qB=q[base+1],qZ=q[base+2];
    const pS=prior[base],pB=prior[base+1],pZ=prior[base+2];

    // G(π) = KL[q‖p*] − β·H[q]
    const kl=qS*Math.log(Math.max(qS/(pS+1e-9),1e-9))
            +qB*Math.log(Math.max(qB/(pB+1e-9),1e-9))
            +qZ*Math.log(Math.max(qZ/(pZ+1e-9),1e-9));
    const H=-(qS*Math.log(qS+1e-9)+qB*Math.log(qB+1e-9)+qZ*Math.log(qZ+1e-9));
    const G=clamp(kl-beta*H, 0, 5);

    const d=distTorus(px[i],py[i],pz[i]);
    const {gx,gy,gz}=gradTorus(px[i],py[i],pz[i]);
    const fmag=lambda*G*forceScale;

    // B elements: pulled toward surface (d→0), force = -d·gradient
    if(qB>0.20){
      const f=-d*fmag*(qB*1.8);
      fx[i]+=f*gx; fy[i]+=f*gy; fz[i]+=f*gz;
    }
    // Z elements: pushed inward (toward tube interior, d<0)
    if(qZ>0.20){
      const f= fmag*(qZ*0.55);   // push along -gradient (inward)
      fx[i]-=f*gx; fy[i]-=f*gy; fz[i]-=f*gz;
    }
    // S elements: pushed outward (d>0)
    if(qS>0.20){
      const f= fmag*(qS*0.40);   // push along +gradient (outward)
      fx[i]+=f*gx; fy[i]+=f*gy; fz[i]+=f*gz;
    }

    // Langevin thermal noise
    fx[i]+=randn()*sigma;
    fy[i]+=randn()*sigma;
    fz[i]+=randn()*sigma;
  }

  // Euler-Maruyama integration
  for(let i=0;i<N;i++){
    vx[i]=vx[i]*DAMPING+fx[i]*DT;
    vy[i]=vy[i]*DAMPING+fy[i]*DT;
    vz[i]=vz[i]*DAMPING+fz[i]*DT;
    px[i]+=vx[i]*DT;
    py[i]+=vy[i]*DT;
    pz[i]+=vz[i]*DT;
    // Soft boundary
    const r=Math.sqrt(px[i]**2+py[i]**2+pz[i]**2);
    if(r>8){const s=8/r;px[i]*=s;py[i]*=s;pz[i]*=s;vx[i]*=-0.2;vy[i]*=-0.2;vz[i]*=-0.2;}
  }
}

// ── METRICS ───────────────────────────────────────────────────────────────────
function computeMetrics(q,prior,px,py,pz){
  let elbo=0,kl=0,H_q=0,wB=0,nBonTorus=0;
  for(let i=0;i<N;i++){
    for(let k=0;k<3;k++){
      const qi=Math.max(q[i*3+k],1e-10),pi=Math.max(prior[i*3+k],1e-10);
      elbo+=qi*(Math.log(pi)-Math.log(qi));
      kl  +=qi*Math.log(qi/pi);
      H_q -=qi*Math.log(qi);
    }
    wB+=q[i*3+1];
    if(q[i*3+1]>0.45&&Math.abs(distTorus(px[i],py[i],pz[i]))<0.55) nBonTorus++;
  }
  const conv=wB>10?nBonTorus/Math.max(wB,1):0;
  return {elbo:elbo/N, kl:kl/N, H_q:H_q/N, conv};
}

// ── 3D CANVAS RENDERER ────────────────────────────────────────────────────────
/**
 * COLOR ENCODING:
 *   cert(i) = (max_k q(ωᵢ=k) − 1/3) / (2/3)   ∈ [0,1]
 *   At cert=0: grey (#303040) — q is uniform, no assignment
 *   At cert=1: full role color (gold/blue/red)
 *
 *   Hue from role, saturation and lightness from cert²:
 *     color = lerp(grey, roleColor, cert²)
 *   Glow appears only when cert > 0.65 AND role = B (blanket forming)
 */
function render3D(canvas,particles,q,angle,iter){
  const {px,py,pz}=particles;
  const ctx=canvas.getContext("2d");
  const W=canvas.width,H=canvas.height;

  ctx.fillStyle=C.bg; ctx.fillRect(0,0,W,H);

  const scale=W*0.088;
  const cx=W/2,cy=H/2;
  const cosA=Math.cos(angle),sinA=Math.sin(angle);
  const tilt=0.38;
  const cosT=Math.cos(tilt),sinT=Math.sin(tilt);

  function proj(x,y,z){
    // Rotate Y
    const rx=x*cosA+z*sinA, ry=y, rz=-x*sinA+z*cosA;
    // Tilt X
    const fx=rx, fy=ry*cosT-rz*sinT, fz=ry*sinT+rz*cosT;
    const d=14, w=d/(d+fz);
    return {sx:cx+fx*scale*w, sy:cy-fy*scale*w, z:fz, w};
  }

  // Target torus wireframe (very faint)
  ctx.strokeStyle="rgba(200,164,90,0.06)"; ctx.lineWidth=0.6;
  for(let phi=0;phi<360;phi+=20){
    const phr=phi*Math.PI/180;
    ctx.beginPath();
    for(let theta=0;theta<=360;theta+=8){
      const thr=theta*Math.PI/180;
      const x=(R_MAJOR+R_MINOR*Math.cos(thr))*Math.cos(phr);
      const y= R_MINOR*Math.sin(thr);
      const z=(R_MAJOR+R_MINOR*Math.cos(thr))*Math.sin(phr);
      const p=proj(x,y,z);
      theta===0?ctx.moveTo(p.sx,p.sy):ctx.lineTo(p.sx,p.sy);
    }
    ctx.stroke();
  }
  for(let theta=0;theta<360;theta+=30){
    const thr=theta*Math.PI/180;
    ctx.beginPath();
    for(let phi=0;phi<=360;phi+=6){
      const phr=phi*Math.PI/180;
      const x=(R_MAJOR+R_MINOR*Math.cos(thr))*Math.cos(phr);
      const y= R_MINOR*Math.sin(thr);
      const z=(R_MAJOR+R_MINOR*Math.cos(thr))*Math.sin(phr);
      const p=proj(x,y,z);
      phi===0?ctx.moveTo(p.sx,p.sy):ctx.lineTo(p.sx,p.sy);
    }
    ctx.stroke();
  }

  // Sort by depth
  const pts=Array.from({length:N},(_,i)=>{
    const p=proj(px[i],py[i],pz[i]);
    return {i,sx:p.sx,sy:p.sy,z:p.z,w:p.w};
  }).sort((a,b)=>a.z-b.z);

  // Draw particles — color encodes certainty
  for(const {i,sx,sy,w} of pts){
    const qS=q[i*3],qB=q[i*3+1],qZ=q[i*3+2];
    const maxQ=Math.max(qS,qB,qZ);
    const role=qS>=qB&&qS>=qZ?0:qB>=qZ?1:2;

    // cert ∈ [0,1]: 0 = uniform q, 1 = fully assigned
    const cert=clamp((maxQ-1/3)/(2/3),0,1);
    const cert2=cert*cert;  // squared for sharper visual onset

    // Role hues: S=0(red), B=38(gold), Z=210(blue)
    const hues=[0,38,210];
    const hue=hues[role];

    // Size: small and dim when uncertain, larger and bright when certain
    const baseR=0.8+cert*1.4;
    const r_pt=baseR*clamp(w,0.4,1.4);
    const sat=Math.round(cert2*70);
    const lit=Math.round(12+cert2*22);
    const alpha=0.10+cert2*0.75;

    // Glow only for high-certainty blanket elements
    if(role===1&&cert>0.62){
      const glowR=r_pt*6;
      const g=ctx.createRadialGradient(sx,sy,0,sx,sy,glowR);
      g.addColorStop(0,`rgba(200,164,90,${(cert-0.55)*0.35})`);
      g.addColorStop(1,`rgba(200,164,90,0)`);
      ctx.fillStyle=g;
      ctx.beginPath();ctx.arc(sx,sy,glowR,0,Math.PI*2);ctx.fill();
    }

    ctx.fillStyle=cert2<0.04
      ? `rgba(48,48,64,${0.15+cert*0.3})`  // grey when uncertain
      : `hsla(${hue},${sat}%,${lit}%,${alpha})`;
    ctx.beginPath();ctx.arc(sx,sy,Math.max(r_pt,0.4),0,Math.PI*2);ctx.fill();
  }

  // Entropy label
  let totalH=0;
  for(let i=0;i<N;i++){
    const qS=q[i*3],qB=q[i*3+1],qZ=q[i*3+2];
    totalH-=(qS*Math.log(qS+1e-9)+qB*Math.log(qB+1e-9)+qZ*Math.log(qZ+1e-9));
  }
  const meanH=totalH/N;
  ctx.fillStyle=C.dim; ctx.font=`8px ${MONO}`;
  ctx.fillText(`H[q]=${meanH.toFixed(3)}  log3=${Math.log(3).toFixed(3)}  iter ${iter}`,8,H-8);
}

// ── MINI CHART ────────────────────────────────────────────────────────────────
function MiniChart({history,color,label,tick}){
  const wrapRef=useRef();
  const canvasRef=useRef();

  const draw=useCallback(()=>{
    const wrap=wrapRef.current; const c=canvasRef.current;
    if(!wrap||!c) return;
    // Match canvas intrinsic size to actual CSS size (fixes blank/blurry canvas)
    const W=wrap.clientWidth||240;
    const H=80;
    if(c.width!==W) c.width=W;
    if(c.height!==H) c.height=H;

    const ctx=c.getContext("2d");
    ctx.fillStyle=C.bg; ctx.fillRect(0,0,W,H);

    if(!history||history.length<2){
      // Draw empty state so panel isn't blank
      ctx.fillStyle=C.dim; ctx.font=`8px ${MONO}`;
      ctx.fillText("waiting for data...",8,H/2+3);
      return;
    }

    const pad={t:12,r:8,b:16,l:40};
    const iw=W-pad.l-pad.r, ih=H-pad.t-pad.b;
    const mn=Math.min(...history), mx=Math.max(...history);
    const rng=mx-mn||0.001;
    const sx=i=>pad.l+(i/(history.length-1))*iw;
    const sy=v=>pad.t+(1-(v-mn)/rng)*ih;

    // Grid lines
    for(let i=0;i<=3;i++){
      const y=pad.t+(i/3)*ih;
      ctx.strokeStyle="#10102a"; ctx.lineWidth=0.5;
      ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(pad.l+iw,y); ctx.stroke();
      ctx.fillStyle=C.dim; ctx.font=`7px ${MONO}`;
      ctx.textAlign="right";
      ctx.fillText((mx-(i/3)*rng).toFixed(3), pad.l-3, y+3);
      ctx.textAlign="left";
    }

    // Area fill
    const gr=ctx.createLinearGradient(0,pad.t,0,pad.t+ih);
    gr.addColorStop(0,color+"44"); gr.addColorStop(1,color+"06");
    ctx.fillStyle=gr; ctx.beginPath();
    ctx.moveTo(sx(0),sy(history[0]));
    history.forEach((v,i)=>ctx.lineTo(sx(i),sy(v)));
    ctx.lineTo(sx(history.length-1),pad.t+ih);
    ctx.lineTo(sx(0),pad.t+ih); ctx.closePath(); ctx.fill();

    // Line
    ctx.strokeStyle=color; ctx.lineWidth=1.6; ctx.lineJoin="round";
    ctx.beginPath();
    history.forEach((v,i)=>i===0?ctx.moveTo(sx(i),sy(v)):ctx.lineTo(sx(i),sy(v)));
    ctx.stroke();

    // Last value
    const last=history[history.length-1];
    const ly=clamp(sy(last)-3, 8, H-8);
    ctx.fillStyle=color; ctx.font=`bold 8px ${MONO}`;
    ctx.textAlign="right";
    ctx.fillText(last.toFixed(4), W-4, ly);
    ctx.textAlign="left";

    // Label
    ctx.fillStyle=C.mid; ctx.font=`7px ${MONO}`;
    ctx.fillText(label, pad.l+2, pad.t+9);
  },[history,color,label]);

  // Redraw whenever tick changes (each sim step) or on mount
  useEffect(()=>{ draw(); },[tick, draw]);

  // Also redraw on resize
  useEffect(()=>{
    const ro=new ResizeObserver(()=>draw());
    if(wrapRef.current) ro.observe(wrapRef.current);
    return()=>ro.disconnect();
  },[draw]);

  return(
    <div ref={wrapRef} style={{width:"100%"}}>
      <canvas ref={canvasRef} style={{display:"block",width:"100%",height:"80px"}}/>
    </div>
  );
}

// ── INVERSION SCHEMA ──────────────────────────────────────────────────────────
function Schema({schemaStep,lambda,elbo,kl,Hq,conv,muB,levers}){
  const ref=useRef();
  useEffect(()=>{
    const c=ref.current; if(!c)return;
    const ctx=c.getContext("2d");
    const W=c.width,H=c.height;
    ctx.fillStyle=C.bg;ctx.fillRect(0,0,W,H);

    const steps=[
      {label:"1 · DESIGNER PRIOR",    sub:"p*(ω) ∝ exp(γ·prop_k(d_torus))",  live:`γ=${(GAMMA0+4*levers.precision).toFixed(1)}`,  col:C.lev1, key:0},
      {label:"2 · ATTEND  [Eq.26]",   sub:"q(ω)←softmax[ll_k + λ·log p*]",   live:`λ=${lambda.toFixed(3)}`,                        col:C.B,    key:1},
      {label:"3 · INFER   [Eq.27]",   sub:"d̄_B=E_{q_B}[d(x)]  σ_B=Std[d]",  live:`d̄_B=${muB.toFixed(3)}`,                        col:C.teal, key:2},
      {label:"4 · EFE FORCES",         sub:"G(π)=KL[q‖p*]−β·H[q]  →  F",     live:`KL=${kl.toFixed(3)}`,                           col:C.strain,key:3},
      {label:"5 · PHYSICS  (E-M)",     sub:"v←α·v+F·dt   x←x+v·dt+ση",       live:`H=${Hq.toFixed(3)}`,                            col:C.mid,  key:4},
      {label:"6 · REIFY CHECK",        sub:"‖xᵢ(B)‖≈R_torus → blanket formed",live:`conv=${(conv*100).toFixed(1)}%`,                col:C.green,key:5},
    ];

    const bW=W-32,bH=34,startY=18,gap=10;

    steps.forEach(({label,sub,live,col,key},idx)=>{
      const bx=16,by=startY+idx*(bH+gap);
      const isActive=key===(schemaStep%6);
      const isPast=key<(schemaStep%6);

      // Box fill
      ctx.fillStyle=isActive?`${col}1a`:isPast?`${C.teal}08`:`${C.surface}`;
      ctx.strokeStyle=isActive?col:isPast?C.teal+"44":C.border;
      ctx.lineWidth=isActive?1.5:0.7;
      ctx.beginPath();ctx.roundRect(bx,by,bW,bH,4);ctx.fill();ctx.stroke();

      // Active indicator
      if(isActive){
        ctx.fillStyle=col;
        ctx.beginPath();ctx.arc(bx+8,by+bH/2,3,0,Math.PI*2);ctx.fill();
      }

      // Label
      ctx.font=`bold 8.5px ${MONO}`;
      ctx.fillStyle=isActive?col:isPast?C.teal:C.mid;
      ctx.fillText(label,bx+18,by+13);

      // Sub
      ctx.font=`7.5px ${MONO}`;
      ctx.fillStyle=isActive?C.text:C.dim;
      ctx.fillText(sub,bx+18,by+26);

      // Live value (right aligned)
      ctx.font=`8px ${MONO}`;
      ctx.fillStyle=isActive?C.green:C.dim;
      ctx.textAlign="right";
      ctx.fillText(live,bx+bW-6,by+13);
      ctx.textAlign="left";

      // Arrow down
      if(key<5){
        const ax=bx+bW/2,ay1=by+bH+1,ay2=by+bH+gap-1;
        ctx.strokeStyle=isActive?col+"88":C.border;
        ctx.lineWidth=1;ctx.setLineDash([2,3]);
        ctx.beginPath();ctx.moveTo(ax,ay1);ctx.lineTo(ax,ay2);ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle=isActive?col+"88":C.border;
        ctx.beginPath();ctx.moveTo(ax,ay2+1);ctx.lineTo(ax-3,ay2-4);ctx.lineTo(ax+3,ay2-4);ctx.closePath();ctx.fill();
      }
    });

    // Feedback loop arrow
    const loopX=W-10;
    const y0=startY+bH/2, y1=startY+5*(bH+gap)+bH/2;
    ctx.strokeStyle=C.teal+"33";ctx.lineWidth=1;ctx.setLineDash([2,6]);
    ctx.beginPath();
    ctx.moveTo(16+bW,y1);ctx.lineTo(loopX,y1);
    ctx.lineTo(loopX,y0);ctx.lineTo(16+bW,y0);
    ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=C.teal+"55";ctx.font=`7px ${MONO}`;ctx.textAlign="center";
    ctx.fillText("REPEAT",loopX-2,H/2);ctx.textAlign="left";

    // Header
    ctx.fillStyle=C.dimAcc;ctx.font=`7px ${MONO}`;
    ctx.fillText("INVERSION SCHEMA — DMBD⁻¹",16,11);
  },[schemaStep,lambda,elbo,kl,Hq,conv,muB,levers]);

  return <canvas ref={ref} width={260} height={300} style={{display:"block",width:"100%"}}/>;
}

// ── LEVER PANEL ───────────────────────────────────────────────────────────────
function LeverPanel({levers,setLevers,running}){
  const defs=[
    {
      key:"precision",label:"Precision Crafting",color:C.lev1,
      what:"Sharpens prior p*(ω)",
      how:"γ = γ₀ + 4·P  →  σ_B shrinks",
      acts:"KL[q‖p*] in G(π)",
      trade:"Fast convergence ↔ local minima",
    },
    {
      key:"curiosity",label:"Curiosity Sculpting",color:C.lev2,
      what:"Epistemic weight β in EFE",
      how:"G(π) = KL − β·H[q],  β = 0.18+0.6·C",
      acts:"−H[q] (epistemic value term)",
      trade:"Exploration ↔ exploitation",
    },
    {
      key:"embedding",label:"Prediction Embedding",color:C.lev3,
      what:"EFE/noise ratio (annealing)",
      how:"σ(t) = σ₀·(1−0.7·E)·exp(−t·E)",
      acts:"σ_Langevin (thermal noise)",
      trade:"Precision ↔ robustness",
    },
  ];
  return(
    <div style={{display:"flex",flexDirection:"column",gap:10}}>
      {defs.map(({key,label,color,what,how,acts,trade})=>(
        <div key={key} style={{background:C.surface,border:`1px solid ${C.border}`,
          borderRadius:6,padding:"9px 11px"}}>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:5}}>
            <span style={{fontSize:9,fontFamily:MONO,color,fontWeight:"bold",letterSpacing:1}}>
              {label}
            </span>
            <span style={{fontSize:9,fontFamily:MONO,color}}>{levers[key].toFixed(2)}</span>
          </div>
          <input type="range" min={0} max={1} step={0.05} value={levers[key]}
            onChange={e=>setLevers(l=>({...l,[key]:parseFloat(e.target.value)}))}
            style={{width:"100%",accentColor:color,marginBottom:6}}/>
          <div style={{fontSize:7.5,fontFamily:MONO,color:C.dim,lineHeight:1.7}}>
            <span style={{color:C.mid}}>↳ {what}</span><br/>
            <span style={{color:color+"bb"}}>{how}</span><br/>
            <span style={{color:C.dim}}>acts on: {acts}</span><br/>
            <span style={{color:C.dim,fontStyle:"italic"}}>{trade}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── EFE LOG ───────────────────────────────────────────────────────────────────
function EFELog({entries}){
  const ref=useRef();
  useEffect(()=>{if(ref.current)ref.current.scrollTop=ref.current.scrollHeight;},[entries]);
  const TC={attend:C.B,infer:C.teal,efe:C.strain,physics:C.mid,emerge:C.green,warn:C.accent,init:C.dim};
  return(
    <div ref={ref} style={{height:"100%",overflowY:"auto",fontFamily:MONO,fontSize:8.5,lineHeight:1.85,paddingRight:2}}>
      {entries.map((e,i)=>(
        <div key={i} style={{color:TC[e.type]||C.dim}}>
          <span style={{color:"#141428",marginRight:6}}>[{String(e.iter).padStart(4,"0")}]</span>
          {e.msg}
        </div>
      ))}
    </div>
  );
}

// ── BAR ───────────────────────────────────────────────────────────────────────
function Bar({v,col,h=4}){
  return(
    <div style={{background:"#0a0a1c",height:h,borderRadius:2,overflow:"hidden",marginTop:2}}>
      <div style={{width:`${clamp(v*100,0,100)}%`,height:"100%",
        background:`linear-gradient(90deg,${col}44,${col})`,
        transition:"width 0.3s",boxShadow:`0 0 4px ${col}22`}}/>
    </div>
  );
}

function Panel({title,children,style={}}){
  return(
    <div style={{background:C.surface,border:`1px solid ${C.border}`,
      borderRadius:6,padding:"10px 12px",...style}}>
      {title&&<div style={{fontSize:7.5,fontFamily:MONO,color:C.dimAcc,letterSpacing:2.5,
        marginBottom:8,textTransform:"uppercase",borderBottom:`1px solid ${C.border}`,paddingBottom:4}}>
        {title}
      </div>}
      {children}
    </div>
  );
}

// ── SIMULATION STATE ──────────────────────────────────────────────────────────
function initSim(levers){
  return{
    particles:initParticles(),
    prior:new Float32Array(N*3).fill(1/3),
    iter:0, lambda:LAMBDA0,
    metrics:{elbo:0,kl:0,H_q:Math.log(3),conv:0},
    macro:{muB:0,muZ:0,muS:0,sigB:1},
    levers:{...levers},
    reified:false, schemaStep:0,
    elboH:[], klH:[], hH:[], convH:[],
    efeLog:[
      {iter:0,type:"init",msg:`N=${N} elements · q(ωᵢ)=Uniform(⅓,⅓,⅓) · H=log3=${Math.log(3).toFixed(3)}`},
      {iter:0,type:"init",msg:`Torus: R_major=${R_MAJOR} R_minor=${R_MINOR} · specified as prior only`},
      {iter:0,type:"init",msg:"All points GREY — no certainty, no blanket yet"},
      {iter:0,type:"warn",msg:"Press ▶ RUN — color will emerge as q(ω) sharpens"},
    ],
  };
}

// ── MAIN APP ──────────────────────────────────────────────────────────────────
export default function App(){
  const [tick,    setTick]    = useState(0);
  const [running, setRunning] = useState(false);
  const [levers,  setLevers]  = useState({precision:0.55,curiosity:0.35,embedding:0.50});

  const canvasRef = useRef();
  const simRef    = useRef(null);
  const angleRef  = useRef(0);
  const animRef   = useRef();
  const ivRef     = useRef();

  useEffect(()=>{simRef.current=initSim(levers);setTick(1);},[]);

  // Render loop
  useEffect(()=>{
    const loop=()=>{
      animRef.current=requestAnimationFrame(loop);
      angleRef.current+=0.005;
      const sim=simRef.current;
      if(sim&&canvasRef.current){
        render3D(canvasRef.current,sim.particles,sim.particles.q,angleRef.current,sim.iter);
      }
    };
    loop();
    return()=>cancelAnimationFrame(animRef.current);
  },[]);

  // Sync levers to sim
  useEffect(()=>{
    if(simRef.current) simRef.current.levers={...levers};
  },[levers]);

  const step=useCallback(()=>{
    const sim=simRef.current; if(!sim)return;
    sim.iter++;
    const it=sim.iter;

    // 1. Prior
    sim.prior=computePrior(
      sim.particles.px,sim.particles.py,sim.particles.pz,
      sim.metrics.conv, sim.levers.precision
    );

    // 2. λ(t) [Eq. 13]
    sim.lambda=LAMBDA0*(1+it*ETA_LAM)*Math.max(0.08,1-sim.metrics.conv*0.65);

    // 3. ATTEND + INFER [Eq. 26–27]
    sim.macro=attendStep(
      sim.particles.px,sim.particles.py,sim.particles.pz,
      sim.particles.q, sim.prior, sim.lambda
    );

    // 4. EFE forces + physics
    physicsStep(sim.particles,sim.prior,sim.particles.q,sim.lambda,sim.levers,it);

    // 5. Metrics
    sim.metrics=computeMetrics(
      sim.particles.q,sim.prior,
      sim.particles.px,sim.particles.py,sim.particles.pz
    );
    sim.reified=sim.metrics.conv>0.68&&it>30;

    // History
    const MAX=160;
    const push=(arr,v)=>{arr.push(v);if(arr.length>MAX)arr.shift();};
    push(sim.elboH,sim.metrics.elbo);
    push(sim.klH,  sim.metrics.kl);
    push(sim.hH,   sim.metrics.H_q);
    push(sim.convH,sim.metrics.conv);

    // Schema step
    sim.schemaStep=sim.reified?5:((it-1)%5)+1;

    // Log
    const L=sim.efeLog;
    if(it===1)  L.push({iter:it,type:"attend",msg:"ATTEND [Eq.26]: q(ω)←softmax[logP(y|ω,macro)+λ·logP*(ω)]"});
    if(it===2)  L.push({iter:it,type:"infer", msg:"INFER [Eq.27]: d̄_B=E_{q_B}[d_torus(x)]  σ_B=Std[d]"});
    if(it===3)  L.push({iter:it,type:"efe",   msg:"EFE: G(π)=KL[q‖p*]−β·H[q]  →  F_B=−d·λG·q_B·∇d"});
    if(it===4)  L.push({iter:it,type:"physics",msg:"PHYSICS: v←α·v+F·dt  x←x+v·dt+σ√dt·η  [Euler-Maruyama]"});
    if(it===8)  L.push({iter:it,type:"attend",msg:`Color emerges: cert=(max_k q−⅓)/(⅔) — watch grey→gold`});
    if(it===15) L.push({iter:it,type:"infer", msg:`d̄_B=${sim.macro.muB.toFixed(3)} — B elements moving to torus surface`});
    if(it%30===0&&it>0) L.push({iter:it,type:"efe",
      msg:`ELBO=${sim.metrics.elbo.toFixed(4)} KL=${sim.metrics.kl.toFixed(4)} H=${sim.metrics.H_q.toFixed(4)} conv=${(sim.metrics.conv*100).toFixed(1)}%`});
    if(sim.reified&&it%50===0) L.push({iter:it,type:"emerge",
      msg:"✓ Toroidal Markov blanket reified — partition co-converged"});
    if(L.length>100)L.shift();

    setTick(t=>t+1);
  },[]);

  // rAF-based loop — no interval gaps when levers change, no stale closures
  const runningRef=useRef(false);
  const lastStepTime=useRef(0);
  const STEP_MS=75;

  useEffect(()=>{ runningRef.current=running; },[running]);

  useEffect(()=>{
    let raf;
    const loop=(ts)=>{
      raf=requestAnimationFrame(loop);
      if(runningRef.current && ts-lastStepTime.current>=STEP_MS){
        lastStepTime.current=ts;
        step();
      }
    };
    raf=requestAnimationFrame(loop);
    return()=>cancelAnimationFrame(raf);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[]);

  const handleReset=()=>{
    clearInterval(ivRef.current);setRunning(false);
    simRef.current=initSim(levers);setTick(t=>t+1);
  };

  const sim=simRef.current;
  const iter=sim?.iter??0;
  const conv=sim?.metrics?.conv??0;
  const reified=sim?.reified??false;
  const lambda=sim?.lambda??LAMBDA0;
  const {elbo=0,kl=0,H_q=Math.log(3)}=sim?.metrics??{};
  const muB=sim?.macro?.muB??0;
  const schema=sim?.schemaStep??0;

  const q=sim?.particles?.q;
  const nB=q?Math.round(Array.from({length:N},(_,i)=>q[i*3+1]).reduce((s,v)=>s+v,0)):0;
  const nZ=q?Math.round(Array.from({length:N},(_,i)=>q[i*3+2]).reduce((s,v)=>s+v,0)):0;
  const nS=N-nB-nZ;

  return(
    <div style={{background:C.bg,minHeight:"100vh",color:C.text,
      fontFamily:MONO,padding:"12px 14px",boxSizing:"border-box",fontSize:12}}>

      {/* HEADER */}
      <div style={{marginBottom:10,borderBottom:`1px solid ${C.border}`,paddingBottom:8,
        display:"flex",alignItems:"baseline",gap:12,flexWrap:"wrap"}}>
        <h1 style={{margin:0,fontFamily:"Georgia,serif",fontSize:17,fontWeight:"normal",
          color:C.text,letterSpacing:0.5}}>Torus Emergence</h1>
        <span style={{color:C.dimAcc,fontSize:9}}>
          Inverted DMBD · Beck &amp; Ramstead (2025) · N={N} elements
        </span>
        <span style={{color:C.dim,fontSize:8.5,marginLeft:4}}>
          Color = certainty cert=(max q−⅓)/(⅔) · grey=uncertain · saturated=assigned
        </span>
        <div style={{marginLeft:"auto",display:"flex",gap:8,alignItems:"center"}}>
          {reified&&<span style={{fontSize:9,color:C.green,
            border:`1px solid ${C.green}44`,padding:"2px 8px",borderRadius:2,letterSpacing:1.5}}>
            ✓ BLANKET REIFIED</span>}
          <span style={{fontSize:9,color:C.accent,
            border:`1px solid ${C.dimAcc}44`,padding:"2px 8px",borderRadius:2}}>
            iter {iter}</span>
        </div>
      </div>

      {/* MAIN ROW */}
      <div style={{display:"grid",gridTemplateColumns:"1fr 270px 240px",gap:11,marginBottom:11}}>

        {/* 3D CANVAS */}
        <Panel title={null} style={{padding:0,overflow:"hidden"}}>
          <div style={{padding:"6px 12px",borderBottom:`1px solid ${C.border}`,
            display:"flex",justifyContent:"space-between",alignItems:"center",flexWrap:"wrap",gap:6}}>
            <span style={{fontSize:8,color:C.dimAcc,letterSpacing:1.5}}>3D PROJECTION — auto-rotating</span>
            <div style={{display:"flex",gap:14,fontSize:8}}>
              {[["B — Blanket (shell)",C.B,nB],["Z — Internal",C.Z,nZ],["S — External",C.S,nS]].map(([l,col,n])=>(
                <span key={l} style={{color:col}}>● {l} ({n})</span>
              ))}
            </div>
          </div>
          <canvas ref={canvasRef} width={540} height={460}
            style={{display:"block",width:"100%",background:C.bg}}/>
          {/* Controls */}
          <div style={{padding:"8px 12px",borderTop:`1px solid ${C.border}`,
            display:"flex",gap:8,alignItems:"center",flexWrap:"wrap"}}>
            <button onClick={()=>setRunning(r=>!r)} style={{
              padding:"6px 14px",fontFamily:MONO,fontSize:9,cursor:"pointer",
              background:running?`${C.S}18`:`${C.green}18`,
              border:`1px solid ${running?C.S:C.green}`,
              color:running?C.S:C.green,borderRadius:3}}>
              {running?"■ PAUSE":"▶ RUN"}
            </button>
            <button onClick={step} disabled={running} style={{
              padding:"6px 10px",fontFamily:MONO,fontSize:9,cursor:"pointer",
              background:"transparent",border:`1px solid ${C.border}`,
              color:running?"#141428":C.mid,borderRadius:3}}>STEP</button>
            <button onClick={handleReset} style={{
              padding:"6px 10px",fontFamily:MONO,fontSize:9,cursor:"pointer",
              background:"transparent",border:`1px solid ${C.border}`,
              color:C.mid,borderRadius:3}}>RESET</button>
            <div style={{marginLeft:"auto",padding:"4px 10px",
              background:reified?`${C.green}10`:C.border,
              border:`1px solid ${reified?C.green:C.border}`,
              borderRadius:3,fontSize:8.5,color:reified?C.green:C.dim}}>
              {reified?"✓ Toroidal blanket reified":"◌ Blanket forming..."}
            </div>
          </div>
        </Panel>

        {/* SCHEMA */}
        <Panel title="Inversion Schema — DMBD⁻¹">
          <Schema schemaStep={schema} lambda={lambda} elbo={elbo}
            kl={kl} Hq={H_q} conv={conv} muB={muB} levers={levers}/>
          {/* Convergence */}
          <div style={{marginTop:10,borderTop:`1px solid ${C.border}`,paddingTop:8}}>
            <div style={{display:"flex",justifyContent:"space-between",marginBottom:2}}>
              <span style={{fontSize:8.5,color:C.mid}}>Radial conv.</span>
              <span style={{fontSize:8.5,color:C.accent}}>{(conv*100).toFixed(1)}%</span>
            </div>
            <Bar v={conv} col={C.accent}/>
            <div style={{display:"flex",justifyContent:"space-between",marginTop:6,marginBottom:2}}>
              <span style={{fontSize:8.5,color:C.mid}}>d̄_B → 0</span>
              <span style={{fontSize:8.5,color:C.B}}>{muB.toFixed(4)}</span>
            </div>
            <Bar v={clamp(1-Math.abs(muB)/R_MINOR,0,1)} col={C.B}/>
          </div>
        </Panel>

        {/* LEVERS */}
        <div style={{display:"flex",flexDirection:"column",gap:0}}>
          <div style={{fontSize:7.5,fontFamily:MONO,color:C.dimAcc,letterSpacing:2.5,
            marginBottom:8,textTransform:"uppercase"}}>
            Three Design Levers
          </div>
          <LeverPanel levers={levers} setLevers={setLevers} running={running}/>
        </div>
      </div>

      {/* BOTTOM ROW */}
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr",gap:11}}>
        <Panel title="ELBO">
          <MiniChart history={sim?.elboH??[]} color={C.accent} label="ELBO=E_q[logP]−KL" tick={tick}/>
          <div style={{fontSize:7.5,color:C.dim,marginTop:3}}>Rising = structure crystallizing</div>
        </Panel>
        <Panel title="KL[q ‖ p*] — Risk">
          <MiniChart history={sim?.klH??[]} color={C.strain} label="KL[q‖p*]" tick={tick}/>
          <div style={{fontSize:7.5,color:C.dim,marginTop:3}}>Falling = q aligned with prior</div>
        </Panel>
        <Panel title="H[q] — Assignment Entropy">
          <MiniChart history={sim?.hH??[]} color={C.Z} label="H[q]=−Σqₖ·logqₖ" tick={tick}/>
          <div style={{fontSize:7.5,color:C.dim,marginTop:3}}>
            log3={Math.log(3).toFixed(3)} at start · falls as blanket sharpens
          </div>
        </Panel>
        <Panel title="Log">
          <div style={{height:90}}>
            <EFELog entries={sim?.efeLog??[]}/>
          </div>
        </Panel>
      </div>

      <div style={{marginTop:10,borderTop:`1px solid ${C.border}`,paddingTop:6,
        display:"flex",justifyContent:"space-between",fontSize:7.5,color:C.dim}}>
        <span>Beck &amp; Ramstead (2025) arXiv:2502.21217 · Torus: d(x)=√((√(x²+z²)−R)²+y²)−r</span>
        <span>p*(ω)∝exp(γ·propₖ) · G(π)=KL[q‖p*]−β·H[q] · cert=(max_k q−⅓)/(⅔)</span>
      </div>
    </div>
  );
}
