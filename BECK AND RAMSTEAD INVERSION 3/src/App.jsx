import { useState, useEffect, useRef, useCallback } from "react";

/* ═══════════════════════════════════════════════════════════════════════════
   TORUS EMERGENCE — Inverted DMBD  v4 (redesigned UI)
   Beck & Ramstead (2025) arXiv:2502.21217
   Physics & math unchanged. UI completely redesigned for clarity.
═══════════════════════════════════════════════════════════════════════════ */

const N=4000,R_MAJOR=2.6,R_MINOR=0.85,DT=0.018,DAMPING=0.80;
const R_EXCL=0.22,K_EXCL=0.55,LAMBDA0=0.52,ETA_LAM=0.004,GAMMA0=2.2;
const K_ROLLOUT=5,N_ROLLOUT=400,T_UPDATE=20;

/* ── DESIGN TOKENS ──────────────────────────────────────────────────────── */
const DC = {
  // Backgrounds
  bg:      "#050508",
  surface: "#0a0a14",
  card:    "#0e0e1c",
  hover:   "#13132a",
  // Borders
  border:  "#1a1a30",
  borderBright: "#2a2a48",
  // Roles — vivid, high contrast
  Z:  "#4a9eff",   // Internal  — sky blue
  Bs: "#ffd166",   // B sensory — bright amber
  Ba: "#c8860a",   // B active  — deep amber
  B:  "#f0b429",   // B combined
  S:  "#ff4d6d",   // External  — rose
  // System
  green:  "#22d87a",
  teal:   "#0ed2b0",
  purple: "#a78bfa",
  orange: "#fb923c",
  red:    "#f87171",
  // Text
  text:   "#e8e4da",
  sub:    "#8a8070",
  dim:    "#3a3830",
  muted:  "#5a5650",
  // Levers
  lev1: "#a78bfa",
  lev2: "#34d399",
  lev3: "#fb923c",
};
const SERIF = "Georgia, 'Times New Roman', serif";
const MONO  = "'JetBrains Mono', 'Fira Code', 'Courier New', monospace";
const SANS  = "'SF Pro Display', 'Helvetica Neue', Arial, sans-serif";

/* ── MATH UTILITIES ─────────────────────────────────────────────────────── */
function randn(){let u,v;do{u=Math.random();}while(!u);do{v=Math.random();}while(!v);return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
function clamp(x,a,b){return x<a?a:x>b?b:x;}
function xlogx(x){return x<1e-12?0:x*Math.log(x);}
function sigmoid(x){return 1/(1+Math.exp(-x));}
function distTorus(x,y,z){const r=Math.sqrt(x*x+z*z);return Math.sqrt((r-R_MAJOR)**2+y*y)-R_MINOR;}
function gradTorus(x,y,z){const r=Math.sqrt(x*x+z*z)+1e-9,dx=r-R_MAJOR,d=Math.sqrt(dx*dx+y*y)+1e-9;return{gx:(dx/d)*(x/r),gy:y/d,gz:(dx/d)*(z/r)};}
function isSensory(x,y,z){return Math.abs(Math.atan2(y,Math.sqrt(x*x+z*z)-R_MAJOR))<Math.PI/2;}

/* ═══════════════════════════════════════════════════════════════════════════
   GENERATIVE MODEL  P(o, s)
   ─────────────────────────────────────────────────────────────────────────
   Hidden state:   s = (x, ω)   position + role ∈ {S=0, B=1, Z=2}
   Observation:    o = d(x)     signed distance from torus surface
   Likelihood:     P(o | s) = N(μ_ω, σ_obs)
                   μ_S =  R_MINOR  (outside surface)
                   μ_B =  0        (on surface)
                   μ_Z = -R_MINOR  (inside tube)
   Preferences:    P*(o) = N(0, σ_pref)   system prefers d ≈ 0
   Transition:     P(s_τ | s_{τ-1}, a) = physics Euler-Maruyama step
   ═══════════════════════════════════════════════════════════════════════ */
const MU_OBS   = [R_MINOR, 0, -R_MINOR];   // [S, B, Z]
const SIG_OBS  = 0.30;                      // observation noise
const SIG_PREF = 0.28;                      // preference width around d=0

/* Log-likelihood P(o | ω=k, macro) */
function logLikelihood(d, muB_macro, sigB_macro){
  // Macro latents (INFER step [Eq.27]): mean & std of d per role
  // Returns [llS, llB, llZ]
  return [
    -0.5*((d-MU_OBS[0])/SIG_OBS)**2,
    -0.5*((d-muB_macro)/(sigB_macro+0.01))**2,
    -0.5*((d-MU_OBS[2])/SIG_OBS)**2,
  ];
}

/* Log-preference: log P*(o) = log N(0, σ_pref) */
function logPref(d){
  return -0.5*(d/SIG_PREF)**2;
}

/* ── INIT ───────────────────────────────────────────────────────────────── */
function initParticles(){
  const px=new Float32Array(N),py=new Float32Array(N),pz=new Float32Array(N);
  const vx=new Float32Array(N),vy=new Float32Array(N),vz=new Float32Array(N);
  const q=new Float32Array(N*3),qPrev=new Float32Array(N*3);
  for(let i=0;i<N;i++){
    px[i]=(Math.random()-0.5)*10;py[i]=(Math.random()-0.5)*10;pz[i]=(Math.random()-0.5)*10;
    vx[i]=randn()*0.03;vy[i]=randn()*0.03;vz[i]=randn()*0.03;
    q[i*3]=1/3;q[i*3+1]=1/3;q[i*3+2]=1/3;
    qPrev[i*3]=1/3;qPrev[i*3+1]=1/3;qPrev[i*3+2]=1/3;
  }
  return{px,py,pz,vx,vy,vz,q,qPrev};
}

/* ── DESIGNER PRIOR ─────────────────────────────────────────────────────── */
function computePrior(px,py,pz,conv,precision){
  const prior=new Float32Array(N*3),gamma=GAMMA0+4.0*precision;
  const sigB=clamp((1.05-conv*0.60-precision*0.25)*R_MINOR,0.12,R_MINOR*1.1);
  for(let i=0;i<N;i++){
    const d=distTorus(px[i],py[i],pz[i]),base=i*3;
    const propB=Math.exp(-0.5*(d/sigB)**2);
    const propZ=sigmoid(-d/0.50),propS=sigmoid(d/0.50);
    const eB=Math.exp(gamma*propB),eZ=Math.exp(gamma*propZ*0.72),eS=Math.exp(gamma*propS*0.55);
    const Z=eB+eZ+eS+1e-12;
    prior[base]=eS/Z;prior[base+1]=eB/Z;prior[base+2]=eZ/Z;
  }
  return prior;
}

/* ── ATTEND + INFER [Eq. 26–27] ─────────────────────────────────────────── */
function attendStep(px,py,pz,q,prior,lambda){
  // INFER: macro latents — weighted mean & std of d per role
  let dB=0,wB=0,dZ=0,wZ=0,dS=0,wS=0;
  for(let i=0;i<N;i++){
    const d=distTorus(px[i],py[i],pz[i]);
    dB+=q[i*3+1]*d;wB+=q[i*3+1];dZ+=q[i*3+2]*d;wZ+=q[i*3+2];dS+=q[i*3]*d;wS+=q[i*3];
  }
  const muB=dB/(wB+1e-9),muZ=dZ/(wZ+1e-9),muS=dS/(wS+1e-9);
  let vB=0,vZ=0,vS=0;
  for(let i=0;i<N;i++){
    const d=distTorus(px[i],py[i],pz[i]);
    vB+=q[i*3+1]*(d-muB)**2;vZ+=q[i*3+2]*(d-muZ)**2;vS+=q[i*3]*(d-muS)**2;
  }
  const sigB=Math.sqrt(vB/(wB+1e-9))+0.18;
  const sigZ=Math.sqrt(vZ/(wZ+1e-9))+0.28,sigS=Math.sqrt(vS/(wS+1e-9))+0.28;
  // ATTEND [Eq.26]
  for(let i=0;i<N;i++){
    const d=distTorus(px[i],py[i],pz[i]),base=i*3;
    const llB=-0.5*((d-muB)/sigB)**2,llZ=-0.5*((d-muZ)/sigZ)**2,llS=-0.5*((d-muS)/sigS)**2;
    const lB=llB+lambda*Math.log(Math.max(prior[base+1],1e-10));
    const lZ=llZ+lambda*Math.log(Math.max(prior[base+2],1e-10));
    const lS=llS+lambda*Math.log(Math.max(prior[base],1e-10));
    const mx=Math.max(lB,lZ,lS);
    const eB=Math.exp(lB-mx),eZ=Math.exp(lZ-mx),eS=Math.exp(lS-mx),sm=eB+eZ+eS+1e-12;
    q[base]=eS/sm;q[base+1]=eB/sm;q[base+2]=eZ/sm;
  }
  return{muB,muZ,muS,sigB,sigZ,sigS};
}

/* ═══════════════════════════════════════════════════════════════════════════
   EFE FRISTONIANA — rigore pieno
   ─────────────────────────────────────────────────────────────────────────
   Per ogni elemento i del subsample, per ogni policy π ∈ {π_S, π_B, π_Z}:
   
   G(π) = Σ_{τ=1..k}  E_{q(o_τ,s_τ|π)} [ log q(s_τ|π) − log P(o_τ,s_τ) ]
   
   Scomposto (Smith et al. 2020b):
   
   G(π) = Risk(π) + Ambiguity(π) − InformationGain(π)
   
   Risk(π)           = KL[ q(o_τ|π) ‖ P*(o) ]
                     = E_{q(o|π)}[ log q(o|π) − log P*(o) ]
   
   Ambiguity(π)      = E_{q(s|π)}[ H[ P(o|s) ] ]
                     = E_{q(s|π)}[ 0.5·log(2πe·σ_obs²) ]   (Gaussian, constant)
   
   InformationGain(π) = E_{q(o|π)}[ KL[ q(s|o,π) ‖ q(s|π) ] ]
                      ≈ H[ q(s|π) ] − E_{q(o|π)}[ H[ q(s|o,π) ] ]
                      = epistemic value of policy π
   
   La forza applicata a i è la mixture pesata da q(ω):
   F_i = Σ_k q(ωᵢ=k) · G(π_k) · ∇_role_k
   
   Il termine InformationGain è la differenza chiave rispetto al k-step
   precedente: spinge gli elementi incerti verso zone dove la likelihood
   P(o|s) separa meglio i ruoli (cioè la superficie del toro).
   ═══════════════════════════════════════════════════════════════════════ */
function computeEFE_Friston(particles, prior, q, lambda, levers, iter, macro){
  const{px,py,pz,vx,vy,vz}=particles;
  const{precision,curiosity,embedding}=levers;
  const{muB,sigB}=macro;
  const beta = 0.18 + curiosity*0.60;  // epistemic weight (Curiosity Sculpting)
  const forceScale = 0.20 + precision*0.35;
  const sigma = clamp((0.040-precision*0.010)*(1-embedding*0.68)*Math.exp(-iter*0.0018*embedding)+0.006,0.005,0.06);

  // G per ogni elemento (inizializzato con EFE istantanea per i non-subsample)
  const G_elem = new Float32Array(N);

  // ── Istantanea per tutti (fallback) ──────────────────────────────────────
  for(let i=0;i<N;i++){
    const base=i*3,qS=q[base],qB=q[base+1],qZ=q[base+2];
    const pS=prior[base],pB=prior[base+1],pZ=prior[base+2];
    // Risk = KL[q‖p*]
    const risk = qS*Math.log(Math.max(qS/(pS+1e-9),1e-9))
               + qB*Math.log(Math.max(qB/(pB+1e-9),1e-9))
               + qZ*Math.log(Math.max(qZ/(pZ+1e-9),1e-9));
    // Epistemic value = H[q] (proxy per IG istantanea)
    const H = -(xlogx(qS)+xlogx(qB)+xlogx(qZ));
    G_elem[i] = clamp(risk - beta*H, 0, 6);
  }

  // ── Rollout Fristoniano su subsample ─────────────────────────────────────
  const stride = Math.floor(N / N_ROLLOUT);

  for(let si=0;si<N_ROLLOUT;si++){
    const i = si*stride;
    const base = i*3;

    // Stato iniziale
    let rx=px[i],ry=py[i],rz=pz[i],rvx=vx[i],rvy=vy[i],rvz=vz[i];
    let qS0=q[base],qB0=q[base+1],qZ0=q[base+2];

    // Per ogni policy π_k: simula k passi con la forza del ruolo k
    // G(π_k) = Σ_τ [ Risk_τ + Ambiguity_τ − IG_τ ]
    const G_policy = new Float32Array(3); // G per π_S, π_B, π_Z

    for(let pk=0;pk<3;pk++){
      let prx=rx,pry=ry,prz=rz,prvx=rvx,prvy=rvy,prvz=rvz;
      let prqS=qS0,prqB=qB0,prqZ=qZ0;
      let Gsum=0;

      for(let tau=1;tau<=K_ROLLOUT;tau++){
        const d=distTorus(prx,pry,prz);
        const{gx,gy,gz}=gradTorus(prx,pry,prz);
        // Forza specifica della policy pk
        let fx=0,fy=0,fz=0;
        const fmag=lambda*G_elem[i]*forceScale;
        if(pk===1){const f=-d*fmag*1.8;fx+=f*gx;fy+=f*gy;fz+=f*gz;}       // π_B: verso superficie
        else if(pk===2){const f=fmag*0.55;fx-=f*gx;fy-=f*gy;fz-=f*gz;}    // π_Z: verso interno
        else{const f=fmag*0.40;fx+=f*gx;fy+=f*gy;fz+=f*gz;}                // π_S: verso esterno
        fx+=randn()*sigma*0.5;fy+=randn()*sigma*0.5;fz+=randn()*sigma*0.5;
        prvx=prvx*DAMPING+fx*DT;prvy=prvy*DAMPING+fy*DT;prvz=prvz*DAMPING+fz*DT;
        prx+=prvx*DT;pry+=prvy*DT;prz+=prvz*DT;

        // Aggiorna q_τ via likelihood (senza macro update per velocità)
        const d2=distTorus(prx,pry,prz),base2=i*3;
        const llB2=-0.5*((d2-muB)/(sigB+0.01))**2;
        const llZ2=-0.5*((d2-MU_OBS[2])/SIG_OBS)**2;
        const llS2=-0.5*((d2-MU_OBS[0])/SIG_OBS)**2;
        const lB2=llB2+lambda*Math.log(Math.max(prior[base2+1],1e-10));
        const lZ2=llZ2+lambda*Math.log(Math.max(prior[base2+2],1e-10));
        const lS2=llS2+lambda*Math.log(Math.max(prior[base2],1e-10));
        const mx2=Math.max(lB2,lZ2,lS2);
        const eB2=Math.exp(lB2-mx2),eZ2=Math.exp(lZ2-mx2),eS2=Math.exp(lS2-mx2),sm2=eB2+eZ2+eS2+1e-12;
        prqS=eS2/sm2;prqB=eB2/sm2;prqZ=eZ2/sm2;

        // ── RISK(π_k, τ): KL[ q(o_τ|π) ‖ P*(o) ] ──────────────────────────
        // q(o_τ|π) ≈ N(E[d_τ], SIG_OBS²) marginalized over q(s_τ)
        // E[d_τ|π] = Σ_k q_k · μ_obs_k
        const E_d = prqS*MU_OBS[0] + prqB*d2 + prqZ*MU_OBS[2];
        // KL between two Gaussians: KL[N(μ1,σ²) ‖ N(0,σ_pref²)]
        const risk_tau = 0.5*((SIG_OBS/SIG_PREF)**2 + (E_d/SIG_PREF)**2 - 1 + 2*Math.log(SIG_PREF/SIG_OBS));

        // ── AMBIGUITY(π_k, τ): E[H[P(o|s)]] ───────────────────────────────
        // P(o|s) = N(μ_ω, σ_obs²)  →  H = 0.5·log(2πe·σ_obs²) = constant
        const ambiguity_tau = 0.5*Math.log(2*Math.PI*Math.E*SIG_OBS*SIG_OBS);

        // ── INFORMATION GAIN(π_k, τ) ────────────────────────────────────────
        // IG ≈ H[q(s_τ|π)] − E_{q(o_τ|π)}[ H[q(s_τ|o_τ,π)] ]
        // H[q(s_τ|π)] = current entropy of role beliefs
        const H_prior_tau = -(xlogx(prqS)+xlogx(prqB)+xlogx(prqZ));
        // E[H[q(s|o,π)]]: posterior entropy after observing o
        // Approximate by sampling 3 observation values (one per role)
        let H_post_avg = 0;
        for(let obs_k=0;obs_k<3;obs_k++){
          const o_sample = MU_OBS[obs_k]; // representative observation
          // Posterior q(s|o_sample, π) via Bayes
          const llS_p=-0.5*((o_sample-MU_OBS[0])/SIG_OBS)**2;
          const llB_p=-0.5*((o_sample-muB)/(sigB+0.01))**2;
          const llZ_p=-0.5*((o_sample-MU_OBS[2])/SIG_OBS)**2;
          const mxp=Math.max(llS_p,llB_p,llZ_p);
          const eS_p=Math.exp(llS_p-mxp)*prqS,eB_p=Math.exp(llB_p-mxp)*prqB,eZ_p=Math.exp(llZ_p-mxp)*prqZ;
          const smp=eS_p+eB_p+eZ_p+1e-12;
          const ps=eS_p/smp,pb=eB_p/smp,pz2=eZ_p/smp;
          const H_post=-(xlogx(ps)+xlogx(pb)+xlogx(pz2));
          H_post_avg+=H_post*(obs_k===1?prqB:obs_k===0?prqS:prqZ); // weighted by q
        }
        const IG_tau = Math.max(0, H_prior_tau - H_post_avg);

        // G(π_k) accumulation
        Gsum += clamp(risk_tau + ambiguity_tau - beta*IG_tau, 0, 8);
      }
      G_policy[pk] = Gsum / K_ROLLOUT;
    }

    // Mixture EFE: G_i = Σ_k q(ω=k) · G(π_k)
    G_elem[i] = q[base]*G_policy[0] + q[base+1]*G_policy[1] + q[base+2]*G_policy[2];
  }

  return G_elem;
}

/* ── PHYSICS ────────────────────────────────────────────────────────────── */
function physicsStep(particles,prior,q,lambda,levers,iter,G_elem){
  const{px,py,pz,vx,vy,vz}=particles;
  const{precision,embedding}=levers;
  const sigma=clamp((0.040-precision*0.010)*(1-embedding*0.68)*Math.exp(-iter*0.0018*embedding)+0.006,0.005,0.06);
  const forceScale=0.20+precision*0.35;
  const fx=new Float32Array(N),fy=new Float32Array(N),fz=new Float32Array(N);
  // Stochastic repulsion
  for(let s=0;s<6000;s++){
    const i=Math.floor(Math.random()*N),j=Math.floor(Math.random()*N);
    if(i===j)continue;
    const dx=px[i]-px[j],dy=py[i]-py[j],dz=pz[i]-pz[j],d2=dx*dx+dy*dy+dz*dz;
    if(d2<R_EXCL*R_EXCL&&d2>1e-8){const d=Math.sqrt(d2),f=K_EXCL*(R_EXCL-d)/d;fx[i]+=f*dx;fy[i]+=f*dy;fz[i]+=f*dz;fx[j]-=f*dx;fy[j]-=f*dy;fz[j]-=f*dz;}
  }
  // EFE forces (mixture-weighted)
  for(let i=0;i<N;i++){
    const base=i*3,qS=q[base],qB=q[base+1],qZ=q[base+2];
    const G=G_elem[i],d=distTorus(px[i],py[i],pz[i]);
    const{gx,gy,gz}=gradTorus(px[i],py[i],pz[i]);
    const fmag=lambda*G*forceScale;
    if(qB>0.15){const f=-d*fmag*(qB*1.8);fx[i]+=f*gx;fy[i]+=f*gy;fz[i]+=f*gz;}
    if(qZ>0.15){const f=fmag*(qZ*0.55);fx[i]-=f*gx;fy[i]-=f*gy;fz[i]-=f*gz;}
    if(qS>0.15){const f=fmag*(qS*0.40);fx[i]+=f*gx;fy[i]+=f*gy;fz[i]+=f*gz;}
    fx[i]+=randn()*sigma;fy[i]+=randn()*sigma;fz[i]+=randn()*sigma;
  }
  // Euler-Maruyama
  for(let i=0;i<N;i++){
    vx[i]=vx[i]*DAMPING+fx[i]*DT;vy[i]=vy[i]*DAMPING+fy[i]*DT;vz[i]=vz[i]*DAMPING+fz[i]*DT;
    px[i]+=vx[i]*DT;py[i]+=vy[i]*DT;pz[i]+=vz[i]*DT;
    const r=Math.sqrt(px[i]**2+py[i]**2+pz[i]**2);
    if(r>8){const s2=8/r;px[i]*=s2;py[i]*=s2;pz[i]*=s2;vx[i]*=-0.2;vy[i]*=-0.2;vz[i]*=-0.2;}
  }
}

/* ── CMI ────────────────────────────────────────────────────────────────── */
function computeCMI(q){
  let proxy=0;const pj=new Float32Array(9);
  for(let i=0;i<N;i++){
    const qS=q[i*3],qB=q[i*3+1],qZ=q[i*3+2];proxy+=qZ*qS;
    const qs=[qS,qB,qZ];for(let k=0;k<3;k++)for(let l=0;l<3;l++)pj[k*3+l]+=qs[k]*qs[l];
  }
  proxy/=N;for(let k=0;k<9;k++)pj[k]/=N;
  const pm=new Float32Array(3);for(let k=0;k<3;k++)for(let l=0;l<3;l++)pm[k]+=pj[k*3+l];
  let mi=0;for(const ki of[0,2])for(const li of[0,2]){const p=pj[ki*3+li],ind=pm[ki]*pm[li];if(p>1e-12&&ind>1e-12)mi+=p*Math.log(p/ind);}
  return{I_norm:clamp(9*proxy,0,1),mi_ZS:Math.max(0,mi)};
}

/* ── TRANSITION MATRIX ──────────────────────────────────────────────────── */
function computeTransitionMatrix(qPrev,qCurr){
  const TM=new Float32Array(9);
  for(let i=0;i<N;i++)for(let k=0;k<3;k++)for(let l=0;l<3;l++)TM[k*3+l]+=qPrev[i*3+k]*qCurr[i*3+l];
  for(let k=0;k<3;k++){let row=0;for(let l=0;l<3;l++)row+=TM[k*3+l];if(row>1e-9)for(let l=0;l<3;l++)TM[k*3+l]/=row;}
  return{TM,T_SZ:TM[2],T_ZS:TM[6]};
}

/* ── METRICS ────────────────────────────────────────────────────────────── */
function computeMetrics(q,prior,px,py,pz){
  let elbo=0,kl=0,H_q=0,wB=0,nBonTorus=0;
  for(let i=0;i<N;i++){
    for(let k=0;k<3;k++){const qi=Math.max(q[i*3+k],1e-10),pi=Math.max(prior[i*3+k],1e-10);elbo+=qi*(Math.log(pi)-Math.log(qi));kl+=qi*Math.log(qi/pi);H_q-=qi*Math.log(qi);}
    wB+=q[i*3+1];if(q[i*3+1]>0.45&&Math.abs(distTorus(px[i],py[i],pz[i]))<0.55)nBonTorus++;
  }
  return{elbo:elbo/N,kl:kl/N,H_q:H_q/N,conv:wB>10?nBonTorus/Math.max(wB,1):0};
}

/* ── 3D RENDERER ────────────────────────────────────────────────────────── */
function render3D(canvas,particles,q,angle,iter){
  const{px,py,pz}=particles;
  const ctx=canvas.getContext("2d"),W=canvas.width,H=canvas.height;
  ctx.fillStyle=DC.bg;ctx.fillRect(0,0,W,H);
  const scale=W*0.092,cx=W/2,cy=H/2,cosA=Math.cos(angle),sinA=Math.sin(angle);
  const tilt=0.35,cosT=Math.cos(tilt),sinT=Math.sin(tilt);
  function proj(x,y,z){const rx=x*cosA+z*sinA,ry=y,rz2=-x*sinA+z*cosA;const fx2=rx,fy2=ry*cosT-rz2*sinT,fz2=ry*sinT+rz2*cosT;const w=14/(14+fz2);return{sx:cx+fx2*scale*w,sy:cy-fy2*scale*w,z:fz2,w};}
  // Wireframe
  ctx.strokeStyle="rgba(74,158,255,0.065)";ctx.lineWidth=0.7;
  for(let phi=0;phi<360;phi+=15){const phr=phi*Math.PI/180;ctx.beginPath();for(let theta=0;theta<=360;theta+=6){const thr=theta*Math.PI/180;const x=(R_MAJOR+R_MINOR*Math.cos(thr))*Math.cos(phr),y=R_MINOR*Math.sin(thr),z=(R_MAJOR+R_MINOR*Math.cos(thr))*Math.sin(phr);const p=proj(x,y,z);theta===0?ctx.moveTo(p.sx,p.sy):ctx.lineTo(p.sx,p.sy);}ctx.stroke();}
  ctx.strokeStyle="rgba(240,180,41,0.055)";
  for(let theta=0;theta<360;theta+=20){const thr=theta*Math.PI/180;ctx.beginPath();for(let phi=0;phi<=360;phi+=5){const phr=phi*Math.PI/180;const x=(R_MAJOR+R_MINOR*Math.cos(thr))*Math.cos(phr),y=R_MINOR*Math.sin(thr),z=(R_MAJOR+R_MINOR*Math.cos(thr))*Math.sin(phr);const p=proj(x,y,z);phi===0?ctx.moveTo(p.sx,p.sy):ctx.lineTo(p.sx,p.sy);}ctx.stroke();}
  // Particles sorted by depth
  const pts=Array.from({length:N},(_,i)=>{const p=proj(px[i],py[i],pz[i]);return{i,sx:p.sx,sy:p.sy,z:p.z,w:p.w};}).sort((a,b)=>a.z-b.z);
  for(const{i,sx,sy,w}of pts){
    const qS=q[i*3],qB=q[i*3+1],qZ=q[i*3+2],maxQ=Math.max(qS,qB,qZ);
    const role=qS>=qB&&qS>=qZ?0:qB>=qZ?1:2;
    const cert=clamp((maxQ-1/3)/(2/3),0,1),cert2=cert*cert;
    const r_pt=clamp((0.8+cert*1.6)*w,0.4,3.2);
    if(cert2<0.03){ctx.fillStyle=`rgba(38,38,62,${0.10+cert*0.18})`;}
    else if(role===1){
      const sens=isSensory(px[i],py[i],pz[i]);
      if(cert>0.52){const glowR=r_pt*7,gc=sens?"rgba(255,209,102,":"rgba(200,134,10,";const g=ctx.createRadialGradient(sx,sy,0,sx,sy,glowR);g.addColorStop(0,gc+`${(cert-0.48)*0.20})`);g.addColorStop(1,gc+"0)");ctx.fillStyle=g;ctx.beginPath();ctx.arc(sx,sy,glowR,0,Math.PI*2);ctx.fill();}
      ctx.fillStyle=`hsla(${sens?43:36},${Math.round(cert2*82)}%,${Math.round(sens?15+cert2*32:9+cert2*16)}%,${0.10+cert2*0.84})`;
    } else {
      if(cert>0.52){const glowR=r_pt*5,gc=role===0?"rgba(255,77,109,":"rgba(74,158,255,";const g=ctx.createRadialGradient(sx,sy,0,sx,sy,glowR);g.addColorStop(0,gc+`${(cert-0.48)*0.16})`);g.addColorStop(1,gc+"0)");ctx.fillStyle=g;ctx.beginPath();ctx.arc(sx,sy,glowR,0,Math.PI*2);ctx.fill();}
      ctx.fillStyle=`hsla(${role===0?348:212},${Math.round(cert2*72)}%,${Math.round(9+cert2*24)}%,${0.10+cert2*0.84})`;
    }
    ctx.beginPath();ctx.arc(sx,sy,Math.max(r_pt,0.4),0,Math.PI*2);ctx.fill();
  }
  // Vignette
  const vg=ctx.createRadialGradient(cx,cy,H*0.28,cx,cy,H*0.82);vg.addColorStop(0,"transparent");vg.addColorStop(1,"rgba(5,5,8,0.52)");ctx.fillStyle=vg;ctx.fillRect(0,0,W,H);
  // Readout
  let totalH=0;for(let i=0;i<N;i++)totalH-=(xlogx(q[i*3])+xlogx(q[i*3+1])+xlogx(q[i*3+2]));
  ctx.fillStyle="rgba(90,86,80,0.65)";ctx.font=`10px ${MONO}`;
  ctx.fillText(`H[q] = ${(totalH/N).toFixed(3)}  /  log3 = ${Math.log(3).toFixed(3)}   iter ${iter}`,12,H-12);
}

/* ── MINI CHART ─────────────────────────────────────────────────────────── */
function MiniChart({history,color,label,tick,target,description}){
  const wrapRef=useRef(),canvasRef=useRef();
  const draw=useCallback(()=>{
    const wrap=wrapRef.current,c=canvasRef.current;if(!wrap||!c)return;
    const W=wrap.clientWidth||280,H=96;if(c.width!==W)c.width=W;if(c.height!==H)c.height=H;
    const ctx=c.getContext("2d");ctx.fillStyle=DC.card;ctx.fillRect(0,0,W,H);
    if(!history||history.length<2){ctx.fillStyle=DC.muted;ctx.font=`10px ${MONO}`;ctx.textAlign="center";ctx.fillText("waiting...",W/2,H/2+4);ctx.textAlign="left";return;}
    const pad={t:8,r:10,b:18,l:46},iw=W-pad.l-pad.r,ih=H-pad.t-pad.b;
    const mn=Math.min(...history),mx=Math.max(...history),rng=mx-mn||0.001;
    const sx=i2=>pad.l+(i2/(history.length-1))*iw,sy=v=>pad.t+(1-(v-mn)/rng)*ih;
    for(let i=0;i<=4;i++){const y=pad.t+(i/4)*ih;ctx.strokeStyle=DC.border;ctx.lineWidth=0.5;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(pad.l+iw,y);ctx.stroke();ctx.fillStyle=DC.muted;ctx.font=`8.5px ${MONO}`;ctx.textAlign="right";ctx.fillText((mx-(i/4)*rng).toFixed(3),pad.l-4,y+3);ctx.textAlign="left";}
    if(target!==undefined){const ty=sy(clamp(target,mn,mx));ctx.strokeStyle=DC.green+"55";ctx.lineWidth=1;ctx.setLineDash([3,5]);ctx.beginPath();ctx.moveTo(pad.l,ty);ctx.lineTo(pad.l+iw,ty);ctx.stroke();ctx.setLineDash([]);}
    const gr=ctx.createLinearGradient(0,pad.t,0,pad.t+ih);gr.addColorStop(0,color+"33");gr.addColorStop(1,color+"04");
    ctx.fillStyle=gr;ctx.beginPath();ctx.moveTo(sx(0),sy(history[0]));history.forEach((v,i2)=>ctx.lineTo(sx(i2),sy(v)));ctx.lineTo(sx(history.length-1),pad.t+ih);ctx.lineTo(sx(0),pad.t+ih);ctx.closePath();ctx.fill();
    ctx.strokeStyle=color;ctx.lineWidth=1.8;ctx.lineJoin="round";ctx.beginPath();history.forEach((v,i2)=>i2===0?ctx.moveTo(sx(i2),sy(v)):ctx.lineTo(sx(i2),sy(v)));ctx.stroke();
    const last=history[history.length-1];const lx=sx(history.length-1),ly=sy(last);ctx.fillStyle=color;ctx.beginPath();ctx.arc(lx,ly,2.5,0,Math.PI*2);ctx.fill();
    ctx.font=`bold 9px ${MONO}`;ctx.textAlign="right";ctx.fillText(last.toFixed(4),W-pad.r,clamp(ly-5,10,H-6));ctx.textAlign="left";
    ctx.fillStyle=DC.sub;ctx.font=`8.5px ${MONO}`;ctx.fillText(label,pad.l+3,pad.t+10);
  },[history,color,label,target]);
  useEffect(()=>{draw();},[tick,draw]);
  useEffect(()=>{const ro=new ResizeObserver(()=>draw());if(wrapRef.current)ro.observe(wrapRef.current);return()=>ro.disconnect();},[draw]);
  return(<div ref={wrapRef} style={{width:"100%"}}><canvas ref={canvasRef} style={{display:"block",width:"100%",height:"96px",borderRadius:"4px"}}/>{description&&<p style={{margin:"4px 0 0",fontSize:10,color:DC.muted,fontFamily:MONO}}>{description}</p>}</div>);
}

/* ── T MATRIX DISPLAY ───────────────────────────────────────────────────── */
function TMatrixDisplay({TM,T_SZ,T_ZS}){
  const roles=["S","B","Z"],ok=T_SZ<0.05&&T_ZS<0.05;
  return(<div style={{fontFamily:MONO}}>
    {!TM&&<p style={{color:DC.muted,fontSize:10,margin:0}}>Computing every {T_UPDATE} iterations…</p>}
    {TM&&<>
      <div style={{display:"grid",gridTemplateColumns:"20px repeat(3,1fr)",gap:3,marginBottom:10}}>
        <div/>{roles.map(r=><div key={r} style={{textAlign:"center",fontSize:9,color:DC.sub,letterSpacing:1}}>{r}</div>)}
        {roles.map((fr,k)=>[
          <div key={"r"+k} style={{fontSize:9,color:DC.sub,display:"flex",alignItems:"center"}}>{fr}</div>,
          ...roles.map((_,l)=>{const v=TM[k*3+l],isV=(k===0&&l===2)||(k===2&&l===0);return(<div key={k+""+l} style={{background:isV?`rgba(248,113,113,${v*0.5})`:`rgba(74,158,255,${v*0.35})`,border:`1px solid ${isV?DC.red+"44":DC.border}`,borderRadius:3,padding:"4px 2px",textAlign:"center",fontSize:10,color:isV?DC.red:DC.text,fontWeight:isV?"bold":"normal"}}>{v.toFixed(3)}</div>);})])}
      </div>
      <div style={{display:"flex",gap:8,marginBottom:8}}>
        {[["T_SZ",T_SZ],["T_ZS",T_ZS]].map(([label,val])=>(<div key={label} style={{flex:1}}><div style={{display:"flex",justifyContent:"space-between",fontSize:10,marginBottom:3}}><span style={{color:DC.red}}>{label}</span><span style={{color:val<0.05?DC.green:DC.red,fontWeight:"bold"}}>{val.toFixed(4)}</span></div><div style={{background:DC.bg,height:5,borderRadius:3}}><div style={{width:`${clamp(val*200,0,100)}%`,height:"100%",background:`linear-gradient(90deg,${DC.red}55,${DC.red})`,borderRadius:3,transition:"width 0.4s"}}/></div></div>))}
      </div>
      <div style={{padding:"6px 10px",borderRadius:4,fontSize:10,background:ok?`${DC.green}12`:`${DC.red}08`,border:`1px solid ${ok?DC.green+"44":DC.red+"22"}`,color:ok?DC.green:DC.muted}}>
        {ok?"✓  T_SZ ≈ T_ZS ≈ 0  —  Eq. 21 satisfied":"◌  Waiting for T_SZ, T_ZS → 0  [Eq. 21]"}
      </div>
    </>}
  </div>);
}

/* ── STAT PILL ──────────────────────────────────────────────────────────── */
function Stat({label,value,color,unit="",sub}){
  return(<div style={{background:DC.card,border:`1px solid ${DC.border}`,borderRadius:6,padding:"10px 14px"}}>
    <div style={{fontSize:9,color:DC.sub,fontFamily:MONO,letterSpacing:1.5,textTransform:"uppercase",marginBottom:4}}>{label}</div>
    <div style={{fontSize:22,fontFamily:MONO,color:color||DC.text,lineHeight:1,fontWeight:"bold"}}>{value}<span style={{fontSize:11,color:DC.muted,marginLeft:3}}>{unit}</span></div>
    {sub&&<div style={{fontSize:9,color:DC.muted,fontFamily:MONO,marginTop:3}}>{sub}</div>}
  </div>);
}

/* ── PROGRESS BAR ───────────────────────────────────────────────────────── */
function ProgressBar({value,color,label}){
  return(<div><div style={{display:"flex",justifyContent:"space-between",fontSize:10,fontFamily:MONO,marginBottom:4}}><span style={{color:DC.sub}}>{label}</span><span style={{color,fontWeight:"bold"}}>{(value*100).toFixed(1)}%</span></div><div style={{background:DC.bg,height:6,borderRadius:4,overflow:"hidden"}}><div style={{width:`${clamp(value*100,0,100)}%`,height:"100%",background:`linear-gradient(90deg,${color}66,${color})`,borderRadius:4,transition:"width 0.5s ease"}}/></div></div>);
}

/* ── LEVER ──────────────────────────────────────────────────────────────── */
function Lever({label,color,equation,tradeoff,value,onChange}){
  return(<div style={{background:DC.card,border:`1px solid ${DC.border}`,borderRadius:8,padding:"12px 14px"}}>
    <div style={{display:"flex",justifyContent:"space-between",alignItems:"baseline",marginBottom:8}}><span style={{fontSize:12,color,fontFamily:SANS,fontWeight:600}}>{label}</span><span style={{fontSize:14,color,fontFamily:MONO,fontWeight:"bold"}}>{value.toFixed(2)}</span></div>
    <input type="range" min={0} max={1} step={0.05} value={value} onChange={e=>onChange(parseFloat(e.target.value))} style={{width:"100%",accentColor:color,marginBottom:8,height:4}}/>
    <div style={{fontSize:9.5,fontFamily:MONO,color:DC.sub,lineHeight:1.6}}><span style={{color:color+"cc"}}>{equation}</span><br/><span style={{color:DC.muted,fontStyle:"italic"}}>{tradeoff}</span></div>
  </div>);
}

/* ── SCHEMA DIAGRAM ─────────────────────────────────────────────────────── */
function SchemaDiagram({step,lambda,kl,Hq,muB,levers,I_norm}){
  const ref=useRef();
  useEffect(()=>{
    const c=ref.current;if(!c)return;const ctx=c.getContext("2d"),W=c.width,H=c.height;ctx.fillStyle=DC.card;ctx.fillRect(0,0,W,H);
    const steps=[
      {icon:"①",label:"DESIGNER PRIOR",eq:"p*(ω) ∝ exp(γ·prop_k(d))",val:`γ=${(GAMMA0+4*levers.precision).toFixed(1)}`,col:DC.lev1},
      {icon:"②",label:"ATTEND [Eq.26]",eq:"q(ω) ← softmax[ℓℓ + λ·log p*]",val:`λ=${lambda.toFixed(3)}`,col:DC.B},
      {icon:"③",label:"INFER  [Eq.27]",eq:"d̄_B = E_{q_B}[d(x)],  σ_B = Std[d]",val:`d̄=${muB.toFixed(3)}`,col:DC.teal},
      {icon:"④",label:"EFE Fristoniana",eq:"G(π)=Risk+Ambig−β·IG  k="+K_ROLLOUT,val:`KL=${kl.toFixed(3)}`,col:DC.orange},
      {icon:"⑤",label:"PHYSICS (E-M)",eq:"v←αv+F·dt,  x←x+v·dt+ση",val:`H=${Hq.toFixed(3)}`,col:DC.sub},
      {icon:"⑥",label:"I(Z;S|B) → 0",eq:"CMI check — blanket verified",val:`I=${I_norm.toFixed(3)}`,col:DC.purple},
    ];
    const bW=W-28,bH=38,startY=14,gap=8,current=step%6;
    steps.forEach(({icon,label,eq,val,col},idx)=>{
      const bx=14,by=startY+idx*(bH+gap),isActive=idx===current,isPast=idx<current;
      ctx.fillStyle=isActive?col+"18":isPast?col+"08":DC.surface;ctx.strokeStyle=isActive?col:isPast?col+"44":DC.border;ctx.lineWidth=isActive?1.5:0.8;ctx.beginPath();ctx.roundRect(bx,by,bW,bH,5);ctx.fill();ctx.stroke();
      if(isActive){ctx.fillStyle=col;ctx.beginPath();ctx.arc(bx+10,by+bH/2,3.5,0,Math.PI*2);ctx.fill();}
      ctx.font=`bold 11px ${SANS}`;ctx.fillStyle=isActive?col:isPast?col+"99":DC.muted;ctx.fillText(icon,bx+20,by+14);
      ctx.font=`bold 9.5px ${MONO}`;ctx.fillStyle=isActive?col:isPast?col+"bb":DC.muted;ctx.fillText(label,bx+36,by+14);
      ctx.font=`8.5px ${MONO}`;ctx.fillStyle=isActive?DC.text:DC.dim;ctx.fillText(eq,bx+36,by+29);
      ctx.font=`bold 9px ${MONO}`;ctx.fillStyle=isActive?DC.green:DC.muted;ctx.textAlign="right";ctx.fillText(val,bx+bW-8,by+14);ctx.textAlign="left";
      if(idx<5){const ax=bx+bW/2,ay1=by+bH+2,ay2=by+bH+gap-2;ctx.strokeStyle=isActive?col+"66":DC.dim;ctx.lineWidth=1;ctx.setLineDash([2,4]);ctx.beginPath();ctx.moveTo(ax,ay1);ctx.lineTo(ax,ay2);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=isActive?col+"66":DC.dim;ctx.beginPath();ctx.moveTo(ax,ay2+2);ctx.lineTo(ax-4,ay2-4);ctx.lineTo(ax+4,ay2-4);ctx.closePath();ctx.fill();}
    });
    const x0=14+bW,yTop=startY+bH/2,yBot=startY+5*(bH+gap)+bH/2;ctx.strokeStyle=DC.teal+"44";ctx.lineWidth=1.5;ctx.setLineDash([3,6]);ctx.beginPath();ctx.moveTo(x0,yBot);ctx.lineTo(W-6,yBot);ctx.lineTo(W-6,yTop);ctx.lineTo(x0,yTop);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=DC.muted;ctx.font=`8px ${MONO}`;ctx.fillText("INVERSION SCHEMA  —  DMBD⁻¹ v5",14,8);
  },[step,lambda,kl,Hq,muB,levers,I_norm]);
  return<canvas ref={ref} width={290} height={310} style={{display:"block",width:"100%",borderRadius:6}}/>;
}

/* ── LOG ────────────────────────────────────────────────────────────────── */
function SimLog({entries}){
  const ref=useRef();useEffect(()=>{if(ref.current)ref.current.scrollTop=ref.current.scrollHeight;},[entries]);
  const cols={attend:DC.B,infer:DC.teal,efe:DC.orange,physics:DC.muted,emerge:DC.green,warn:DC.Bs,init:DC.sub,cmi:DC.purple,tmat:DC.red,friston:DC.orange};
  return(<div ref={ref} style={{height:"100%",overflowY:"auto",fontFamily:MONO,fontSize:9.5,lineHeight:2}}>{entries.length===0&&<span style={{color:DC.dim}}>—</span>}{entries.map((e,i)=>(<div key={i} style={{display:"flex",gap:8,color:cols[e.type]||DC.muted}}><span style={{color:DC.dim,flexShrink:0}}>[{String(e.iter).padStart(4,"0")}]</span><span>{e.msg}</span></div>))}</div>);
}

/* ── SIM STATE ──────────────────────────────────────────────────────────── */
function initSim(levers){
  return{
    particles:initParticles(),prior:new Float32Array(N*3).fill(1/3),
    iter:0,lambda:LAMBDA0,metrics:{elbo:0,kl:0,H_q:Math.log(3),conv:0},
    macro:{muB:0,sigB:0.3},levers:{...levers},reified:false,schemaStep:0,
    elboH:[],klH:[],hH:[],convH:[],cmiH:[],T_SZ_H:[],T_ZS_H:[],igH:[],
    cmi:{I_norm:1},tmat:{TM:null,T_SZ:1,T_ZS:1},meanIG:0,
    efeLog:[
      {iter:0,type:"init",msg:`N=${N}  ·  q = Uniform(⅓,⅓,⅓)  ·  H = log3 = ${Math.log(3).toFixed(4)}`},
      {iter:0,type:"init",msg:`Torus  R=${R_MAJOR}  r=${R_MINOR}  ·  EFE Fristoniana  k=${K_ROLLOUT}`},
      {iter:0,type:"friston",msg:"G(π) = Risk(π) + Ambiguity(π) − β·InformationGain(π)"},
      {iter:0,type:"friston",msg:"IG(π) = H[q(s|π)] − E_{q(o|π)}[H[q(s|o,π)]]  (epistemic value)"},
      {iter:0,type:"cmi", msg:"I(Z;S|B) starts at 1.0 — converges to 0 as blanket separates Z from S"},
      {iter:0,type:"warn",msg:"▶  Press RUN — colour emerges as q(ω) sharpens"},
    ],
  };
}

/* ── MAIN APP ───────────────────────────────────────────────────────────── */
export default function App(){
  const[tick,setTick]=useState(0);
  const[running,setRunning]=useState(false);
  const[levers,setLevers]=useState({precision:0.55,curiosity:0.35,embedding:0.50});
  const canvasRef=useRef(),simRef=useRef(null),angleRef=useRef(0),animRef=useRef();
  const runningRef=useRef(false),lastStep=useRef(0);

  useEffect(()=>{simRef.current=initSim(levers);setTick(1);},[]);
  useEffect(()=>{
    const loop=()=>{animRef.current=requestAnimationFrame(loop);angleRef.current+=0.005;const sim=simRef.current;if(sim&&canvasRef.current)render3D(canvasRef.current,sim.particles,sim.particles.q,angleRef.current,sim.iter);};
    loop();return()=>cancelAnimationFrame(animRef.current);
  },[]);
  useEffect(()=>{runningRef.current=running;},[running]);
  useEffect(()=>{if(simRef.current)simRef.current.levers={...levers};},[levers]);

  const step=useCallback(()=>{
    const sim=simRef.current;if(!sim)return;sim.iter++;const it=sim.iter;
    sim.particles.qPrev.set(sim.particles.q);
    sim.prior=computePrior(sim.particles.px,sim.particles.py,sim.particles.pz,sim.metrics.conv,sim.levers.precision);
    sim.lambda=LAMBDA0*(1+it*ETA_LAM)*Math.max(0.08,1-sim.metrics.conv*0.65);
    sim.macro=attendStep(sim.particles.px,sim.particles.py,sim.particles.pz,sim.particles.q,sim.prior,sim.lambda);
    // EFE Fristoniana
    const G_elem=computeEFE_Friston(sim.particles,sim.prior,sim.particles.q,sim.lambda,sim.levers,it,sim.macro);
    // Compute mean IG as diagnostic
    let igSum=0;for(let i=0;i<N;i++)igSum+=G_elem[i];sim.meanIG=igSum/N;
    physicsStep(sim.particles,sim.prior,sim.particles.q,sim.lambda,sim.levers,it,G_elem);
    sim.metrics=computeMetrics(sim.particles.q,sim.prior,sim.particles.px,sim.particles.py,sim.particles.pz);
    sim.reified=sim.metrics.conv>0.68&&it>30;
    sim.cmi=computeCMI(sim.particles.q);
    if(it%T_UPDATE===0)sim.tmat=computeTransitionMatrix(sim.particles.qPrev,sim.particles.q);
    const MAX=180,push=(arr,v)=>{arr.push(v);if(arr.length>MAX)arr.shift();};
    push(sim.elboH,sim.metrics.elbo);push(sim.klH,sim.metrics.kl);push(sim.hH,sim.metrics.H_q);
    push(sim.convH,sim.metrics.conv);push(sim.cmiH,sim.cmi.I_norm);
    push(sim.T_SZ_H,sim.tmat.T_SZ);push(sim.T_ZS_H,sim.tmat.T_ZS);push(sim.igH,sim.meanIG);
    sim.schemaStep=sim.reified?5:((it-1)%6);
    const L=sim.efeLog;
    if(it===1) L.push({iter:it,type:"attend",msg:"ATTEND [Eq.26]: q ← softmax[ℓℓ_k + λ·log p*(ω)]"});
    if(it===2) L.push({iter:it,type:"friston",msg:"EFE: G(π_k) = Σ_τ [Risk_τ + Ambiguity_τ − β·IG_τ]"});
    if(it===3) L.push({iter:it,type:"friston",msg:"Risk = KL[q(o_τ|π) ‖ P*(o)]  P*(o)=N(0,σ_pref)"});
    if(it===4) L.push({iter:it,type:"friston",msg:"IG = H[q(s|π)] − E[H[q(s|o,π)]]  (epistemic value)"});
    if(it===5) L.push({iter:it,type:"cmi",msg:"I(Z;S|B) = Σᵢ q_Z·q_S / N  →  watch fall to 0"});
    if(it===T_UPDATE) L.push({iter:it,type:"tmat",msg:`T matrix: T_SZ=${sim.tmat.T_SZ.toFixed(4)}  T_ZS=${sim.tmat.T_ZS.toFixed(4)}`});
    if(it===25) L.push({iter:it,type:"init",msg:"B_s = bright gold (sensory |φ|<π/2)  ·  B_a = dark gold (active |φ|≥π/2)"});
    if(it%50===0&&it>0) L.push({iter:it,type:"cmi",msg:`I=${sim.cmi.I_norm.toFixed(4)}  T_SZ=${sim.tmat.T_SZ.toFixed(4)}  T_ZS=${sim.tmat.T_ZS.toFixed(4)}  conv=${(sim.metrics.conv*100).toFixed(1)}%`});
    if(sim.reified&&it%60===0) L.push({iter:it,type:"emerge",msg:"✓  Toroidal Markov blanket reified — partition co-converged"});
    if(L.length>150)L.shift();
    setTick(t=>t+1);
  },[]);

  useEffect(()=>{
    let raf;const STEP_MS=90;// slightly slower due to Fristoniana EFE cost
    const loop=(ts)=>{raf=requestAnimationFrame(loop);if(runningRef.current&&ts-lastStep.current>=STEP_MS){lastStep.current=ts;step();}};
    raf=requestAnimationFrame(loop);return()=>cancelAnimationFrame(raf);
  },[]);

  const handleReset=()=>{setRunning(false);simRef.current=initSim(levers);setTick(t=>t+1);};
  const sim=simRef.current;
  const iter=sim?.iter??0,conv=sim?.metrics?.conv??0,reified=sim?.reified??false;
  const lambda=sim?.lambda??LAMBDA0,{kl=0,H_q=Math.log(3)}=sim?.metrics??{};
  const muB=sim?.macro?.muB??0,schema=sim?.schemaStep??0;
  const I_norm=sim?.cmi?.I_norm??1;
  const{TM=null,T_SZ=1,T_ZS=1}=sim?.tmat??{};
  const blanketOK=T_SZ<0.05&&T_ZS<0.05&&I_norm<0.15&&reified;

  const q=sim?.particles?.q,px_=sim?.particles?.px,py_=sim?.particles?.py,pz_=sim?.particles?.pz;
  let nBs=0,nBa=0,nZ=0,nS=0;
  if(q&&px_){for(let i=0;i<N;i++){nS+=q[i*3];nZ+=q[i*3+2];if(q[i*3+1]>q[i*3]&&q[i*3+1]>q[i*3+2]){if(isSensory(px_[i],py_[i],pz_[i]))nBs++;else nBa++;}}}
  nBs=Math.round(nBs);nBa=Math.round(nBa);nZ=Math.round(nZ);nS=Math.round(nS);

  const btnBase={padding:"8px 18px",fontFamily:MONO,fontSize:11,cursor:"pointer",borderRadius:5,border:"1px solid",transition:"all 0.15s"};

  return(
    <div style={{background:DC.bg,minHeight:"100vh",color:DC.text,fontFamily:SANS,fontSize:13,lineHeight:1.5}}>
      {/* HEADER */}
      <div style={{padding:"14px 20px 12px",borderBottom:`1px solid ${DC.border}`,display:"flex",alignItems:"center",gap:16,flexWrap:"wrap"}}>
        <div>
          <h1 style={{margin:0,fontFamily:SERIF,fontSize:20,fontWeight:"normal",color:DC.text,letterSpacing:0.3}}>Torus Emergence</h1>
          <p style={{margin:0,fontSize:10,color:DC.sub,fontFamily:MONO,letterSpacing:0.5}}>
            Inverted DMBD · Beck &amp; Ramstead (2025) · N={N} · EFE Fristoniana G(π)=Risk+Ambig−β·IG · k={K_ROLLOUT}
          </p>
        </div>
        <div style={{marginLeft:"auto",display:"flex",gap:8,alignItems:"center",flexWrap:"wrap"}}>
          <div style={{padding:"5px 12px",borderRadius:20,fontSize:10,fontFamily:MONO,fontWeight:"bold",letterSpacing:0.8,background:blanketOK?`${DC.green}15`:reified?`${DC.B}12`:DC.border,border:`1px solid ${blanketOK?DC.green:reified?DC.B:DC.border}`,color:blanketOK?DC.green:reified?DC.B:DC.muted}}>
            {blanketOK?"✓  BLANKET VERIFIED":reified?"◕  FORMING":"◌  INITIALIZING"}
          </div>
          {blanketOK&&<span style={{fontSize:10,color:DC.sub,fontFamily:MONO}}>I(Z;S|B) = {I_norm.toFixed(3)}</span>}
          <div style={{padding:"5px 12px",borderRadius:20,fontSize:10,fontFamily:MONO,background:DC.card,border:`1px solid ${DC.border}`,color:DC.sub}}>iter {iter}</div>
        </div>
      </div>

      {/* MAIN */}
      <div style={{display:"grid",gridTemplateColumns:"1fr 308px",gap:0,height:"calc(100vh - 64px)"}}>

        {/* LEFT */}
        <div style={{display:"flex",flexDirection:"column",borderRight:`1px solid ${DC.border}`,overflow:"auto"}}>
          {/* Canvas */}
          <div style={{position:"relative",flexShrink:0}}>
            <div style={{position:"absolute",top:10,left:12,zIndex:2,display:"flex",gap:10,flexWrap:"wrap"}}>
              {[[DC.Bs,"B_s  sensory",nBs],[DC.Ba,"B_a  active",nBa],[DC.Z,"Z  internal",nZ],[DC.S,"S  external",nS]].map(([col,lbl,n])=>(
                <div key={lbl} style={{display:"flex",alignItems:"center",gap:5,background:"rgba(5,5,8,0.78)",padding:"3px 8px",borderRadius:12,border:`1px solid ${col}33`}}>
                  <div style={{width:8,height:8,borderRadius:"50%",background:col,flexShrink:0}}/>
                  <span style={{fontSize:10,color:col,fontFamily:MONO}}>{lbl}</span>
                  <span style={{fontSize:9,color:DC.sub,fontFamily:MONO}}>({n})</span>
                </div>
              ))}
            </div>
            <canvas ref={canvasRef} width={700} height={420} style={{display:"block",width:"100%",background:DC.bg}}/>
            <div style={{position:"absolute",bottom:10,left:12,display:"flex",gap:8,alignItems:"center"}}>
              <button onClick={()=>setRunning(r=>!r)} style={{...btnBase,background:running?`${DC.S}18`:`${DC.green}18`,borderColor:running?DC.S:DC.green,color:running?DC.S:DC.green}}>{running?"■  PAUSE":"▶  RUN"}</button>
              <button onClick={step} disabled={running} style={{...btnBase,background:"transparent",borderColor:running?DC.dim:DC.borderBright,color:running?DC.dim:DC.sub}}>STEP</button>
              <button onClick={handleReset} style={{...btnBase,background:"transparent",borderColor:DC.border,color:DC.muted}}>RESET</button>
              <span style={{fontSize:9,color:DC.muted,fontFamily:MONO,marginLeft:4}}>cert=(max q−⅓)/(⅔) · B_s bright · B_a dark</span>
            </div>
          </div>

          {/* Stats */}
          <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:8,padding:"12px 14px",borderTop:`1px solid ${DC.border}`,flexShrink:0}}>
            <Stat label="Convergence" value={(conv*100).toFixed(1)} unit="%" color={DC.B} sub="B elements on torus"/>
            <Stat label="I(Z;S|B)" value={I_norm.toFixed(4)} color={DC.purple} sub="→ 0 = blanket valid"/>
            <Stat label="H[q] entropy" value={H_q.toFixed(4)} color={DC.Z} sub={`log3 = ${Math.log(3).toFixed(4)}`}/>
            <Stat label="Ḡ  mean EFE" value={(sim?.meanIG??0).toFixed(4)} color={DC.orange} sub="Risk+Ambig−β·IG"/>
          </div>

          {/* Charts */}
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:10,padding:"0 14px 14px",flexShrink:0}}>
            <div><p style={{margin:"0 0 4px",fontSize:10,color:DC.sub,fontFamily:MONO,letterSpacing:1,textTransform:"uppercase"}}>ELBO</p><MiniChart history={sim?.elboH??[]} color={DC.B} label="ELBO" tick={tick} description="Rising = structure crystallising"/></div>
            <div><p style={{margin:"0 0 4px",fontSize:10,color:DC.sub,fontFamily:MONO,letterSpacing:1,textTransform:"uppercase"}}>KL Risk → 0</p><MiniChart history={sim?.klH??[]} color={DC.orange} label="KL[q‖p*]" tick={tick} target={0} description="Aligned with designer prior"/></div>
            <div><p style={{margin:"0 0 4px",fontSize:10,color:DC.sub,fontFamily:MONO,letterSpacing:1,textTransform:"uppercase"}}>I(Z;S|B) → 0</p><MiniChart history={sim?.cmiH??[]} color={DC.purple} label="CMI" tick={tick} target={0} description="→ 0 = Markov blanket valid"/></div>
          </div>
        </div>

        {/* RIGHT */}
        <div style={{overflowY:"auto",display:"flex",flexDirection:"column"}}>
          {/* Schema */}
          <div style={{padding:"14px 14px 10px",borderBottom:`1px solid ${DC.border}`}}>
            <p style={{margin:"0 0 8px",fontSize:9,color:DC.sub,fontFamily:MONO,letterSpacing:2,textTransform:"uppercase"}}>Algorithm</p>
            <SchemaDiagram step={schema} lambda={lambda} kl={kl} Hq={H_q} muB={muB} levers={levers} I_norm={I_norm}/>
            <div style={{display:"flex",flexDirection:"column",gap:6,marginTop:10}}>
              <ProgressBar value={conv} color={DC.B} label="Radial convergence"/>
              <ProgressBar value={1-I_norm} color={DC.purple} label="Blanket separation  1 − I(Z;S|B)"/>
            </div>
          </div>

          {/* Levers */}
          <div style={{padding:"14px 14px 10px",borderBottom:`1px solid ${DC.border}`}}>
            <p style={{margin:"0 0 10px",fontSize:9,color:DC.sub,fontFamily:MONO,letterSpacing:2,textTransform:"uppercase"}}>Three Design Levers</p>
            <div style={{display:"flex",flexDirection:"column",gap:8}}>
              <Lever label="Precision Crafting" color={DC.lev1} equation="γ = γ₀ + 4·P,  σ_B tightens" tradeoff="convergence speed  ↔  local traps" value={levers.precision} onChange={v=>setLevers(l=>({...l,precision:v}))}/>
              <Lever label="Curiosity Sculpting" color={DC.lev2} equation="β = 0.18 + 0.6·C,  weights IG in G(π)" tradeoff="epistemic exploration  ↔  exploitation" value={levers.curiosity} onChange={v=>setLevers(l=>({...l,curiosity:v}))}/>
              <Lever label="Prediction Embedding" color={DC.lev3} equation="σ(t) = σ₀·exp(−t·E)  noise anneal" tradeoff="prediction precision  ↔  robustness" value={levers.embedding} onChange={v=>setLevers(l=>({...l,embedding:v}))}/>
            </div>
          </div>

          {/* T Matrix */}
          <div style={{padding:"14px 14px 10px",borderBottom:`1px solid ${DC.border}`}}>
            <p style={{margin:"0 0 10px",fontSize:9,color:DC.sub,fontFamily:MONO,letterSpacing:2,textTransform:"uppercase"}}>Transition Matrix T  <span style={{color:DC.dim,letterSpacing:0}}>— Eq. 21</span></p>
            <TMatrixDisplay TM={TM} T_SZ={T_SZ} T_ZS={T_ZS}/>
          </div>

          {/* Log */}
          <div style={{padding:"14px 14px",flex:1,minHeight:120}}>
            <p style={{margin:"0 0 8px",fontSize:9,color:DC.sub,fontFamily:MONO,letterSpacing:2,textTransform:"uppercase"}}>Event Log</p>
            <div style={{height:130}}><SimLog entries={sim?.efeLog??[]}/></div>
          </div>

          {/* Footer */}
          <div style={{padding:"8px 14px",borderTop:`1px solid ${DC.border}`,fontSize:9,color:DC.muted,fontFamily:MONO,lineHeight:1.8}}>
            <div>Beck &amp; Ramstead (2025) · Smith et al. (2020b) · G(π)=Risk+Ambig−β·IG</div>
            <div>IG=H[q(s|π)]−E[H[q(s|o,π)]] · I(Z;S|B)→0 · T_SZ=T[S→Z]→0 [Eq.21]</div>
          </div>
        </div>
      </div>
    </div>
  );
}
