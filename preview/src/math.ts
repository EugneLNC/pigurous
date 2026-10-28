import type {Presentation,Primitive} from './performance';
const TAU=Math.PI*2;
function hsv(h:number){h=((h%1)+1)%1;const k=h*6,i=Math.floor(k),f=k-i,p=.58,q=1-.42*f,t=.58+.42*f;return [[1,t,p],[q,1,p],[p,1,t],[p,q,1],[t,p,1],[1,p,q]][i%6];}

export function mathematicalCurve(family:number,travel:number,expansion:number,spin:number,phase:number,pulse=0,points=721){
  const out:number[][]=[];
  for(let j=0;j<points;j++){let t=j/(points-1)*TAU,x:number,y:number;
    if(family===0){const morph=.72+.20*travel,radius=.78*((1-morph)+morph*Math.cos(5*t))*(1+.025*pulse);x=radius*Math.cos(t);y=radius*Math.sin(t);}
    else if(family===1){const delta=.08+.14*Math.sin(phase*.24)+.12*travel;x=.70*Math.sin(3*t+delta);y=(.45+.025*spin)*Math.sin(2*t);}
    else if(family===2){t*=2;const d=2.35+.55*expansion+.18*Math.sin(phase*.19),norm=.78/(3+d);x=norm*(3*Math.cos(t)+d*Math.cos(1.5*t));y=norm*(3*Math.sin(t)-d*Math.sin(1.5*t));}
    else throw Error('Unknown mathematical action');
    out.push([x,y]);
  }out[out.length-1]=out[0].slice();return out;
}

export function mathematicalSample(p:Presentation,time:number){
  let l=0,r=p.times.length;while(l<r){const m=(l+r)>>>1;if(p.times[m]<=time)l=m+1;else r=m;}
  const i=Math.max(0,Math.min(p.times.length-2,l-1)),u=Math.max(0,Math.min(1,(time-p.times[i])/Math.max(1e-8,p.times[i+1]-p.times[i])));
  const controls=p.controls[i].map((x,j)=>x*(1-u)+p.controls[i+1][j]*u);
  const pose=(p.poses?.[i]||[0,0,0,1]).map((x,j)=>x*(1-u)+(p.poses?.[i+1]||[0,0,0,1])[j]*u);
  controls[4]=p.hits.reduce((v,[at,s])=>time>=at&&time-at<.7?Math.max(v,s*Math.exp(-(time-at)/.13)):v,0);
  return {controls,pose};
}

export function mathematicalPrimitives(p:Presentation,time:number):Primitive[]{
  const {controls,pose}=mathematicalSample(p,time),[visibility,expansion,travel,spin,pulse,clock,...weights]=controls;
  const fade=Math.min(1,Math.max(0,time/1.8),Math.max(0,(p.duration-time)/2)),gain=fade*Math.min(1,visibility*2),phase=clock*.11;
  const emphasized=weights.map(w=>Math.max(0,w)),denom=Math.max(1e-9,emphasized.reduce((a,b)=>a+b,0)),mixes=emphasized.map(v=>v/denom);
  const [x,y,angle,size]=pose,c=Math.cos(angle),s=Math.sin(angle),hue=p.hue,out:Primitive[]=[];
  const add=(layer:number,pts:number[][],color:number,alpha:number,stroke=1.4,dot=false,posed=true)=>{
    if(alpha*gain<.004)return;if(posed)pts=pts.map(([px,py])=>[x+size*(px*c-py*s),y+size*(px*s+py*c)]);
    out.push({layer,points:pts,rgb:hsv(color),alpha:alpha*gain,stroke,dot});
  };
  for(let j=0;j<72;j++){const theta=j*2.399963229728653+phase*.025,radius=.86+.12*Math.sqrt((j+.5)/72);
    add(4,[[radius*Math.cos(theta),radius*Math.sin(theta)]],hue+.04,(.025+.04*spin)*weights.reduce((a,b)=>a+b,0),.8,true,false);}
  if(emphasized.reduce((a,b)=>a+b,0)>1e-8){
    const bases=[0,1,2].map(f=>mathematicalCurve(f,travel,expansion,spin,phase,pulse));
    let pts=Array.from({length:721},(_,j)=>[mixes.reduce((v,w,f)=>v+w*bases[f][j][0],0),mixes.reduce((v,w,f)=>v+w*bases[f][j][1],0)]);
    const norm=.76*(1+.025*pulse)/Math.max(.1,...pts.map(([px,py])=>Math.hypot(px,py)));pts=pts.map(([px,py])=>[px*norm,py*norm]);
    for(const echo of [1,0]){const scaled=pts.map(([px,py])=>[px*(1+echo*.045),py*(1+echo*.045)]);
      for(let segment=0;segment<24;segment++){const start=segment*30,color=hue+.045*Math.sin(segment/24*TAU+phase*.2);
        add(echo?1:0,scaled.slice(start,start+31),color,echo ? .10 : .92,echo?1:1.7);}}
    const index=Math.floor((((phase*.10)%1+1)%1)*720);add(0,[pts[index]],hue+.03,.85,2.2,true);
    const guide=Array.from({length:161},(_,j)=>[.79*Math.cos(j/160*TAU),.79*Math.sin(j/160*TAU)]);add(2,guide,hue+.06,.055*mixes[2],.9);
  }
  const active=p.hits.filter(([at])=>time>=at&&time-at<.36).slice(-2);
  for(const [at,strength] of active){const age=time-at,alpha=strength*(1-age/.36)**2,radius=.80+.11*age/.36;
    for(let j=0;j<12;j++){const theta=j/12*TAU+phase*.03;add(3,[[radius*Math.cos(theta),radius*Math.sin(theta)]],hue+.50,alpha*.6,1.4,true);}}
  return out;
}
