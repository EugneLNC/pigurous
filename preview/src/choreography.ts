import type {Primitive} from './performance';
export const materials=['stroke','orbit','ribbon','tunnel','spark','stars','rose','lissajous','hypotrochoid'];
export const verbs=['enter','exit','hold','translate','rotate','scale','draw','morph','split','merge','gather','scatter'];
export type ActionCall={object:number;material:string;verb:string;beat:number;duration:number;from:number[];to:number[];curve?:string;blend_material?:boolean;motif_id?:string;role?:string};
export type Program={profile:string;version:string;duration:number;tempo_map:{beat:number;time:number;bpm:number}[];groups:{id:number;name?:string}[];calls:ActionCall[];hits?:number[][];seed?:number;accent_object?:number};
const TAU=Math.PI*2,defaults=[.5,.5,.35,.5,0,.52,.4,.5,0,0,1,.3];
const channels:Record<string,number[]>={enter:[0,1,2,3,4,5,6,7,8,9,10,11],exit:[4],hold:[],translate:[0,1,7],rotate:[3],scale:[2],draw:[10],morph:[6],split:[8,9],merge:[8,9],gather:[0,1,8],scatter:[0,1,8]};
function hsv(h:number){h=((h%1)+1)%1;const k=h*6,i=Math.floor(k),f=k-i,p=.48,q=1-.52*f,t=.48+.52*f;return [[1,t,p],[q,1,p],[p,1,t],[p,q,1],[t,p,1],[1,p,q]][i%6].map(x=>x*.98);}
function beatAt(p:Program,time:number){let i=0;while(i<p.tempo_map.length-1&&p.tempo_map[i+1].time<=time)i++;const point=p.tempo_map[i];return point.beat+(time-point.time)*point.bpm/60;}
function timeAt(p:Program,beat:number){let i=0;while(i<p.tempo_map.length-1&&p.tempo_map[i+1].beat<=beat)i++;const point=p.tempo_map[i];return point.time+(beat-point.beat)*60/point.bpm;}
export function validateProgram(p:Program){
 if(p.profile!=='choreography-v3'||p.version!=='choreography-actions-v2'||!(p.duration>0)||!p.tempo_map.length)throw Error('不支持的动作程序');
 const ids=new Set(p.groups.map(g=>g.id));if(ids.size!==p.groups.length||p.groups.some(g=>!Number.isInteger(g.id)||g.id<0))throw Error('无效动作对象');
 let previous=-Infinity;for(const c of p.calls){if(!ids.has(c.object)||!materials.includes(c.material)||!verbs.includes(c.verb)||!Number.isFinite(c.beat+c.duration)||c.beat<previous-1e-7||c.duration<0||timeAt(p,c.beat)<-1e-4||timeAt(p,c.beat+c.duration)>p.duration+1e-4)throw Error('无效动作引用或时间');for(const v of [c.from,c.to])if(v.length!==12||v.some(x=>!Number.isFinite(x)||x<0||x>1))throw Error('无效动作参数');if(!['smooth','linear','step'].includes(c.curve||'smooth'))throw Error('无效动作曲线');if(c.blend_material!==undefined&&typeof c.blend_material!=='boolean')throw Error('无效素材过渡');previous=c.beat;}
 return p;
}
export function actionStates(p:Program,time:number){
 const beat=beatAt(p,time),result=new Map<number,{values:number[];material:string;motif_id:string;role:string;morph_from?:string;morph_progress?:number;morph_shape?:number}>(p.groups.map(g=>[g.id,{values:[...defaults],material:'stroke',motif_id:'',role:'foreground'}]));
 for(const call of p.calls){if(call.beat>beat)break;const u=call.duration===0?1:Math.min(1,Math.max(0,(beat-call.beat)/call.duration)),curve=call.curve||'smooth',q=curve==='smooth'?u*u*(3-2*u):curve==='step'?Number(u>=1):u,values=call.from.map((a,i)=>a+(call.to[i]-a)*q);
  if(call.verb==='translate'&&u<1){const dx=call.to[0]-call.from[0],dy=call.to[1]-call.from[1],bend=(call.to[7]-.5)*.8*Math.sin(Math.PI*u);values[0]=Math.max(0,Math.min(1,values[0]-dy*bend));values[1]=Math.max(0,Math.min(1,values[1]+dx*bend));}
  const state=result.get(call.object)!;for(const channel of channels[call.verb])state.values[channel]=values[channel];if((call.verb==='morph'||(call.verb==='enter'&&call.blend_material===true))&&call.material!==state.material){state.morph_from=state.material;state.morph_progress=q;state.morph_shape=call.from[6];}else if(['enter','morph'].includes(call.verb)){delete state.morph_from;delete state.morph_progress;}if(['enter','morph'].includes(call.verb))state.material=call.material;state.motif_id=call.motif_id||'';state.role=call.role||'foreground';
 }return result;
}
function contour(material:string,shape:number,phase:number){const points:number[][]=[];for(let j=0;j<129;j++){const u=j/128;let t=u*TAU,x:number,y:number;if(material==='stroke'){x=u*2-1;y=.12*shape*Math.sin(t+phase);}else if(material==='ribbon'){x=u*2-1;y=(.15+.35*shape)*Math.sin(x*(2+4*shape)+phase)*Math.max(0,1-x*x);}else if(material==='rose'){const r=.25+.65*Math.cos(5*t);x=r*Math.cos(t)+.10*shape*Math.cos(2*t+phase);y=r*Math.sin(t)+.10*shape*Math.sin(3*t-phase);}else if(material==='lissajous'){x=.85*Math.sin(3*t+.8*shape);y=.65*Math.sin(2*t);}else if(material==='hypotrochoid'){t*=2;const d=1.6+2*shape;x=(3*Math.cos(t)+d*Math.cos(1.5*t))/(3+d);y=(3*Math.sin(t)-d*Math.sin(1.5*t))/(3+d);}else{const r=.68+.10*shape*Math.sin(5*t+phase)+.04*Math.sin(9*t-phase);x=r*Math.cos(t);y=r*Math.sin(t);}points.push([x,y]);}return points;}
export function choreographyPrimitives(p:Program,time:number):Primitive[]{const out:Primitive[]=[],seed=p.seed??42;
 for(const [ident,state] of drawableStates(p,time)){let [x,y,size,angle,alpha,hue,shape,,spread,copies,reveal,depth]=state.values;if(alpha<.003||reveal<.001)continue;size=.12+size*.95;angle=(angle-.5)*TAU*2;x=(x-.5)*2.7;y=(y-.5)*1.6;const material=state.material,phase=time*(.20+shape*.8)+ident*.73,rgb=hsv(hue),count=1+copies*9;
  for(let k=0;k<Math.ceil(count);k++){const visibility=Math.min(1,Math.max(0,count-k)),theta=k*TAU/Math.max(1,count)+ident*.17,ox=k?Math.cos(theta)*spread*.75:0,oy=k?Math.sin(theta)*spread*.45:0,c=Math.cos(angle),sn=Math.sin(angle);
   const add=(pts:number[][],a=1,stroke=1.7,dot=false)=>out.push({layer:ident,points:pts.map(([px,py])=>[x+ox+size*(px*c-py*sn),y+oy+size*(px*sn+py*c)]),rgb,alpha:alpha*visibility*a,stroke,dot});
   if(material==='stars'){for(let j=0;j<36;j++){const n=j+seed*3+ident*137,rawA=Math.sin(n*127.1)*43758.5453,rawB=Math.sin((n+41)*127.1)*43758.5453,a=rawA-Math.floor(rawA),b=rawB-Math.floor(rawB);add([[(a*2-1)*1.7,b*2-1]],.35+.25*Math.sin(phase+j)**2,1.7,true);}}
   else if(material==='spark'){for(let j=0;j<12;j++){const a=j*TAU/12,r=.15+.65*shape;add([[r*Math.cos(a),r*Math.sin(a)],[(r+.12)*Math.cos(a),(r+.12)*Math.sin(a)]],.8,2);}}
   else{const rings=['orbit','tunnel','ribbon'].includes(material)?4:1;for(let ring=0;ring<rings;ring++){const factor=material==='tunnel'?.65+ring*.20+depth*.2:1+ring*.045,pts=contour(material,shape,phase+ring*.25).map(([px,py])=>[px*factor,material==='ribbon'?py*factor+(ring-1.5)*.055:py*factor]);add(pts.slice(0,Math.max(2,Math.ceil(pts.length*reveal))),1-ring*.14,1.5+(ring===0?.8:0));}}
  }
 }
 for(const [at,strength] of p.hits||[]){const age=time-at;if(age<0||age>=.55)continue;const radius=.16+age*1.8;out.push({layer:p.accent_object??p.groups[0]?.id??0,points:Array.from({length:65},(_,j)=>[radius*Math.cos(j*TAU/64),radius*Math.sin(j*TAU/64)]),rgb:[.9,.8,1],alpha:strength*(1-age/.55)**2,stroke:1.5,dot:false});}
 return out;
}

function drawableStates(p:Program,time:number){const out:Array<[number,{values:number[];material:string}]>=[];for(const [id,state] of actionStates(p,time)){if(state.morph_from&&(state.morph_progress??1)<1){const q=state.morph_progress!,old={...state,values:[...state.values]};old.material=state.morph_from;old.values[4]*=1-q;old.values[6]=state.morph_shape??old.values[6];out.push([id,old]);const current={...state,values:[...state.values]};current.values[4]*=q;out.push([id,current]);}else out.push([id,state]);}return out;}
