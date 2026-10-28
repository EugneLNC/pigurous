import * as THREE from 'three';
import type {Score,State} from './runtime';
import {mathematicalPrimitives} from './math';
import {choreographyPrimitives,type Program} from './choreography';

export type Presentation={profile:string;times:number[];controls:number[][];hits:number[][];duration:number;hue:number;seed:number;strict_gates?:boolean;poses?:number[][]};
export type Primitive={layer:number;points:number[][];rgb:number[];alpha:number;stroke:number;dot:boolean};
const TAU=Math.PI*2;
function hsv(h:number){h=((h%1)+1)%1;const k=h*6,i=Math.floor(k),f=k-i,p=.45,q=1-.55*f,t=.45+.55*f;return [[1,t,p],[q,1,p],[p,1,t],[p,q,1],[t,p,1],[1,p,q]][i%6];}
export function sample(p:Presentation,time:number){let l=0,r=p.times.length;while(l<r){const m=(l+r)>>>1;if(p.times[m]<=time)l=m+1;else r=m;}const i=Math.max(0,Math.min(p.times.length-2,l-1)),u=Math.max(0,Math.min(1,(time-p.times[i])/Math.max(1e-8,p.times[i+1]-p.times[i])));const value=p.controls[i].map((v,j)=>v*(1-u)+p.controls[i+1][j]*u);value[4]=p.hits.reduce((v,[at,s])=>time>=at&&time-at<.7?Math.max(v,s*Math.exp(-(time-at)/.13)):v,0);return value;}
export function primitives(p:Presentation,time:number):Primitive[]{
  if(p.profile==='choreography-v3')return choreographyPrimitives(p as unknown as Program,time);
  if(p.profile==='mathematical-v2')return mathematicalPrimitives(p,time);
  const [e,bass,mid,high,pulse,beat,orbit,ribbons,tunnel]=sample(p,time),fade=Math.min(1,Math.max(0,time/1.2),Math.max(0,(p.duration-time)/2)),gain=fade*Math.min(1,e*2.5),phase=beat*.09,hue=p.hue,out:Primitive[]=[];
  const strict=!!p.strict_gates;let pose:number[]|undefined;
  if(p.poses){let i=0;while(i<p.times.length-2&&p.times[i+1]<=time)i++;const u=Math.max(0,Math.min(1,(time-p.times[i])/Math.max(1e-8,p.times[i+1]-p.times[i])));pose=p.poses[i].map((v,j)=>v*(1-u)+p.poses![i+1][j]*u);}
  const add=(layer:number,points:number[][],color:number,alpha:number,stroke=1.5,dot=false)=>{if(strict&&layer===4)alpha*=high;if(pose){const [x,y,a,size]=pose,c=Math.cos(a),s=Math.sin(a);points=points.map(([px,py])=>[x+size*(px*c-py*s),y+size*(px*s+py*c)]);}if(alpha*gain>.002)out.push({layer,points,rgb:hsv(color),alpha:alpha*gain,stroke,dot});};
  const noise=(i:number)=>{const x=Math.sin(i*127.1+p.seed*31.7)*43758.5453;return x-Math.floor(x);};
  for(let i=0;i<84;i++){const x=(noise(i+1)*2-1)*1.7,y=(noise(i+201)*2-1)*.92+.025*Math.sin(phase+noise(i+401)*TAU);add(4,[[x,y]],hue+.08,.10+.28*high*noise(i+501),.8+noise(i+601)*1.4,true);}
  for(let i=0;i<10;i++){const z=((i/10+beat*.018)%1+1)%1,radius=.12+z*z*1.38,angle=phase*.28+z*.8,pts:number[][]=[];for(let j=0;j<97;j++){const theta=j/96*TAU,r=radius*(.85+.15*Math.abs(Math.sin(theta*2)))*(1+.035*mid*Math.sin(theta*6+phase));pts.push([r*Math.cos(theta+angle),r*.64*Math.sin(theta+angle)]);}add(2,pts,hue+.16*z,((strict?0:.08)+.66*tunnel)*(1-z)*(.45+.45*bass),1.1);}
  for(let k=0;k<5;k++){const pts:number[][]=[];for(let j=0;j<193;j++){const theta=j/192*TAU;let r=(.40+k*.038)*(1+.28*pulse);r+=(.018+.055*mid)*Math.sin(theta*(3+k%2)+phase*(k%2?1:-1)+k*.35);r+=.018*high*Math.sin(theta*15-phase*3);pts.push([r*Math.cos(theta+phase*.22),r*Math.sin(theta+phase*.22)]);}add(0,pts,hue+k*.027,((strict?0:.08)+orbit)*(1-k*.08),2+(k===0?1:0));}
  for(let k=0;k<6;k++){const pts:number[][]=[];for(let j=0;j<161;j++){const x=(j/160*2-1)*1.52,env=Math.max(0,1-(x/1.55)**2);let y=Math.sin(x*3.3+phase+k*.17)*(.12+.26*mid)*env;y+=Math.sin(x*8-phase*1.3+k*.25)*(.018+.055*high)*env;y+=(k-2.5)*.046;pts.push([x,y]);}add(1,pts,hue+.38+k*.012,(strict?0:.04)+.96*ribbons,2);}
  p.hits.forEach(([at,strength],index)=>{const age=time-at;if(age<0||age>.65)return;const radius=.36+.65*age,alpha=strength*(1-age/.65)**2,pts:number[][]=[];for(let j=0;j<97;j++)pts.push([radius*Math.cos(j/96*TAU),radius*Math.sin(j/96*TAU)]);add(3,pts,hue+.02,alpha*.8,2);for(let j=0;j<14;j++){const theta=j/14*TAU+index*.31,r=.4+age*(.65+noise(j+index)*.7);add(3,[[r*Math.cos(theta),r*Math.sin(theta)],[(r+.025+age*.04)*Math.cos(theta),(r+.025+age*.04)*Math.sin(theta)]],hue+.40+.18*noise(j+index+50),alpha,2);}});
  return out;
}

export class PerformanceSurface{
  canvas=document.createElement('canvas');ink=document.createElement('canvas');texture:THREE.CanvasTexture;mesh:THREE.Mesh;
  constructor(public score:Score){this.canvas.width=1280;this.canvas.height=720;const factor=score.metadata.render_profile==='mathematical-v2'?2:1;this.ink.width=1280*factor;this.ink.height=720*factor;this.texture=new THREE.CanvasTexture(this.canvas);this.texture.colorSpace=THREE.SRGBColorSpace;this.mesh=new THREE.Mesh(new THREE.PlaneGeometry(1920,1080),new THREE.MeshBasicMaterial({map:this.texture,depthTest:false}));this.mesh.renderOrder=-1;}
  draw(time:number,states:Map<number,State>,offsets:Map<number,number>){const w=this.ink.width,h=this.ink.height,ctx=this.ink.getContext('2d')!;ctx.clearRect(0,0,w,h);ctx.lineJoin='round';
    for(const v of primitives(this.score.metadata.presentation as Presentation,time)){const s=states.get(v.layer);if(!s)continue;const alpha=v.alpha*s.alpha,c=Math.cos(s.rotation*Math.PI/180),sn=Math.sin(s.rotation*Math.PI/180);const pts=v.points.map(([x,y])=>{x*=s.scale_x;y*=s.scale_y;return [w/2+(s.x+(offsets.get(v.layer)||0))*w/2+(x*c-y*sn)*h/2,h/2-s.y*h/2-(x*sn+y*c)*h/2];});const rgb=v.rgb.map((value,i)=>Math.max(0,Math.min(255,Math.round(value*alpha*[s.red,s.green,s.blue][i]*255))));ctx.strokeStyle=ctx.fillStyle=`rgb(${rgb.join(',')})`;ctx.lineWidth=Math.max(1,v.stroke*h/540);ctx.beginPath();if(v.dot){ctx.arc(pts[0][0],pts[0][1],Math.max(.7,v.stroke*h/1080),0,TAU);ctx.fill();}else{pts.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();}}
    const out=this.canvas.getContext('2d')!,cw=this.canvas.width,ch=this.canvas.height,math=this.score.metadata.render_profile==='mathematical-v2';out.globalCompositeOperation='source-over';out.filter='none';out.fillStyle='#03060f';out.fillRect(0,0,cw,ch);out.globalCompositeOperation='lighter';out.drawImage(this.ink,0,0,cw,ch);out.filter=`blur(${ch*(math?.004:.008)}px) brightness(${math?210:350}%)`;out.drawImage(this.ink,0,0,cw,ch);out.filter=`blur(20px) brightness(${math?65:300}%)`;out.drawImage(this.ink,0,0,cw,ch);out.filter='none';this.texture.needsUpdate=true;
  }
  dispose(){this.mesh.geometry.dispose();(this.mesh.material as THREE.Material).dispose();this.texture.dispose();}
}
