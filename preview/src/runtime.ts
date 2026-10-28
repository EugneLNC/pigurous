import {validateProgram,type Program} from './choreography';
export const properties=['x','y','rotation','scale_x','scale_y','alpha','red','green','blue'] as const;
export type Property=typeof properties[number];
export type State=Record<Property,number>&{pulse?:number};
export type Entity={id:number;kind:string;parent_id:number|null;width:number;height:number;initial:Partial<State>;source?:Record<string,unknown>};
export type Event={beat:number;op:'create'|'destroy'|'transform'|'pulse';entity:number;duration:number;property:Property;start:number;end:number};
export type Score={version:string;coordinate_space:string;duration:number;tempo_map:{beat:number;time:number;bpm:number}[];entities:Entity[];events:Event[];segments:Record<string,unknown>[];metadata:Record<string,unknown>};
export const defaults:State={x:0,y:0,rotation:0,scale_x:1,scale_y:1,alpha:1,red:1,green:1,blue:1};
function last<T>(values:T[],number:number,key:(x:T)=>number){let l=0,r=values.length;while(l<r){const m=(l+r)>>>1;if(key(values[m])<=number)l=m+1;else r=m;}return l-1;}
export class Runtime{
  tracks=new Map<string,Event[]>();life=new Map<number,Event[]>();pulses=new Map<number,Event[]>();
  constructor(public score:Score){
    if(score.version!=='1.0'||score.coordinate_space!=='world_normalized_2d'||!(score.duration>0)||!Number.isFinite(score.duration))throw Error('不支持的 SceneScore 版本或时长');
    if(!score.tempo_map.length)throw Error('缺少 tempo_map');
    for(let i=0;i<score.tempo_map.length;i++){const p=score.tempo_map[i],prev=score.tempo_map[i-1];if(![p.beat,p.time,p.bpm].every(Number.isFinite)||p.bpm<=0||prev&&(p.beat<=prev.beat||Math.abs(p.time-prev.time-(p.beat-prev.beat)*60/prev.bpm)>1e-5))throw Error('无效时间映射');}
    const ids=new Set<number>();for(const e of score.entities){if(ids.has(e.id)||!['line','tap','hold','flick','drag','circle'].includes(e.kind)||e.width<=0||e.height<=0||![e.width,e.height,...Object.values(e.initial)].every(Number.isFinite))throw Error('无效对象');ids.add(e.id);}
    if(score.metadata.render_profile==='choreography-v3'){const p=validateProgram(score.metadata.presentation as Program);if(p.duration!==score.duration||JSON.stringify(p.tempo_map)!==JSON.stringify(score.tempo_map)||p.groups.length!==ids.size||p.groups.some(g=>!ids.has(g.id)))throw Error('动作程序与工程不匹配');}
    const alive=new Set<number>(),created=new Set<number>();let prev=-Infinity;
    for(const e of score.events){
      if(!ids.has(e.entity)||e.beat<prev||e.duration<0||![e.beat,e.duration,e.start,e.end].every(Number.isFinite)||!properties.includes(e.property)||this.time(e.beat)<-1e-5||this.time(e.beat+e.duration)>score.duration+1e-4)throw Error('无效事件');
      if(e.op==='create'){if(created.has(e.entity))throw Error('对象 ID 不可重复创建');alive.add(e.entity);created.add(e.entity);}else{if(!alive.has(e.entity))throw Error('对象尚未创建');if(e.op==='destroy')alive.delete(e.entity);}
      if(e.op==='transform'){const k=`${e.entity}:${e.property}`;if(!this.tracks.has(k))this.tracks.set(k,[]);this.tracks.get(k)!.push(e);}
      else if(e.op==='create'||e.op==='destroy'){if(!this.life.has(e.entity))this.life.set(e.entity,[]);this.life.get(e.entity)!.push(e);}
      else if(e.op==='pulse'){if(!this.pulses.has(e.entity))this.pulses.set(e.entity,[]);this.pulses.get(e.entity)!.push(e);}else throw Error('未知操作');prev=e.beat;
    }
  }
  beat(time:number){const ps=this.score.tempo_map,p=ps[Math.max(0,last(ps,time,x=>x.time))];return p.beat+(time-p.time)*p.bpm/60;}
  time(beat:number){const ps=this.score.tempo_map,p=ps[Math.max(0,last(ps,beat,x=>x.beat))];return p.time+(beat-p.beat)*60/p.bpm;}
  state(time:number){const b=this.beat(time),result=new Map<number,State>();for(const e of this.score.entities){const life=this.life.get(e.id)||[],i=last(life,b,x=>x.beat);if(i<0||life[i].op!=='create')continue;const s={...defaults,...e.initial};for(const p of properties){const es=this.tracks.get(`${e.id}:${p}`)||[],j=last(es,b,x=>x.beat);if(j>=0){const v=es[j],u=v.duration?Math.min(1,Math.max(0,(b-v.beat)/v.duration)):1;s[p]=v.start+(v.end-v.start)*u;}}s.pulse=Math.max(0,...(this.pulses.get(e.id)||[]).filter(v=>v.beat<=b&&b<v.beat+v.duration).map(v=>1-(b-v.beat)/Math.max(v.duration,1e-6)));result.set(e.id,s);}return result;}
}
