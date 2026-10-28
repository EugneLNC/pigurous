import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {Runtime} from './runtime';
import {primitives,sample} from './performance';
for(const file of ['presentation_parity.json','action_parity.json','math_parity.json'])test(`${file}: controls and geometry agree with Python at arbitrary seek times`,()=>{
  const fixture=JSON.parse(fs.readFileSync('../tests/fixtures/'+file,'utf8'));
  function near(a:number,b:number){assert.ok(Math.abs(a-b)<1e-6,`${a} != ${b}`);}
  for(const f of fixture.frames){sample(fixture.presentation,f.time).forEach((v,i)=>near(v,f.sample[i]));const got=primitives(fixture.presentation,f.time);assert.equal(got.length,f.primitives.length);got.forEach((p,i)=>{const ref=f.primitives[i];assert.equal(p.layer,ref.layer);assert.equal(p.points.length,ref.count);assert.equal(p.dot,ref.dot);near(p.alpha,ref.alpha);near(p.stroke,ref.stroke);p.rgb.forEach((v,j)=>near(v,ref.rgb[j]));[p.points[0],p.points[Math.floor(p.points.length/2)],p.points[p.points.length-1]].forEach((xy,j)=>xy.forEach((v,k)=>near(v,ref.points[j][k])));});}
});
test('Python/TypeScript renderer fixture agrees at shuffled seek times',()=>{
  const fixture=JSON.parse(fs.readFileSync('../tests/fixtures/runtime_parity.json','utf8'));
  const rt=new Runtime(fixture.score);
  for(const frame of fixture.frames){const actual=rt.state(frame.time);for(const [id,s] of Object.entries(frame.states)){const state=actual.get(Number(id));assert.ok(state);for(const [p,v] of Object.entries(s as Record<string,number>)){assert.ok(Math.abs((state as any)[p]-v)<1e-7,`${id}.${p}`);}}assert.equal(actual.size,Object.keys(frame.states).length);}
});
