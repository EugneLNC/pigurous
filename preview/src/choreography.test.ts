import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';
import {actionStates,choreographyPrimitives,validateProgram,type Program} from './choreography';
const f=JSON.parse(fs.readFileSync('../tests/fixtures/choreography_parity.json','utf8'));
test('independent object action program agrees with Python and survives shuffled seeks',()=>{
 const p=validateProgram(f.program as Program);
 for(const sample of [...f.samples].reverse()){
  const result=choreographyPrimitives(p,sample.time);
  assert.equal(result.length,sample.primitives.length);
  for(let i=0;i<result.length;i++){const a=result[i],b=sample.primitives[i];assert.equal(a.layer,b[0]);assert.equal(a.dot,b[5]);assert.ok(Math.abs(a.alpha-b[3])<1e-7);for(let j=0;j<a.points.length;j++)for(let k=0;k<2;k++)assert.ok(Math.abs(a.points[j][k]-b[1][j][k])<1e-7);for(let k=0;k<3;k++)assert.ok(Math.abs(a.rgb[k]-b[2][k])<1e-7);}
  for(const [id,s] of actionStates(p,sample.time))s.values.forEach((x,i)=>assert.ok(Math.abs(x-sample.states[String(id)].values[i])<1e-7));
 }
});
test('invalid group references are rejected',()=>{const p=structuredClone(f.program);p.calls[0].object=100;assert.throws(()=>validateProgram(p));});
test('zero duration action applies at its exact start',()=>{const p=structuredClone(f.program);p.calls[0].duration=0;assert.equal(actionStates(p,0).get(0)!.values[4],p.calls[0].to[4]);});
