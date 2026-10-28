"""Conservative additional corpus and pre-generation frozen song-family holdout."""
import argparse,json,hashlib,unicodedata
from pathlib import Path
from choreographer.importer import NativeChart
from choreographer.audio import sha256
from choreographer.action_data import prepare

p=argparse.ArgumentParser();p.add_argument('--select-only',action='store_true');p.add_argument('--device',default='cuda');a=p.parse_args()
root=Path('data/corpus-v3');root.mkdir(parents=True,exist_ok=True)
old=json.loads(Path('data/actions-v1/manifest.json').read_text(encoding='utf-8'))
new=json.loads(Path('data/phira-v3/sources.json').read_text(encoding='utf-8'))
def family(r):
 s=r['song_id'].split('|')[0]
 if 'phira' not in r.get('sources',{}) and s.endswith('.0'):s=s.rsplit('.',2)[0]
 return ''.join(c for c in unicodedata.normalize('NFKC',s).casefold() if c.isalnum())
old_hash={r['audio_hash'] for r in old};old_names={family(r) for r in old}
prior_errors_path=Path('data/actions-v3/quarantine.json')
prior_failed={r['song_id'] for r in json.loads(prior_errors_path.read_text(encoding='utf-8'))} if prior_errors_path.exists() else set()
accepted=[];errors=[];seen=set()
for r in new:
 try:
  native=NativeChart(json.loads(Path(r['chart']).read_text(encoding='utf-8-sig')),lines_only=True)
  digest=sha256(r['audio']);name=family(r)
  if r['song_id'] in prior_failed:raise ValueError('Failed full native trajectory sampling in corpus preflight')
  if digest in old_hash or name in old_names:raise ValueError('Existing song family or identical audio; not a fresh holdout')
  if digest in seen:raise ValueError('Duplicate audio in new collection')
  seen.add(digest);accepted.append(dict(r,audio_hash=digest,group_key=name,line_count=len(native.lines)))
 except Exception as e:errors.append(dict(song_id=r['song_id'],reason=str(e)))
accepted.sort(key=lambda r:hashlib.sha256(r['group_key'].encode()).hexdigest())
# Whole song families, not charts/encodings, are frozen before model/LLM experimentation.
frozen_path=root/'fresh-holdout.json'
if frozen_path.exists():
 original=json.loads(frozen_path.read_text(encoding='utf-8'))
 available={r['group_key'] for r in accepted}
 frozen={family(r) for r in original}&available
 if not (root/'holdout-preflight-history.json').exists():
  (root/'holdout-preflight-history.json').write_text(json.dumps(dict(original=original,reason='canonical title deduplication and full native sampling preflight; before fresh model/LLM inference'),ensure_ascii=False,indent=2),encoding='utf-8')
 for r in accepted:
  if len(frozen)>=20:break
  frozen.add(r['group_key'])
else:
 frozen=set()
 for r in accepted:
  frozen.add(r['group_key'])
  if len(frozen)>=20:break
for r in accepted:
 h=int(hashlib.sha256(r['group_key'].encode()).hexdigest()[:8],16)
 r['split']='test' if r['group_key'] in frozen else ('val' if h%10==0 else 'train')
 r['fresh_holdout']=r['group_key'] in frozen
fresh=[r for r in accepted if r['fresh_holdout']]
frozen_path.write_text(json.dumps(fresh,ensure_ascii=False,indent=2),encoding='utf-8')
(root/'new-sources.json').write_text(json.dumps(accepted,ensure_ascii=False,indent=2),encoding='utf-8')
(root/'quarantine.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(new_supported=len(accepted),fresh_song_groups=len(frozen),excluded=len(errors))),flush=True)
if not a.select_only:
 result=prepare(['data/actions-v1/manifest.json',str(root/'new-sources.json')],'data/actions-v3',device=a.device)
 print(json.dumps(result),flush=True)
