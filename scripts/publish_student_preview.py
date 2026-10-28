"""Publish only validated same-song pairs and their matching local media."""
import argparse,json,hashlib,shutil
from pathlib import Path
from choreographer.score import SceneScore

p=argparse.ArgumentParser();p.add_argument('manifest');a=p.parse_args()
rows=json.loads(Path(a.manifest).read_text(encoding='utf-8'));web=Path('preview/public/comparison');web.mkdir(parents=True,exist_ok=True);items=[]
for r in rows:
    ident=r['chart_sha256'][:16];audio=Path(r['audio']);digest=hashlib.sha256(audio.read_bytes()).hexdigest()
    media=web/(ident+audio.suffix);shutil.copyfile(audio,media)
    paths={};checkpoints=[]
    for variant in ('model','llm'):
        original=Path(r['variants'][variant]['score']);score=SceneScore.load(original)
        if score.metadata['audio']['audio_sha256']!=digest:raise ValueError('Audio/pair hash mismatch')
        checkpoints.append(score.metadata['checkpoint_sha256'])
        score.metadata['preview_audio_url']='/comparison/'+media.name
        path=web/original.name;score.save(path);paths[variant]='/comparison/'+path.name
        video=original.with_suffix('.mp4')
        if video.exists():shutil.copyfile(video,web/video.name);paths[variant+'_video']='/comparison/'+video.name
    if checkpoints[0]!=checkpoints[1]:raise ValueError('A/B checkpoints differ')
    items.append(dict(song_id=r['song_id'],audio='/comparison/'+media.name,**paths))
(web/'manifest.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
index=next(i for i in items if 'dcef4cf813abf326' in i['model'])
for variant,name in [('model','choreography.json'),('llm','choreography-llm.json')]:
    shutil.copyfile(Path('preview/public')/index[variant].lstrip('/'),Path('preview/public')/name)
print('Published',len(items),'same-weight pairs, default NWAD and matching audio')
