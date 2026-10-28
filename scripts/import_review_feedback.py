import argparse,json,shutil
from pathlib import Path
from choreographer.reviews import import_reviews

p=argparse.ArgumentParser()
p.add_argument('file');p.add_argument('--manifest',default='preview/public/reviews/manifest.json')
p.add_argument('--plans',default='data/choreography-v3');p.add_argument('--output',default='data/choreography-v3/user-reviews.json')
a=p.parse_args();records=import_reviews(a.file,a.manifest,a.plans,a.output)
copy=Path(a.output).with_name(Path(a.output).stem+'-original.json')
shutil.copyfile(a.file,copy)
print(json.dumps(dict(records=len(records),selected=sum(r['preferred_candidate'] is not None for r in records),
    rejected=sum(r['preferred_candidate'] is None for r in records),output=a.output),ensure_ascii=False))
