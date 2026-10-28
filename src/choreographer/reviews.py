"""Bind human preferences to the exact candidates and intervals they watched."""
import hashlib
import json
from pathlib import Path


def import_reviews(source, manifest, plans, output):
    raw = Path(source).read_bytes()
    judgments = json.loads(raw.decode('utf-8-sig'))
    items = json.loads(Path(manifest).read_text(encoding='utf-8'))['items']
    by_chart = {i['chart_sha256']: i for i in items}
    if not isinstance(judgments, list) or not judgments:
        raise ValueError('Expected a nonempty list of actual human preferences')
    normalized = []
    seen = set()
    for j in judgments:
        chart = j['chart_sha256']; item = by_chart.get(chart)
        if item is None or chart in seen:
            raise ValueError('Unknown or duplicate reviewed chart')
        seen.add(chart)
        if j.get('source') != 'explicit_user_review':
            raise ValueError('Preference must come from an explicit human review')
        if any(abs(float(j[k])-float(item[k])) > 1e-6 for k in ('start_seconds','end_seconds')):
            raise ValueError('Watched interval differs from the displayed candidate')
        candidates = [c['candidate'] for c in item['candidates']]
        choice = j['preferred_candidate']
        if choice is not None and (type(choice) is not int or choice not in candidates):
            raise ValueError('Selected candidate was not displayed')
        hashes = {}
        for c in item['candidates']:
            path = Path(plans)/chart[:16]/('plan-'+str(c['candidate'])+'.json')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != c['plan_sha256']:
                raise ValueError('Reviewed candidate plan has changed')
            hashes[str(c['candidate'])] = digest
        normalized.append(dict(j, reviewed_candidates=candidates, candidate_plan_hashes=hashes,
                               feedback_file_sha256=hashlib.sha256(raw).hexdigest()))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(normalized,ensure_ascii=False,indent=2),encoding='utf-8')
    return normalized
