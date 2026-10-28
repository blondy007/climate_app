"""Read-only audit of immutable prospective records; no radar downloads."""

from collections import Counter
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
import hashlib, json, math
import numpy as np

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--root',required=True)
parser.add_argument('--model',default=str(Path(__file__).with_name('model.py')))
args=parser.parse_args()
root=Path(args.root)
report = json.loads((root / 'prospective-report.json').read_text())
cutoff = datetime.fromisoformat(report['generated_at'])
config = json.loads((root / 'live-config.json').read_text())
deployment = json.loads((root / 'deployment.json').read_text())
assert not deployment['public_forecast_enabled'] and not report['approved_for_public_use']
model_hash = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
assert model_hash == config['model_sha256']

def stamp(when):
    return when.strftime('%Y%m%dT%H%M')

@lru_cache(maxsize=48)
def metadata(product, key):
    return json.loads((root / 'frames' / product / (key + '.json')).read_text())

@lru_cache(maxsize=24)
def field(key):
    with np.load(root / 'frames' / 'RATE' / (key + '.npz'), allow_pickle=False) as data:
        return {site: data[site] for site in ('los-alcazares', 'torrejon')}

def finite(value):
    return value is not None and math.isfinite(value)

def categorical(rows, key):
    counts = dict(hits=0, misses=0, false_alarms=0, correct_dry=0)
    for row in rows:
        assert finite(row[key]) and finite(row['observed_rate'])
        wet, observed = row[key] >= .1, row['observed_rate'] >= .1
        counts['hits' if wet and observed else 'misses' if observed else 'false_alarms' if wet else 'correct_dry'] += 1
    h, m, f = (counts[k] for k in ('hits', 'misses', 'false_alarms'))
    return {**counts, 'csi': h / (h + m + f) if h + m + f else None,
            'probability_of_detection': h / (h + m) if h + m else None,
            'false_alarm_ratio': f / (h + f) if h + f else None}

counts, emitted, rejected = Counter(), Counter(), Counter()
rows = []
unique_rows = set()
maximum_record_seconds = 0
first_issue = last_issue = None
for path in sorted((root / 'prospective').glob('*.json')):
    batch = json.loads(path.read_text())
    issue = datetime.fromisoformat(batch['issued_at'])
    recorded = datetime.fromisoformat(batch['recorded_at'])
    if recorded > cutoff:
        continue
    assert issue <= recorded and not batch['approved_for_public_use']
    maximum_record_seconds = max(maximum_record_seconds, (recorded - issue).total_seconds())
    first_issue = first_issue or batch['issued_at']
    last_issue = batch['issued_at']
    counts['issue_batches'] += 1
    rejected.update(batch['unavailable'])
    seen_sites = set()
    for record in batch['forecasts']:
        site = record['site']
        assert site not in seen_sites
        seen_sites.add(site)
        assert record['issued_at'] == batch['issued_at']
        assert record['algorithm'] == config['algorithm'] and record['mode'] == 'prospective'
        assert not record['approved_for_public_use']
        source_times = record['source_dbzh'] + [record['source_rate']]
        assert len(source_times) == len(record['source_crop_hashes']) == 4
        for n, source_time in enumerate(source_times):
            meta = metadata('DBZH' if n < 3 else 'RATE', stamp(datetime.fromisoformat(source_time)))
            assert datetime.fromisoformat(meta['nominal_end']) <= issue
            assert datetime.fromisoformat(meta['retrieved_at']) <= issue
            assert meta['sha256_crop'] == record['source_crop_hashes'][n]
        emitted[site] += 1
        for point in record['points']:
            target = datetime.fromisoformat(point['valid_time'])
            assert recorded < target
            identity = (site, batch['issued_at'], point['valid_time'])
            assert identity not in unique_rows
            unique_rows.add(identity)
            counts['wet_prediction_points'] += finite(point['rain_rate']) and point['rain_rate'] >= .1
            if target.minute % 15 or target.second or target.microsecond:
                counts['unsupported_non_15_minute_targets'] += 1
                continue
            key = stamp(target)
            if target > cutoff or not (root / 'frames' / 'RATE' / (key + '.json')).exists():
                counts['pending_points'] += 1
                continue
            meta = metadata('RATE', key)
            if datetime.fromisoformat(meta['retrieved_at']) > cutoff:
                counts['pending_points'] += 1
                continue
            cy, cx = meta['sites'][site]['centre']
            observed = float(field(key)[site][cy, cx])
            rows.append({'site': site, 'issued_at': batch['issued_at'], **point,
                         'observed_rate': observed if math.isfinite(observed) else None})

assert len(rows) == report['verified_points']
assert sum(emitted.values()) == report['forecast_site_records']
assert counts['pending_points'] == report['pending_points']
assert counts['unsupported_non_15_minute_targets'] == report['unsupported_non_15_minute_targets']
assert sum(emitted.values()) + sum(rejected.values()) == counts['issue_batches'] * 2
results = []
for site in ('los-alcazares', 'torrejon'):
    for lead in (15, 30, 60):
        selected = [row for row in rows if row['site'] == site and row['lead_minutes'] == lead]
        paired = [row for row in selected if all(finite(row[k]) for k in ('rain_rate', 'persistence_rate', 'observed_rate'))]
        live = next(r for r in report['results'] if r['site'] == site and r['lead_minutes'] == lead)
        assert live['cases'] == len(selected)
        results.append({'site': site, 'lead_minutes': lead, 'available_reference_targets': len(selected),
                        'paired_cases': len(paired), 'excluded_from_comparison': len(selected) - len(paired),
                        'missing_model': sum(not finite(row['rain_rate']) for row in selected),
                        'missing_persistence': sum(not finite(row['persistence_rate']) for row in selected),
                        'missing_reference': sum(not finite(row['observed_rate']) for row in selected),
                        'nowcast': categorical(paired, 'rain_rate'),
                        'persistence': categorical(paired, 'persistence_rate')})
value = {'checked_at': datetime.now(timezone.utc).isoformat(), 'as_of': cutoff.isoformat(),
         'first_issue': first_issue, 'last_issue': last_issue,
         'model_sha256': model_hash, 'chronology_and_source_availability': 'passed',
         'maximum_record_seconds': maximum_record_seconds, 'counts': dict(counts),
         'availability': {site: {'attempts': counts['issue_batches'], 'emitted': emitted[site],
                                'rejected': counts['issue_batches'] - emitted[site]}
                          for site in ('los-alcazares', 'torrejon')},
         'rejected_issue_site_reasons': dict(rejected), 'verified_points': len(rows),
         'unique_observation_targets': len({(row['site'], row['valid_time']) for row in rows}),
         'wet_reference_points': sum(finite(row['observed_rate']) and row['observed_rate'] >= .1 for row in rows),
         'results': results, 'approved_for_public_use': False,
         'limitations': ['Overlapping targets are not independent episodes',
                        'Reference is dependent OPERA RATE, not a rain gauge',
                        'Only exact 15-minute reference timestamps; no interpolation',
                        'Rejected origins are retained as unavailable, never counted as dry']}

# Availability includes missing scheduled origins, not just surviving batches.
start=datetime.fromisoformat(deployment.get('starts_at',config['started_at']))
end=datetime.fromisoformat(deployment['expires_at'])
as_of=min(datetime.now(timezone.utc),end)
slots=max(0,math.floor((as_of-start).total_seconds()/300))
emission_slots={site:set() for site in ('los-alcazares','torrejon')}
observed_slots=set()
opportunities=Counter()
for path in sorted((root/'prospective').glob('*.json')):
    batch=json.loads(path.read_text())
    issue=datetime.fromisoformat(batch['issued_at'])
    if datetime.fromisoformat(batch['recorded_at'])>cutoff:
        continue
    slot=math.floor((issue-start).total_seconds()/300)
    records={r['site']:r for r in batch['forecasts']}
    if 0<=slot<slots:
        observed_slots.add(slot)
        for site in records:
            emission_slots[site].add(slot)
    aligned=datetime.fromtimestamp(math.ceil(issue.timestamp()/300)*300,timezone.utc)
    for site in ('los-alcazares','torrejon'):
        for lead in (15,30,60):
            target=aligned+timedelta(minutes=lead)
            if target.minute%15 or target>cutoff:
                continue
            key=stamp(target)
            if not (root/'frames'/'RATE'/(key+'.json')).is_file():
                opportunities[site+':missing_reference']+=1
                continue
            meta=metadata('RATE',key)
            if datetime.fromisoformat(meta['retrieved_at'])>cutoff:
                opportunities[site+':missing_reference']+=1
                continue
            cy,cx=meta['sites'][site]['centre']
            observed=float(field(key)[site][cy,cx])
            if not finite(observed):
                opportunities[site+':missing_reference']+=1
                continue
            points={p['lead_minutes']:p for p in records.get(site,{}).get('points',[])}
            available=lead in points and finite(points[lead]['rain_rate'])
            kind='wet' if observed>=.1 else 'dry'
            opportunities[site+':'+kind+'_reference']+=1
            opportunities[site+':'+kind+('_with_model' if available else '_without_model')]+=1
value['calendar_availability']={
    'starts_at':start.isoformat(),'as_of':as_of.isoformat(),
    'scheduled_completed_5minute_slots':slots,'slots_with_recorded_batch':len(observed_slots),
    'slots_without_batch':slots-len(observed_slots),
    'sites':{site:{'slots_with_prediction':len(items),
                   'availability':len(items)/slots if slots else None}
             for site,items in emission_slots.items()}}
value['all_recorded_origin_reference_opportunities']=dict(opportunities)
value['limitations'] += [
    'Calendar availability includes downtime and an early budget stop; emission-only availability is conditional',
    'Wet opportunities include rejected origins when exact RATE truth exists; overlapping targets are not independent episodes',
    'When no batch or no reference exists, rainfall at that origin remains unknown']

print(json.dumps(value, ensure_ascii=True))
