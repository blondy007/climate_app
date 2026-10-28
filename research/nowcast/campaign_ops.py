"""Bounded orchestration for the separately authorized 30 October radar trial.

Runs on the Oracle host. The scientific implementation is mounted unchanged.
No application database, credentials, new public port or paid service is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

UTC = timezone.utc
START = datetime(2026, 10, 30, 11, tzinfo=UTC)
END = datetime(2026, 11, 2, 11, tzinfo=UTC)
ROOT = Path('/var/lib/docker/volumes/xk53thwe7wzp3wah1qxvttvg-climate-app-data/_data/radar-lab-20261030')
CODE = Path('/home/ubuntu/climate-radar-lab/code-b107f760a60b/research')
IMAGE = 'sha256:80329987f703a94fd21fea5fe3816d4d4dd969e477d71045d42fa549114cd387'
MODEL = '5e8e4205924afd7c9514b9de5d851cd63ed53e90ec87b9046867ac68eacd952e'
NAME = 'climate-radar-capture-20261030'
# Includes two bounded preparation attempts and three start probes, each <=150MB.
# These reserves are conservative upper bounds, not measured network traffic.
PROBE_RESERVE = 750_000_000

PROBE = r'''
import hashlib, io, json
import numpy as np
from research.nowcast.data import list_day, utcnow, parse_stamp, request_bytes, decode_frame, HOST
from research.nowcast.model import forecast
now=utcnow()
items=list_day(now.date(),'openradar-24h')
selected={p:sorted([x for x in items if x['product']==p and parse_stamp(x['stamp'])<=now],key=lambda x:x['stamp']) for p in ('DBZH','RATE')}
assert len(selected['DBZH'])>=3 and selected['RATE'], 'missing_recent_files'
chosen=selected['DBZH'][-3:]+selected['RATE'][-1:]
reserved=sum(x['bytes']*3 for x in chosen)
assert reserved<=150000000,'probe_download_cap'
frames={}; summary=[]
for item in chosen:
    body=request_bytes(HOST+'/'+item['bucket']+'/'+item['key'])
    arrays,meta=decode_frame(body,item['product'],parse_stamp(item['stamp']))
    meta['retrieved_at']=utcnow().isoformat()
    crop=io.BytesIO(); np.savez_compressed(crop,**arrays)
    meta['sha256_crop']=hashlib.sha256(crop.getvalue()).hexdigest()
    frames[(item['product'],item['stamp'])]=(arrays,meta)
    summary.append({'product':item['product'],'stamp':item['stamp'],'catalog_last_modified':item['last_modified'],'age_minutes':round((utcnow()-parse_stamp(item['stamp'])).total_seconds()/60,2),'sites':meta['sites']})
issue=utcnow(); results={}
for site in ('los-alcazares','torrejon'):
    db=[frames[('DBZH',x['stamp'])] for x in chosen[:3]]
    rr=frames[('RATE',chosen[-1]['stamp'])]
    try:
        record=forecast([x[0][site] for x in db],[x[1] for x in db],rr[0][site],rr[1],site,issue,replay=False)
        results[site]={'can_issue':True,'points':record['points']}
    except (ValueError,RuntimeError) as exc:
        results[site]={'can_issue':False,'reason':str(exc)}
elapsed=max((now-now.replace(hour=0,minute=0,second=0,microsecond=0)).total_seconds(),1)
print(json.dumps({'at':issue.isoformat(),'reserved_download_bytes':reserved,'catalog_files':len(items),'estimated_72h_reserved_bytes':int(sum(x['bytes']*3 for x in items if x['product'] in ('DBZH','RATE'))/elapsed*72*3600),'frames':summary,'sites':results,'approved_for_public_use':False}))
'''


def now():
    return datetime.now(UTC)


def exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as out:
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)


def run(args, timeout=45):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=True)


def constraints(name, network_none=False):
    args = ['docker', 'run', '--rm', '--name', name, '--cpus', '0.5', '--memory', '768m',
            '--pids-limit', '128', '--read-only', '--cap-drop', 'ALL', '--security-opt',
            'no-new-privileges', '--tmpfs', '/tmp:rw,noexec,nosuid,size=64m',
            '--mount', f'type=bind,src={CODE},dst=/work/research,readonly']
    if network_none:
        args += ['--network', 'none']
    return args


def probe(label):
    assert label in ('preparation', 'start-1', 'start-2', 'start-3')
    assert hashlib.sha256((CODE/'nowcast/model.py').read_bytes()).hexdigest() == MODEL
    result = run(constraints('climate-radar-probe-20261030-'+label)+[IMAGE,'python','-c',PROBE], 150)
    value = json.loads(result.stdout.strip().splitlines()[-1])
    exclusive(ROOT/(label+'-probe.json'), value)
    return value


def start():
    # A missed timer cannot extend the agreed window or use an exhausted session.
    if not START <= now() < START.replace(minute=35):
        raise RuntimeError('Outside the authorized capture window')
    if (ROOT/'started.json').exists():
        print(json.dumps({'status':'already_started','budget_reset':False}))
        return
    assert shutil.disk_usage('/').free > 10*1024**3, 'Insufficient host disk margin'
    if (ROOT/'live-config.json').exists() or (ROOT/'live-budget.json').exists():
        raise RuntimeError('Existing campaign must never be recreated or reset')
    attempts=[ROOT/f'start-attempt-{n}.json' for n in (1,2,3)]
    number=next((n for n,p in enumerate(attempts,1) if not p.exists()),None)
    if number is None:
        raise RuntimeError('All three bounded source checks were already used')
    exclusive(attempts[number-1],{'at':now().isoformat(),'attempt':number})
    value = probe(f'start-{number}')
    # Coverage and freshness are mandatory. Dry/unstable optical flow is a
    # scientific availability outcome; no thresholds are relaxed to make it pass.
    ages = {p:max(x['age_minutes'] for x in value['frames'] if x['product']==p)
            for p in ('DBZH','RATE')}
    latest = value['frames'][2]
    coverage = latest['sites']['los-alcazares']['nearby_coverage']
    reason = value['sites']['los-alcazares'].get('reason')
    if ages['DBZH'] > 30 or ages['RATE'] > 45 or coverage < .95 or reason in (
            'Insufficient radar coverage','No coverage at research location',
            'Stale radar observations','Reflectivity scan window too old',
            'Missing or irregular radar sequence'):
        exclusive(ROOT/f'start-blocked-{number}.json',{'at':now().isoformat(),'reason':'source_gate_failed',
                  'probe':value,'approved_for_public_use':False})
        raise RuntimeError('Radar source gate failed; campaign not launched')
    exclusive(ROOT/'live-budget.json',{'reserved_download_bytes':PROBE_RESERVE})
    # Only the new research subdirectory is exposed to the collector.
    args = ['docker','create','--name',NAME,'--cpus','0.5','--memory','768m',
            '--pids-limit','128','--read-only','--cap-drop','ALL','--security-opt',
            'no-new-privileges','--tmpfs','/tmp:rw,noexec,nosuid,size=64m',
            '--user','1000:1000','--restart','on-failure:5','--log-opt','max-size=5m',
            '--log-opt','max-file=2','--mount',f'type=bind,src={CODE},dst=/work/research,readonly',
            '--mount',f'type=bind,src={ROOT},dst=/data',IMAGE,
            'python','-m','research.nowcast.live','--root','/data','--expires-at',END.isoformat()]
    container_id = run(args).stdout.strip()
    run(['docker','start',NAME])
    exclusive(ROOT/'started.json',{'at':now().isoformat(),'container':NAME,'container_id':container_id,
              'expires_at':END.isoformat(),'approved_for_public_use':False})


def finish():
    if now() < END:
        raise RuntimeError('Never stop the collector before the authorized deadline')
    result=subprocess.run(['docker','inspect','--format','{{.State.Running}}',NAME],
                          capture_output=True,text=True,timeout=15)
    stopped=False
    if result.returncode == 0 and result.stdout.strip() == 'true':
        run(['docker','stop','--time','20',NAME],45)
        stopped=True
    path=ROOT/'deadline-stop.json'
    if not path.exists():
        exclusive(path,{'checked_at':now().isoformat(),'expires_at':END.isoformat(),
                        'container':NAME,'stop_was_needed':stopped,'approved_for_public_use':False})


def status():
    value = {'checked_at':now().isoformat(),'root':str(ROOT),'approved_for_public_use':False}
    for name in ('deployment','preparation-probe','start-1-probe','start-2-probe','start-3-probe',
                 'started','start-blocked-1','start-blocked-2','start-blocked-3','deadline-stop',
                 'live-status','live-config','live-budget','prospective-report'):
        path = ROOT/(name+'.json')
        if path.is_file():
            value[name] = json.loads(path.read_text())
    result = subprocess.run(['docker','inspect','--format',
                '{{json .State}}',NAME],capture_output=True,text=True,timeout=15)
    if result.returncode == 0:
        state=json.loads(result.stdout)
        value['container']={k:state.get(k) for k in ('Status','Running','ExitCode','OOMKilled','FinishedAt','StartedAt')}
        value['container']['health']=state.get('Health',{}).get('Status')
    value['storage_bytes']=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    print(json.dumps(value))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action', choices=('probe','start','finish','status'))
    args=parser.parse_args()
    if args.action == 'probe':
        print(json.dumps(probe('preparation')))
    elif args.action == 'start':
        start()
    elif args.action == 'finish':
        finish()
    else:
        status()
