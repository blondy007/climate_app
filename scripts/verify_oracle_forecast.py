"""Read-only app checks or an isolated, additive backup/restore exercise over SSH.

No credentials or application data values are printed. Administration uses the
existing Tailscale route; the public application does not require it.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess

PREFIX = "xk53thwe7wzp3wah1qxvttvg"


def remote(command):
    key = os.environ.get("ORACLE_VPS_SSH_KEY")
    if not key:
        raise SystemExit("Falta la configuración privada de SSH")
    result = subprocess.run(["ssh", "-i", key, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                             "ubuntu@100.120.251.18", command], capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise SystemExit(f"La comprobación remota falló ({result.returncode}); revisar conectividad o estado de la app")
    return result.stdout.strip()


STATUS = """
import json,sqlite3
from climate_app.forecast.store import Store
from climate_app.health import main as health
from climate_app.auth import owner_config, HTTPS_MARKER
health()
s=Store()
with s.connect() as c:
    counts={t:c.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in ['locations','runs','points','observations']}
    counts['integrity']=c.execute('PRAGMA integrity_check').fetchone()[0]
print(json.dumps({'composite_health':'ok','counts':counts,
 'owner_login_configured':bool(owner_config()),'https_login_verified':HTTPS_MARKER.is_file(),
 'collector':s.state('collector'),
 'providers':{l.name:{p:s.state('task:'+l.id+':'+p,{}).get('status') for p in ['ecmwf','icon','gfs','metno','weathernext3','observations','climate']} for l in s.locations()}}))
"""

RESTORE = """
import json,pathlib,sqlite3,tarfile,tempfile,shutil
from climate_app.forecast.store import Store
s=Store()
snapshot=s.backup()
root=pathlib.Path(tempfile.mkdtemp(prefix='climateapp-recovery-',dir='/tmp')).resolve()
assert root.parent == pathlib.Path('/tmp') and root.name.startswith('climateapp-recovery-')
archive=root/'climate-app-data.tar.gz'
files=[snapshot,s.path.parent/'meteostat_master.csv',s.path.parent/'wind_thresholds.json',s.path.parent/'auth-owner.json']
with tarfile.open(archive,'w:gz') as t:
    for path in files:
        if path.is_file(): t.add(path,arcname=path.name,recursive=False)
restored=root/'restored'
restored.mkdir()
with tarfile.open(archive,'r:gz') as t:
    for item in t.getmembers():
        assert item.isfile() and pathlib.PurePosixPath(item.name).name == item.name
        with t.extractfile(item) as source, (restored/item.name).open('wb') as dest:
            shutil.copyfileobj(source,dest)
c=sqlite3.connect(restored/'forecasts-backup.sqlite3')
assert c.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
counts={table:c.execute('SELECT count(*) FROM '+table).fetchone()[0] for table in ['locations','runs','points','observations']}
assert counts['runs']>0 and counts['points']>0 and counts['observations']>0
c.close()
historical=restored/'meteostat_master.csv'
assert historical.exists() and historical.stat().st_size>100
owner=s.path.parent/'auth-owner.json'
owner_restored=owner.is_file() and (restored/owner.name).read_bytes()==owner.read_bytes()
if owner.is_file(): assert owner_restored
print(json.dumps({'restore':'ok','isolated_directory':str(root),'archive_bytes':archive.stat().st_size,
 'counts':counts,'historical_csv_bytes':historical.stat().st_size,'owner_account_restored':owner_restored}))
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "restore-test", "verify-https"])
    args = parser.parse_args()
    names = remote("sudo docker ps --filter " + shlex.quote("name=" + PREFIX) + " --format '{{.Names}}'").splitlines()
    if len(names) != 1 or not names[0].startswith(PREFIX) or not names[0].replace("-", "").isalnum():
        raise SystemExit("No hay un único contenedor activo inequívoco de ClimateApp")
    container = shlex.quote(names[0])
    if args.action == "verify-https":
        print(remote(f"sudo docker exec {container} python -m climate_app.auth verify-https"))
    else:
        script = STATUS if args.action == "status" else RESTORE
        output = json.loads(remote(f"sudo docker exec {container} python -c {shlex.quote(script)}"))
        if args.action == "status":
            template = '{"image":{{json .Config.Image}},"state":{{json .State.Status}},"health":{{json .State.Health.Status}}}'
            output["container"] = json.loads(remote(f"sudo docker inspect --format {shlex.quote(template)} {container}"))
        print(json.dumps(output))


if __name__ == "__main__":
    main()
