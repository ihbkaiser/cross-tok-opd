#!/usr/bin/env python3
"""External eval telemetry: no queue/source mutation, secrets or answer text."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


class Counter:
    def __init__(self): self.files={}
    def count(self,path):
        stat=path.stat();identity=(stat.st_dev,stat.st_ino)
        old=self.files.get(str(path),(identity,0,0,0))
        _,offset,count,mtime=old
        if old[0]!=identity or stat.st_size<offset or (stat.st_size==offset and stat.st_mtime_ns!=mtime): offset=count=0
        with path.open('rb') as f:
            f.seek(offset)
            while chunk:=f.read(1024*1024): count+=chunk.count(b'\n')
            offset=f.tell()
        self.files[str(path)]=(identity,offset,count,stat.st_mtime_ns)
        return count


def read(path):
    try: return Path(path).read_text().strip()
    except OSError: return None


def workers(plan):
    found={}
    table={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            table[int(p.name)]={'ppid':int(fields[1]),'ticks':int(fields[11])+int(fields[12]),'start':fields[19],
                                'rss_bytes':int(fields[21])*os.sysconf('SC_PAGE_SIZE')}
            argv=(p/'cmdline').read_bytes().split(b'\0')
            if b'worker' in argv and b'--plan' in argv and argv[argv.index(b'--plan')+1].decode()==str(plan): found[int(p.name)]={}
        except (OSError,ValueError,IndexError,UnicodeError): continue
    for pid in found:
        family={pid}
        while True:
            larger=family|{k for k,v in table.items() if v['ppid'] in family}
            if larger==family: break
            family=larger
        found[pid]={'processes':{str(k):table[k] for k in family if k in table},'descendants':len(family)-1}
    return found


def gpu():
    try:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5)
        keys=('gpu','util_pct','memory_util_pct','memory_used_mib','memory_total_mib','power_w')
        return [dict(zip(keys,[v.strip() for v in row])) for row in csv.reader(raw.splitlines())]
    except (OSError,subprocess.SubprocessError) as e: return {'error':type(e).__name__}


def sample(root,plan,counter):
    cells=[]
    for cell in sorted((root/'cells').glob('*/*/*')):
        if not cell.is_dir(): continue
        responses=counter.count(cell/'responses.jsonl') if (cell/'responses.jsonl').exists() else 0
        scores=counter.count(cell/'scores.jsonl') if (cell/'scores.jsonl').exists() else 0
        cells.append({'cell':str(cell.relative_to(root/'cells')),'responses':responses,'scores':scores,
                      'waiting_for_score':responses-scores,'complete':(cell/'metrics.json').exists()})
    server=[]
    for log in root.glob('*-server.log'):
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-32768));lines=f.read().decode(errors='replace').splitlines()
        for line in reversed(lines):
            match=re.search(r'Decode batch, #running-req: (\d+).*gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)',line)
            if match:
                server.append({'log':log.name,'log_mtime':log.stat().st_mtime,'last_decode_time':line.split(']')[0].lstrip('['),
                               'running_requests':int(match[1]),'tokens_per_second':float(match[2]),'queued_requests':int(match[3])});break
    mem={}
    for line in (read('/proc/meminfo') or '').splitlines():
        k,v=line.split(':',1)
        if k in ('MemTotal','MemAvailable','MemFree','Cached','SwapFree'): mem[k]=v.strip()
    limits={name:read('/sys/fs/cgroup/'+name) for name in ('memory.current','memory.max','cpu.max','memory/memory.usage_in_bytes','memory/memory.limit_in_bytes','cpu/cpu.cfs_quota_us','cpu/cpu.cfs_period_us')}
    return {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'epoch':time.time(),'gpu':gpu(),
            'cells':cells,'server_last_decode':server,'workers':workers(plan),'meminfo':mem,'cgroup':limits,
            'loadavg':read('/proc/loadavg'),'pressure':{k:read('/proc/pressure/'+k) for k in ('cpu','memory','io')},
            'cpu_stat':(read('/proc/stat') or '').splitlines()[0]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=int,default=600);p.add_argument('--interval',type=float,default=5)
    a=p.parse_args()
    if not 5<=a.seconds<=3600 or not 2<=a.interval<=60: p.error('seconds 5..3600; interval 2..60')
    plan=a.plan.resolve(strict=True);counter=Counter();last=None
    with a.out.open('x') as output:
        output.write(json.dumps({'type':'metadata','plan':str(plan),'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),
          'cpu_affinity':len(os.sched_getaffinity(0)),'clock_ticks':os.sysconf('SC_CLK_TCK'),'scope':'External samples; row counts can race live writes, RSS double-counts shared pages; no answer text captured'})+'\n');output.flush()
        end=time.monotonic()+a.seconds
        while time.monotonic()<end:
            try:
                row=sample(plan.parent,plan,counter)
                totals={k:sum(c[k] for c in row['cells']) for k in ('responses','scores','waiting_for_score')}
                row['totals']=totals
                if last:
                    dt=row['epoch']-last['epoch'];row['rates_per_second']={k:(totals[k]-last['totals'][k])/dt for k in ('responses','scores')}
                output.write(json.dumps(row)+'\n');output.flush();last=row
            except (OSError,ValueError) as e:
                output.write(json.dumps({'type':'sample_error','error':type(e).__name__,'epoch':time.time()})+'\n');output.flush()
            time.sleep(min(a.interval,max(0,end-time.monotonic())))
    print('TELEMETRY_DONE='+str(a.out))

if __name__=='__main__': main()
