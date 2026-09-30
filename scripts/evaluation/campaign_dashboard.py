"""Standalone interactive SVG dashboard: eval seeds and training replicas stay distinct."""
import json
from pathlib import Path

def render(data, output):
    payload={'runs':{k:{a:v for a,v in r.items() if a in ('name','method','train_seed','group','primary','training_status','last_step')} for k,r in data['runs'].items()},'points':data['points']}
    for name,r in payload['runs'].items():
        r['expected_steps']=([40,80,120,156,200,240,280,312] if r['primary'] and r['method']!='sft'
                             else sorted({p['step'] for p in data['points'] if p['run']==name}))
    template=Path(__file__).with_name('campaign_dashboard.html').read_text()
    output.write_text(template.replace('__DATA__',json.dumps(payload).replace('</','<\\/')))

if __name__=='__main__':
    import sys
    render(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
