from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[2]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def read(p):return json.loads(Path(p).read_text())
def seed(study,phase,i,part=''):return int.from_bytes(hashlib.sha256(f'{study}/{phase}/{i}/{part}'.encode()).digest()[:4],'big')
def ci(values,alpha=.05):
 x=np.array(values,float);assert len(x)>1;m=float(x.mean());h=float(t.ppf(1-alpha/2,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)));return {'n':len(x),'mean':m,'interval':[m-h,m+h],'alpha':alpha,'values':x.tolist()}
def signature(study):
 paths=[]
 for d in ['experiments/regret_budget','experiments/oversight_coverage','experiments/approval_factorial','experiments/vase_blind_spot','testbeds/ued_gridworld','testbeds/interactive_gridworld','harness']:
  paths += list((ROOT/d).rglob('*.py'))
 driver={'online-start-replay-v1':'online_replay.py','changing-mask-factorial-v1':'changing_mask.py','rr-coverage-v1':'rr_coverage.py','fresh-adaptive-monitor-v1':'fresh_monitor.py'}[study]
 paths += [ROOT/'ideas/accel-levels/method.py',ROOT/'experiments/four_followups/common.py',ROOT/'experiments/four_followups'/driver]
 return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
def freeze(out,protocol):
 assert not (out/'freeze.json').exists();dump(out/'protocol.json',protocol);dump(out/'freeze.json',{'created_utc':now(),'protocol_sha256':sha(out/'protocol.json'),'source_hashes':signature(out.name),'confirmation_accessed':False})
def verify(out,registered=False):
 f=read(out/'freeze.json');assert sha(out/'protocol.json')==f['protocol_sha256'];assert signature(out.name)==f['source_hashes'],'Source changed after freeze'
 if registered:
  r=read(out/'registration-receipt.json');assert r['protocol_sha256']==f['protocol_sha256'] and r['revision']>=1
 return f
