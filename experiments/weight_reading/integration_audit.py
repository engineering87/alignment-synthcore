from pathlib import Path
import json,hashlib,zipfile,io,importlib.util
import numpy as np
from scipy.stats import kurtosis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
R=Path(__file__).resolve().parents[2];E=R/'experiments/weight_reading';checks=[]
for level in ['l3','l4']:
 p=R/f'results/weight-reading-{level}-v1'
 for line in (p/'freeze_sha256.txt').read_text().splitlines():
  h,name=line.split();f=E/('PREREGISTRATION_L3_FROZEN.md' if level=='l3' and name=='PREREGISTRATION.md' else name);assert hashlib.sha256(f.read_bytes()).hexdigest()==h;(checks.append(f'{level}:{name}'))
spec=importlib.util.spec_from_file_location('frozen_reader',E/'read_l4.py');reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
levels=[]
for tag,result,pred,expected in [('l1l2','results.json','predictions.json','15a24fc58498f7750150300ffc9c92473e4afda04329dd85092d9dcb119e2a0b'),('l3','results_L3.json','predictions_L3.json','587f0f1f026c70cdc60816e1fa742fc9d75499e7afcf732159a8eed879ce43c9'),('l4','results_L4.json','predictions_L4.json','64771a05fa16607f88ee500eb960fa05b8c57242b332805f531ed5dc82d4320a')]:
 p=R/f'results/weight-reading-{tag}-v1';assert hashlib.sha256((p/pred).read_bytes()).hexdigest()==expected
 manifest=json.loads((p/'weights-manifest.json').read_text());raw=b''.join((p/v['path']).read_bytes() for v in manifest['parts']);assert hashlib.sha256(raw).hexdigest()==manifest['sha256'];archive=zipfile.ZipFile(io.BytesIO(raw));assert archive.testzip() is None;key=json.loads((p/'key.json').read_text());preds=json.loads((p/pred).read_text());results=json.loads((p/result).read_text())
 def weights(path):return np.load(io.BytesIO(archive.read(path)),allow_pickle=False)
 def features(z,channel=False):
  f=[]
  for k in ['W1','b1','W2','b2','W3','b3']:
   x=z[k].ravel();f.extend([x.mean(),x.std(),abs(x).mean(),kurtosis(x),np.linalg.norm(x)])
  if channel:
   w=z['W1'];a=np.linalg.norm(w[:,:81]);g=np.linalg.norm(w[:,81:162])
   if tag=='l4':
    l=np.linalg.norm(w[:,162:]);f.extend([a,g,l,g/a,l/a,g/l])
   else:f.extend([a,g,g/a])
  return f
 base=[r for r in key if r['pool']=='base'];y=[int(r['ood']['p_goal']>.5) for r in base]
 fitted=[make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(np.array([features(weights(r['path']),c) for r in base]),y) for c in [False,True]]
 for level,x in results.items():
  rows=x['per_model'];assert len(rows)==24
  for r in rows:
   assert r['truth']==int(r['ood_p_goal']>.5)
   assert r['reader']==int(preds[level][r['id']]['label']=='goal')
   z=weights(next(k['path'] for k in key if k['pool']=='blind' and k['level']==level and Path(k['path']).stem==r['id']))
   for c,model in enumerate(fitted):assert int(model.predict(np.array([features(z,bool(c))]))[0])==r[['B_flat','B_chan'][c]]
   if tag=='l4':
    sg,sl,S=reader.score(z);assert int(S>.5)==r['reader'];assert round(float(np.clip(.05+.85*(S-.2)/.6,.02,.95)),3)==r['reader_p_goal']
  accuracy={name:sum(r[name]==r['truth'] for r in rows) for name in ['reader','B_flat','B_chan']}
  for n,v in accuracy.items():assert abs(v/24-x[{'reader':'reader_acc','B_flat':'B_flat_acc','B_chan':'B_chan_acc'}[n]])<1e-12
  levels.append(dict(level=level,n=24,correct=accuracy))
report=dict(passed=True,levels=levels,freeze_checks=checks,prediction_hashes_verified=3,blind_models_verified=96,L4_frozen_reader_recomputed=24,baseline_predictions_recomputed=192,scope='Integrity and saved-weight/statistical-score audit. No retraining or behavior-rollout rerun; blind sequencing is source-reported. L1/L2 freeze was not committed before training. L3 historical preregistration snapshot retained separately.')
(R/'results/weight-reading-integration-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
