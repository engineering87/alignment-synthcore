from pathlib import Path
import hashlib,json,subprocess,sys
R=Path(__file__).resolve().parents[1]
def test_weight_archives_reassemble_exactly():
 for p in (R/'results').glob('weight-reading-*-v1/weights-manifest.json'):
  m=json.loads(p.read_text());parts=[]
  for e in m['parts']:
   b=(p.parent/e['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==e['sha256'];parts.append(b)
  assert hashlib.sha256(b''.join(parts)).hexdigest()==m['sha256']
def test_saved_weight_analysis_reproduces_results():
 subprocess.run([sys.executable,str(R/'experiments/weight_reading/integration_audit.py')],check=True,capture_output=True)
 x=json.loads((R/'results/weight-reading-integration-audit.json').read_text());assert x['blind_models_verified']==96 and x['passed']
