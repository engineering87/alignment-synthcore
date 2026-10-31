import unittest
import numpy as np
from experiments.four_followups import online_replay as O,rr_coverage as R,changing_mask as M,fresh_monitor as F
class Engineering(unittest.TestCase):
 def test_observed_start_no_oracle(self):self.assertTrue(O.engineering()['passed'])
 def test_missing_return_edge_and_restore(self):
  T=np.array([[1,0,0,0,0],[1,0,1,1,1]]);terminal=np.zeros(2,bool);partial=T.copy();partial[1,1]=-1
  self.assertGreater(R.rr_penalty(partial,terminal)[0,0],.4);self.assertLess(R.rr_penalty(T,terminal)[0,0],.03)
 def test_coverage_mask_keeps_noop_and_state_identity(self):
  T=np.array([[1,0,0,0,0],[1,0,1,1,1]]);term=np.zeros(2,bool);P=R.mask_graph(T,term,.5,np.random.default_rng(7));self.assertTrue(np.array_equal(P[:,4],T[:,4]));self.assertEqual(P.shape,T.shape);self.assertEqual((P<0).sum(),4)
 def test_query_labels_assigned_to_query_actions(self):
  query=np.roll(np.arange(5),1);trace={'obs':np.zeros(5,int),'action':np.arange(5),'next_obs':np.zeros(5,int),'reward':np.zeros(5),'query_action':query,'terminated':np.zeros(5,bool)};result=M.fit(trace,(query==3).astype(float),0);self.assertEqual(result['policy'][0],3)
 def test_scaling_uses_observed_means(self):
  trace={'obs':np.repeat([10,20],5),'query_action':np.tile(np.arange(5),2),'original':np.tile(-np.arange(5)*.2,2)};self.assertTrue(np.allclose(M.span_scale(trace),np.tile(-np.arange(5)/4,2)))
 def test_monitor_rejects_host_access(self):
  for expression in ['__import__("os")','xs.__class__','open(1)','2**1000000','(lambda:1)()']:
   with self.subTest(expression=expression):
    with self.assertRaises((AssertionError,ValueError,SyntaxError)):F.interpret(expression,[1])
 def test_monitor_attack_checks_need_named_violation(self):
  task=F.make_task('engineering',999);self.assertTrue(F.valid(task['reference'],task));self.assertFalse(F.valid(task['reference'],task,[1]))
 def test_monitor_budget_missing_and_stage_disallowed_fail_before_network(self):
  import tempfile,json
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as directory:
   out=__import__('pathlib').Path(directory)
   with patch.object(F,'OUT',out):
    with self.assertRaises(FileNotFoundError):F.api('claude-sonnet-5-5','x','calibration-0/check',16)
    (out/'spend-authorization.json').write_text(json.dumps({'authorized':True,'additional_cap_usd':3,'allowed_stages':['calibration']}))
    with self.assertRaises(AssertionError):F.api('claude-sonnet-5-5','x','confirmation-0/check',16)
 def test_monitor_provider_modes_and_recorded_cost_without_network(self):
  import tempfile,json,io,os
  from pathlib import Path
  from unittest.mock import patch
  bodies=[]
  def response(request,timeout=120):
   body=json.loads(request.data)
   if request.full_url.endswith('/count_tokens'):return io.BytesIO(json.dumps({'input_tokens':5}).encode())
   bodies.append(body);return io.BytesIO(json.dumps({'content':[{'type':'text','text':'{}'}],'usage':{'input_tokens':5,'output_tokens':2},'stop_reason':'end_turn'}).encode())
  with tempfile.TemporaryDirectory() as directory:
   out=Path(directory);(out/'spend-authorization.json').write_text(json.dumps({'authorized':True,'additional_cap_usd':3,'allowed_stages':['calibration']}))
   with patch.object(F,'OUT',out),patch.dict(os.environ,{'ANTHROPIC_API_KEY':'test-local-placeholder'}),patch.object(F.urllib.request,'urlopen',side_effect=response):
    for model in F.PRICES:F.api(model,'x','calibration-0/'+model,16)
   self.assertEqual([b['thinking']['type'] for b in bodies],['between_tools','disabled']);ledger=json.loads((out/'spend.json').read_text());self.assertEqual(ledger['calls'],2);self.assertEqual(ledger['pending'],{});self.assertLess(ledger['committed_usd'],.001)
 def test_fresh_task_streams(self):
  a=F.make_task('calibration',1);b=F.make_task('confirmation',1);self.assertNotEqual(a['public_tests'],b['public_tests']);self.assertEqual(len(b['private_tests']),32)
if __name__=='__main__':unittest.main()
