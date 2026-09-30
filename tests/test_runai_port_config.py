import importlib.util,os,unittest
from pathlib import Path
from unittest.mock import patch
p=Path(__file__).resolve().parents[1]/"kdflow/port_config.py"
spec=importlib.util.spec_from_file_location("port_config",p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class PortTests(unittest.TestCase):
 def test_default_without_gpu_environment(self):
  with patch.dict(os.environ,{},clear=True):
   self.assertEqual(m.configured_port("KDFLOW_ROLLOUT_PORT_BASE",15000),15000)
   self.assertIsNone(m.configured_port("KDFLOW_ROUTER_PROMETHEUS_PORT"))
 def test_two_jobs_disjoint(self):
  for slot in (0,1):
   with patch.dict(os.environ,{"KDFLOW_ROLLOUT_PORT_BASE":str(15000+1000*slot),"KDFLOW_ROUTER_PROMETHEUS_PORT":str(20000+1000*slot)}):
    self.assertEqual(m.configured_port("KDFLOW_ROLLOUT_PORT_BASE"),15000+1000*slot)
    self.assertEqual(m.configured_port("KDFLOW_ROUTER_PROMETHEUS_PORT"),20000+1000*slot)
 def test_invalid_ports_fail(self):
  for value in ("0","1023","65536","abc","0,1"):
   with patch.dict(os.environ,{"TEST_PORT":value}):
    with self.assertRaises(ValueError):m.configured_port("TEST_PORT")
if __name__=="__main__":unittest.main()
