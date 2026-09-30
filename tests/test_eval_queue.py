"""Offline queue regression tests; synthetic code only, no GPU/network."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/evaluation"))
import eval_queue as Q
import queue_data as D
E=Q.E

class QueueTests(unittest.TestCase):
    def test_default_is_separated_generation_256(self):
        argv=['eval_queue.py','worker','--plan','x','--gpu','0','--internal-code-execution']
        with patch.object(Q,'worker') as worker, patch.object(sys,'argv',argv):
            Q.main()
        a=worker.call_args.args[0]
        self.assertEqual((a.phase,a.concurrency,a.score_buffer),('generate',256,256))
        Q.GENERATION_ONLY=False

    def test_cli_accepts_32_16_128(self):
        argv=['eval_queue.py','worker','--plan','plan.json','--gpu','0','--concurrency','32','--score-workers','16','--score-buffer','128','--internal-code-execution']
        with patch.object(Q,'worker') as worker, patch.object(sys,'argv',argv):
            Q.main()
        a=worker.call_args.args[0]
        self.assertEqual((a.concurrency,a.score_workers,a.score_buffer),(32,16,128))

    def test_cli_accepts_64_and_rejects_over_limit_or_small_buffer(self):
        base=['eval_queue.py','worker','--plan','plan.json','--gpu','0','--phase','combined','--score-workers','16','--internal-code-execution']
        with patch.object(Q,'worker') as worker, patch.object(sys,'argv',base+['--concurrency','64','--score-buffer','128']):
            Q.main()
        self.assertEqual(worker.call_args.args[0].concurrency,64)
        for concurrency,buffer in [('65','128'),('64','32')]:
            with patch.object(sys,'argv',base+['--concurrency',concurrency,'--score-buffer',buffer]),self.assertRaises(SystemExit):
                Q.main()

    def test_generation_256_only(self):
        base=['eval_queue.py','worker','--plan','x','--gpu','0','--concurrency','256','--score-buffer','256','--internal-code-execution']
        with patch.object(Q,'worker') as w,patch.object(sys,'argv',base+['--phase','generate']): Q.main()
        self.assertEqual(w.call_args.args[0].concurrency,256)
        Q.GENERATION_ONLY=False
        with patch.object(sys,'argv',base+['--phase','combined']),self.assertRaises(SystemExit): Q.main()

    def test_native_context_preserves_full_output_cap(self):
        from context_check import check_items,CONTEXT_LENGTH
        class Tok:
            def apply_chat_template(self,messages,**kwargs):
                assert kwargs["return_dict"] is False
                return [1]*messages
        self.assertEqual(CONTEXT_LENGTH,8192)
        report=check_items(Tok(),[{"id":"fit","messages":4096},{"id":"long","messages":4097}],4096)
        self.assertEqual([x["id"] for x in report["oversized"]],["long"])

    def test_startup_recovery_preserves_clock_and_rejects_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/"old";old.mkdir();data=old/"data.json";E.write_new(data,{})
            plan={"profile":D.PROFILE,"jobs":[{"id":"sft-0"}],"protocol":{},"data":{"x":{"path":str(data),"sha256":E.file_hash(data)}}}
            E.write_new(old/"plan.json",plan)
            state={"plan_sha256":E.file_hash(old/"plan.json"),"started":1,"admit_until":2,"deadline":3,"durations":[],"jobs":{"sft-0":{"status":"failed"}}}
            E.write_new(old/"state.json",state)
            new=Path(tmp)/"new"
            Q.recover_startup(argparse.Namespace(from_plan=old/"plan.json",out=new))
            recovered=E.read_json(new/"state.json")
            self.assertEqual(recovered["deadline"],3);self.assertEqual(recovered["jobs"],{})
            self.assertEqual(E.read_json(old/"state.json"),state)
            (old/"cells").mkdir();E.write_new(old/"cells/results.json",{})
            with self.assertRaises(ValueError):
                Q.recover_startup(argparse.Namespace(from_plan=old/"plan.json",out=Path(tmp)/"bad"))

    def test_trajectory(self):
        rows=[x for tier in D.tiers() for x in tier]
        self.assertEqual(len(rows),17)
        self.assertEqual(len(set(rows)),17)
        self.assertEqual([x for x in rows if x[0]=="sft"],[("sft",0)])
        for mode in ("atomic","fixed"):
            self.assertEqual(sorted(x[1] for x in rows if x[0]==mode),[40,80,120,156,200,240,280,312])
        self.assertEqual(D.SEEDS,(42,43,44))

    def test_journal_repairs_only_interrupted_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"rows"
            p.write_bytes(b'{"id":"a"}')
            self.assertEqual(set(Q.journal(p)),{"a"})
            Q.append(p,{"id":"b"})
            self.assertEqual(set(Q.journal(p)),{"a","b"})
            with p.open("ab") as f: f.write(b'{"id":')
            self.assertEqual(len(Q.journal(p)),2)
            self.assertTrue(Path(str(p)+".interrupted-tail").exists())
            p.write_bytes(b'broken\n')
            with self.assertRaises(ValueError): Q.journal(p)

    def test_tier_deadline_failure_and_resume_clock(self):
        plan={"jobs":[{"id":"a","tier":0},{"id":"b","tier":1}],"hours":20,"admit_hours":18}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            state=Q.read_state(root,"hash",plan)
            self.assertEqual(Q.read_state(root,"hash",plan)["deadline"],state["deadline"])
            now=state["started"]
            self.assertEqual([x["id"] for x in Q.next_jobs(plan,state,now)],["a"])
            state["jobs"]["a"]={"status":"completed"}
            self.assertEqual([x["id"] for x in Q.next_jobs(plan,state,now)],["b"])
            self.assertEqual(Q.next_jobs(plan,state,state["admit_until"]),[])
            state["jobs"]["b"]={"status":"failed"}
            with self.assertRaises(RuntimeError): Q.next_jobs(plan,state,now)

    def test_internal_qualification(self):
        self.assertEqual(len(Q.preflight(sys.executable)),10)
        with self.assertRaises(Q.Deadline): Q.score_job(sys.executable,{},time.time()-1)
        with self.assertRaises(RuntimeError):
            Q.score_job(sys.executable,{"profile":"author-code"},time.time()+10)

    def test_functional_prompt_and_source_row_hash(self):
        prompt=D.prompt("live-code-bench-v6",{"fn_name":"add","prompt":"sum","starter_code":"class Solution:"})
        self.assertIn("Solution",prompt[0]["content"])
        self.assertIn("add",prompt[0]["content"])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"data";p.write_bytes(b'{}')
            with self.assertRaises(ValueError):
                D.score_item("live-code-bench-v6",{"raw":{"path":str(p),"offset":0,"length":2,"sha256":"wrong"}})

    def test_generation_spool_then_cpu_scoring_without_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"data.json"
            items=[{"id":str(i),"messages":[{"role":"user","content":"sum"}],"gold":"#### 2"} for i in range(4)]
            E.write_new(data,{"items":items})
            plan={"data":{"gsm8k":{"path":str(data),"sha256":E.file_hash(data),"count":4}}}
            job={"id":"a","checkpoint":{"sha256":"cp"}}
            args=argparse.Namespace(phase="generate",min_free_gib=0,concurrency=2,score_workers=2,score_buffer=4,score_python=sys.executable)
            def generate(base,item,benchmark,seed,deadline):
                return {"id":item["id"],"seed":seed,"request_sha256":E.digest(E.encoded(E.generation_payload("eval-gemma",item,benchmark,seed))),"response":{"choices":[{"finish_reason":"stop","message":{"content":"#### 2"}}]}}
            with patch.object(Q,"generate_one",side_effect=generate),patch.object(Q,"score_job",side_effect=AssertionError("GPU must not score")):
                Q.run_cell(root,plan,"ph",job,"gsm8k",42,"",{},args,time.time()+20)
            cell=root/"cells/a/gsm8k/42"
            self.assertTrue((cell/"generation-complete.json").exists())
            self.assertFalse((cell/"metrics.json").exists())
            before=E.file_hash(cell/"responses.jsonl")
            args.phase="score"
            with patch.object(Q,"generate_one",side_effect=AssertionError("CPU must not generate")),patch.object(Q,"score_job",return_value={"passed":True}):
                result=Q.run_cell(root,plan,"ph",job,"gsm8k",42,None,{},args,time.time()+20)
            self.assertEqual(result["score"],1)
            self.assertEqual(E.file_hash(cell/"responses.jsonl"),before)

    def test_slow_scores_do_not_consume_generation_slots(self):
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"data.json"
            items=[{"id":str(i),"messages":[{"role":"user","content":"sum"}],"gold":"#### 2"} for i in range(30)]
            E.write_new(data,{"items":items})
            plan={"data":{"gsm8k":{"path":str(data),"sha256":E.file_hash(data),"count":30}}}
            job={"id":"atomic-40","checkpoint":{"sha256":"cp"}}
            args=argparse.Namespace(concurrency=4,score_workers=2,score_buffer=8,score_python=sys.executable)
            release=threading.Event();filled=threading.Event();lock=threading.Lock();counts={"generated":0,"decoded":0}
            def generate(base,item,benchmark,seed,deadline):
                with lock:
                    counts["generated"]+=1
                    if counts["generated"]>=10: filled.set()
                return {"id":item["id"],"seed":seed,"request_sha256":E.digest(E.encoded(E.generation_payload("eval-gemma",item,benchmark,seed))),
                        "response":{"choices":[{"finish_reason":"stop","message":{"content":"#### 2"}}]}}
            def decode(benchmark,item):
                with lock: counts["decoded"]+=1
                return item
            def score(*args):
                if not release.wait(5): raise RuntimeError("test release timeout")
                return {"passed":True}
            evidence={}
            def monitor():
                evidence["filled"]=filled.wait(3)
                time.sleep(.05)
                with lock: evidence.update(counts)
                release.set()
            t=threading.Thread(target=monitor);t.start()
            with patch.object(Q,"generate_one",side_effect=generate),patch.object(D,"score_item",side_effect=decode),patch.object(Q,"score_job",side_effect=score):
                result=Q.run_cell(root,plan,"plan",job,"gsm8k",42,"",{},args,time.time()+20)
            t.join()
            self.assertTrue(evidence["filled"])
            self.assertEqual(evidence["generated"],10) # 8 buffered/in-flight + 2 scorers
            self.assertEqual(evidence["decoded"],2) # no queued decoded test payloads
            self.assertEqual(result["count"],30)

    def test_cell_resume_no_repeated_generation_or_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"data.json"
            items=[{"id":str(i),"messages":[{"role":"user","content":"sum"}],"gold":"#### 2"} for i in range(3)]
            E.write_new(data,{"items":items})
            plan={"data":{"gsm8k":{"path":str(data),"sha256":E.file_hash(data),"count":3}}}
            job={"id":"atomic-40","checkpoint":{"sha256":"cp"}}
            args=argparse.Namespace(concurrency=1,score_workers=1,score_python=sys.executable)
            def generate(base,item,benchmark,seed,deadline):
                return {"id":item["id"],"seed":seed,"request_sha256":E.digest(E.encoded(E.generation_payload("eval-gemma",item,benchmark,seed))),
                        "response":{"choices":[{"finish_reason":"stop","message":{"content":"#### 2"}}]}}
            cell=root/"cells/atomic-40/gsm8k/42";cell.mkdir(parents=True)
            E.write_new(cell/"contract.json",Q.cell_contract("plan",job,plan["data"],"gsm8k",42,{}))
            first=generate("",items[0],"gsm8k",42,0);Q.append(cell/"responses.jsonl",first)
            with patch.object(Q,"generate_one",side_effect=generate) as g:
                m=Q.run_cell(root,plan,"plan",job,"gsm8k",42,"",{},args,time.time()+30)
                self.assertEqual(g.call_count,2);self.assertEqual(m["score"],1.)
            with patch.object(Q,"generate_one",side_effect=AssertionError("repeat")), patch.object(Q,"score_job",side_effect=AssertionError("repeat")):
                Q.run_cell(root,plan,"plan",job,"gsm8k",42,"",{},args,time.time()+30)
            with (cell/"scores.jsonl").open("ab") as f: f.write(b" ")
            with self.assertRaises(ValueError):
                Q.run_cell(root,plan,"plan",job,"gsm8k",42,"",{},args,time.time()+30)


class GpuSharingGateTests(unittest.TestCase):
    """Tolerating an occupied GPU must be opt-in; the default keeps the historical gate."""

    def test_default_constant_is_the_historical_gate(self):
        self.assertEqual(Q.DEFAULT_TOLERATE_USED_MIB,1024)

    def test_gate_semantics_keep_the_one_gib_boundary(self):
        self.assertTrue(Q.gpu_occupancy_ok(0,1024))
        self.assertTrue(Q.gpu_occupancy_ok(1023,1024))
        self.assertFalse(Q.gpu_occupancy_ok(1024,1024))
        self.assertFalse(Q.gpu_occupancy_ok(8698,1024))
        self.assertTrue(Q.gpu_occupancy_ok("8698",16384))
        self.assertFalse(Q.gpu_occupancy_ok(16384,16384))

    def test_worker_cli_defaults_to_refusing_an_occupied_gpu(self):
        argv=['eval_queue.py','worker','--plan','x','--gpu','3','--internal-code-execution']
        with patch.object(Q,'worker') as worker, patch.object(sys,'argv',argv):
            Q.main()
        try:
            self.assertEqual(worker.call_args.args[0].tolerate_used_mib,1024)
        finally:
            Q.GENERATION_ONLY=False

    def test_worker_cli_accepts_an_explicit_tolerance(self):
        argv=['eval_queue.py','worker','--plan','x','--gpu','3',
              '--tolerate-used-mib','16384','--internal-code-execution']
        with patch.object(Q,'worker') as worker, patch.object(sys,'argv',argv):
            Q.main()
        try:
            self.assertEqual(worker.call_args.args[0].tolerate_used_mib,16384)
        finally:
            Q.GENERATION_ONLY=False


if __name__=="__main__": unittest.main()
