"""Explicit company-internal scorer. Resource limits are NOT an OS sandbox.

Only the queue's opt-in internal profile invokes this process. Existing
contract_worker/bubblewrap profiles remain unchanged.
"""
import contextlib
import json
import os
import resource
import sys
from contract_eval import author_helpers
from contract_worker import load_lcb, math_score, package_version


def main():
    job=json.load(sys.stdin)
    if job.get("profile") != "company-internal-v1":
        raise ValueError("explicit company-internal profile required")
    benchmark=job["benchmark"]
    helpers=author_helpers()
    official=load_lcb() if benchmark=="live-code-bench-v6" else None
    versions={k:package_version(k) for k in ("numpy","sympy")}
    if versions["sympy"] is None:
        raise RuntimeError("internal profile requires sympy; no silent math fallback")
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**2,8*1024**2))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    item=job["item"]
    text=helpers["strip_thinking_content"](job["text"])[0]
    with open(os.devnull,"w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        if benchmark in {"gsm8k","math500"}:
            passed=math_score(helpers,benchmark,"author-code",item,text)
        elif benchmark=="mbpp":
            passed=helpers["_run_mbpp_tests"](helpers["_extract_code_block"](text),item["tests"],item.get("setup",""),timeout=10.)
        elif benchmark=="live-code-bench-v6":
            tests=item["tests"]
            sample={"input_output":json.dumps({"inputs":[t["input"] for t in tests],
                    "outputs":[t["output"] for t in tests],"fn_name":item.get("fn_name")})}
            result,_=official.run_test(sample,test=helpers["_extract_code_block"](text),timeout=10)
            passed=len(result)==len(tests) and all(bool(x==True) for x in result)
        else:
            raise ValueError("unknown benchmark")
    print(json.dumps({"passed":bool(passed),"versions":versions}))

if __name__=="__main__":
    try: main()
    except Exception as exc:
        print(json.dumps({"error":type(exc).__name__+": "+str(exc)}))
        sys.exit(1)
