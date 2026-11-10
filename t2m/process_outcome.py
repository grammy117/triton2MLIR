"""Bounded observation of an owned worker; no failure is inferred as success."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess


def record(path):
    data = Path(path).read_bytes()
    return {"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}


def started(directory, *, backend, program, arguments, implementation):
    marker = {"schema":"t2m.invocation-started.v1", "backend":backend,
        "program":record(program),"arguments":record(arguments),"implementation":implementation}
    (Path(directory) / "invocation-started.json").write_text(json.dumps(marker,sort_keys=True,indent=2)+"\n")


_LIMITS = """import os,resource,sys
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024*1024,64*1024*1024))
os.execvp(sys.argv[1],sys.argv[1:])
"""


def observe(command, directory, *, environment, timeout):
    """Retain exact byte streams and termination, including setup failures."""
    if not 0 < timeout <= 60:
        raise ValueError("worker observation requires a timeout at most 60 seconds")
    directory = Path(directory)
    output, error = directory / "stdout.bin", directory / "stderr.bin"
    with output.open("xb") as stdout, error.open("xb") as stderr:
        process = subprocess.Popen([command[0],"-c",_LIMITS,*command],stdout=stdout,stderr=stderr,
            env=environment,start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    outcome = {"schema":"t2m.worker-outcome.v1","returncode":process.returncode,"timed_out":timed_out,
        "stdout":record(output),"stderr":record(error),"command":command,"timeout_seconds":timeout,
        "limits":{"core_bytes":0,"file_bytes":64*1024*1024}}
    (directory / "outcome.json").write_text(json.dumps(outcome,sort_keys=True,indent=2)+"\n")
    return outcome
