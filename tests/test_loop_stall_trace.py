import subprocess
import sys


def test_native_stall_trace_survives_gil_hold_and_cancels(tmp_path):
    trace = tmp_path / "trace.log"
    code = r"""
import ctypes
import sys
import time
from pathlib import Path
from myumiq_vrchat.loop_diagnostics import LoopStallTrace
trace = LoopStallTrace(Path(sys.argv[1]), timeout_s=.1)
for _ in range(20):
    trace.tick()
    time.sleep(.001)
assert Path(sys.argv[1]).stat().st_size == 0
if sys.platform == 'win32':
    ctypes.PyDLL('kernel32').Sleep(250)
else:
    time.sleep(.25)
trace.close()
size = Path(sys.argv[1]).stat().st_size
assert size > 0
time.sleep(.15)
assert Path(sys.argv[1]).stat().st_size == size
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(trace)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert "Timeout" in trace.read_text()
