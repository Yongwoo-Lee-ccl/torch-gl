"""Run the unchanged stage harness under an explicit common scheduling request.

On macOS this requests USER_INITIATED QoS for the current thread. This is a
scheduling preference, not CPU affinity or a guarantee of exclusive CPU access.
"""

import ctypes
import json
import os
import platform
import runpy
import sys
from pathlib import Path


def request_qos():
    result = {
        "cpu_affinity": "unsupported on macOS" if sys.platform == "darwin" else "external taskset",
        "qos": None,
    }
    if sys.platform != "darwin":
        return result
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    library.pthread_self.restype = ctypes.c_void_p
    library.pthread_set_qos_class_self_np.argtypes = [ctypes.c_uint, ctypes.c_int]
    library.pthread_get_qos_class_np.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_int),
    ]
    qos, relative = ctypes.c_uint(), ctypes.c_int()
    before = library.pthread_get_qos_class_np(
        library.pthread_self(), ctypes.byref(qos), ctypes.byref(relative)
    )
    result["previous_qos"] = qos.value if before == 0 else None
    status = library.pthread_set_qos_class_self_np(0x19, 0)
    if status:
        raise OSError(status, "pthread_set_qos_class_self_np failed")
    status = library.pthread_get_qos_class_np(
        library.pthread_self(), ctypes.byref(qos), ctypes.byref(relative)
    )
    if status or qos.value != 0x19:
        raise RuntimeError("requested QoS could not be verified")
    result.update(
        qos="QOS_CLASS_USER_INITIATED", qos_numeric=qos.value, relative_priority=relative.value
    )
    return result


if __name__ == "__main__":
    control = request_qos()
    print("Execution control: " + json.dumps(control), flush=True)
    if sys.argv[1:] == ["--probe"]:
        raise SystemExit(0)
    checkout = Path(sys.argv[1]).resolve()
    arguments = sys.argv[2:]
    os.chdir(checkout)
    sys.path.insert(0, str(checkout / "src"))
    script = checkout / "scripts/benchmark_switch_stack.py"
    sys.argv = [str(script)] + arguments
    runpy.run_path(str(script), run_name="__main__")
    output = Path(arguments[arguments.index("--output") + 1])
    report = json.loads(output.read_text())
    report["execution_control"] = control
    report["runner_python"] = platform.python_version()
    output.write_text(json.dumps(report, indent=2) + "\n")
