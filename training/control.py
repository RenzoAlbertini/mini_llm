import json
import os
import signal
import subprocess
import time
from pathlib import Path


DEFAULT_CONTROL_PATH = Path("data/logs/training_control.json")
DEFAULT_PID_PATH = Path("data/logs/training.pid")


def _read_json(path, default):
    path = Path(path)
    if not path.exists():
        return dict(default)
    try:
        return {**default, **json.loads(path.read_text(encoding="utf-8"))}
    except (OSError, json.JSONDecodeError):
        return dict(default)


def read_control(path=DEFAULT_CONTROL_PATH):
    return _read_json(path, {"paused": False, "stop_requested": False, "updated_at": None})


def write_control(paused=None, stop_requested=None, path=DEFAULT_CONTROL_PATH):
    path = Path(path)
    state = read_control(path)
    if paused is not None:
        state["paused"] = bool(paused)
    if stop_requested is not None:
        state["stop_requested"] = bool(stop_requested)
    state["updated_at"] = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def write_pid(pid=None, path=DEFAULT_PID_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid or os.getpid()), encoding="utf-8")
    return path


def read_pid(path=DEFAULT_PID_PATH):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def clear_pid(path=DEFAULT_PID_PATH):
    path = Path(path)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def is_process_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        try:
            import ctypes

            process = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if process:
                ctypes.windll.kernel32.CloseHandle(process)
                return True
            return False
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_process(pid, force=False):
    if not pid:
        return False, "pid mancante"
    if os.name == "nt":
        args = ["taskkill", "/PID", str(pid), "/T"]
        if force:
            args.append("/F")
        result = subprocess.run(args, capture_output=True, text=True)
        ok = result.returncode == 0
        return ok, (result.stdout or result.stderr).strip()
    try:
        os.kill(pid, signal.SIGTERM)
        return True, "SIGTERM inviato"
    except OSError as exc:
        return False, str(exc)


def wait_while_paused(path=DEFAULT_CONTROL_PATH, poll_seconds=2.0):
    while True:
        state = read_control(path)
        if state.get("stop_requested"):
            raise KeyboardInterrupt("stop richiesto dalla dashboard")
        if not state.get("paused"):
            return
        print("training in pausa dalla dashboard")
        time.sleep(poll_seconds)
