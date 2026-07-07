from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_selftest_script():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "selftest.py")], cwd=str(ROOT), text=True, capture_output=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
