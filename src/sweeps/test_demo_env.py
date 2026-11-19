import subprocess
import sys

sys.path.insert(0, "/home/fmannella/Dropbox/Projects/current/Kick-starting-Sources/.claude/worktrees/l2-objects/src")
import policy_demos as pd_  # noqa: E402

out = subprocess.run([sys.executable, "-c", "import box2dsim, SMEnv; print(box2dsim.__file__); print(SMEnv.__file__)"], env=pd_.ENV, capture_output=True, text=True, cwd="/tmp")
print(out.stdout.strip(), out.stderr.strip()[-300:])
print("PYTHONPATH in the subprocess environment:", pd_.ENV["PYTHONPATH"])
