import subprocess,sys
r=subprocess.run([sys.executable,"short_arena_v2.py"])
raise SystemExit(r.returncode)
