import subprocess
import sys
import os
from pathlib import Path

GENERACIONES = 50

def run(cmd):
    """Run a command, wait for it to finish, and stop the script if it fails."""
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def rename(src, dst):
    src, dst = Path(src), Path(dst)
    print(f"\n>>> rename {src} -> {dst}", flush=True)
    if not src.exists():
        sys.exit(f"ERROR: {src} does not exist")
    if dst.exists():
        sys.exit(f"ERROR: {dst} already exists")
    src.rename(dst)

def train():
    run([sys.executable, "simulacion_simulador/afk.py"])
    run([sys.executable, "mass_plot.py"])
    run([sys.executable, "plot_redes.py"])

if __name__ == "__main__":
    train()
    # run(["git", "add", "--all"])
    # run(["git", "commit", "-m", "logs contr"])
    # run(["git", "push"])
    print("\nAll done.")