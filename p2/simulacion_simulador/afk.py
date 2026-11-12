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
    for i in [1,2,3,4]:
        for j in range(3):
            train_cmd = [sys.executable, "train.py", "--entrenar", "--ruta-config",
                        "config.txt", "--generaciones", "200", "--pasos-max", "150",
                        "--imagen-pista", "track.png", "--imagen-robot", "robobo.png",
                        "--dt", "0.3", "--modo", str(i), "--ruta-modelo", f"mejor_genoma_c{i}_{j}.plk",
                        "--prefijo", f"C{i}_{j}", "--archivo-final", f"evolucion_c{i}_{j}.csv"]
            run(train_cmd)

if __name__ == "__main__":
    train()
    # run(["git", "add", "--all"])
    # run(["git", "commit", "-m", "logs contr"])
    # run(["git", "push"])
    print("\nAll done.")