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
    train_cmd = [sys.executable, "p2_entrenar.py", "--entrenar", "--generaciones", "1"]
    run(train_cmd)

    reanudar_entrenamiento()

def reanudar_entrenamiento(i = 1):
    temp_i = i
    try:
        for j in range(i, GENERACIONES):
            train_cmd = [sys.executable, "p2_entrenar.py", "--reanudar-entrenamiento",
                        "--pasos-faltantes", "1", "--checkpoint", str(j)]
            run(train_cmd)
            temp_i = j
        
    except KeyboardInterrupt:
        return
    except:
        reanudar_entrenamiento(temp_i)

if __name__ == "__main__":
    train()
    # run(["git", "add", "--all"])
    # run(["git", "commit", "-m", "logs contr"])
    # run(["git", "push"])
    print("\nAll done.")