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
    for j in range(3):
        modelos = [f"C{k}=simulacion_simulador/mejor_genoma_c{k}_{j}.pkl" for k in range(1, 5)]
        train_cmd = [
            sys.executable, "p2_entrenar.py",
            "--evaluar", "--episodios", "1",
            "--modelos", *modelos,
            "--csv-pasos", f"pasos_{j}.csv",
            "--json-metricas", f"metricas_{j}.json",
            "--pasos-max", "400"
        ]
        run(train_cmd)

if __name__ == "__main__":
    train()
    # run(["git", "add", "--all"])
    # run(["git", "commit", "-m", "logs contr"])
    # run(["git", "push"])
    print("\nAll done.")