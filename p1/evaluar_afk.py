import subprocess
import sys
import os
from pathlib import Path

base = "seguir_modelo_sac_"
modelos = ["var_"]

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

def echo_to_file(text, dst):
    """Equivalent of: echo "<text>" > dst"""
    dst = Path(dst)
    print(f"\n>>> echo <text> > {dst}", flush=True)
    dst.write_text(text + "\n", encoding="utf-8")

def evaluate():
    ev = [sys.executable, "entrenar_seguir.py", "--evaluar", "--pasos", "600", "--episodios", "2"]
    for tipo in modelos:
        for i in range(1, 3):
            echo_to_file(f"RUTA_MODELO = \"{base}{tipo}{i+1}\"", "./.env")
            run(ev)

if __name__ == "__main__":
    evaluate()
    run(["git", "add", "--all"])
    run(["git", "commit", "-m", "logs contr"])
    run(["git", "push"])
    print("\nAll done.")