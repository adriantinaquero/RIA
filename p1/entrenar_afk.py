import subprocess
import sys
from pathlib import Path


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


def main():
    train = [sys.executable, "entrenar_seguir.py", "--entrenar"]

    # Run 2
    run(train)
    rename("./seguir_logs/SAC_1", "./seguir_logs/SAC_CONTR_2")
    rename("./seguir_modelo.zip", "./seguir_modelo_sac_contr_2.zip")

    # Run 3
    run(train)
    rename("./seguir_logs/SAC_1", "./seguir_logs/SAC_CONTR_3")
    rename("./seguir_modelo.zip", "./seguir_modelo_sac_contr_3.zip")

    # Git
    run(["git", "add", "--all"])
    run(["git", "commit", "-m", "logs contr"])
    run(["git", "push"])

    print("\nAll done.")


if __name__ == "__main__":
    main()