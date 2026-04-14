from __future__ import annotations
import subprocess
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent

def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"\n==> Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)

def main() -> int:
    try:
        # 1) run shell stage
        # sh = PIPELINE / "sample_resourcedumps.sh"
        # run(["bash", str(sh)])

        # 2) run python stages
        run([sys.executable, str(PIPELINE / "01_parse_rdf.py"), "-max-document", "10000"])
        run([sys.executable, str(PIPELINE / "02_process_text.py")])
        run([sys.executable, str(PIPELINE / "run_embedding_clustering.py")])

        print("\n✅ Pipeline finished")
        return
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Step failed with exit code {e.returncode}")
        return e.returncode

if __name__ == "__main__":
    raise SystemExit(main())
