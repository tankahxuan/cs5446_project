"""Execute 06_agent_evaluation.ipynb (training skipped) to refresh charts."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = PROJECT_ROOT / "notebooks" / "06_agent_evaluation.ipynb"


def main() -> None:
    kernels_root = Path(tempfile.mkdtemp(prefix="jupyter_path_")) / "kernels"
    kernel_dir = kernels_root / "citylearn-temp"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "citylearn-temp",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )

    # jupyter_client discovers kernels under <jupyter_path>/kernels/<name>
    import os

    os.environ["JUPYTER_PATH"] = str(kernels_root.parent)

    nb = nbformat.read(NB_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="citylearn-temp",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    print(f"Executing {NB_PATH} with cwd={PROJECT_ROOT}", flush=True)
    try:
        client.execute()
    finally:
        shutil.rmtree(kernels_root.parent, ignore_errors=True)

    NB_PATH.write_text(nbformat.writes(nb), encoding="utf-8")
    print(f"Wrote executed notebook outputs to {NB_PATH}", flush=True)


if __name__ == "__main__":
    main()
