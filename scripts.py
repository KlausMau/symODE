import shutil
import subprocess
from pathlib import Path


def run_ruff(arguments: list[str]):
    subprocess.run(["ruff", *arguments], check=True)


def autofix():
    run_ruff(["format", "."])
    run_ruff(["check", "--fix"])


def typecheck():
    subprocess.run(["mypy", "."], check=True)


def test():
    subprocess.run(["pytest", "-v", "-ra"], check=True)


def notebooks():
    notebook_paths = sorted(Path("examples").glob("*.ipynb"))
    output_dir = Path(".notebook-output")
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir()

    for notebook_path in notebook_paths:
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=300",
                "--ExecutePreprocessor.kernel_name=python3",
                "--output-dir",
                str(output_dir),
                str(notebook_path),
            ],
            check=True,
        )


def all_checks():
    run_ruff(["format", "--check", "."])
    run_ruff(["check"])
    typecheck()
    test()
