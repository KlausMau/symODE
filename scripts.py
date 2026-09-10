import os
import subprocess


def fmt():
    subprocess.run(
        ["ruff", "format", "--check", "./examples/", "./symode/", "./tests/"],
        check=True,
    )


def lint():
    subprocess.run(["ruff", "check"], check=True)


def typecheck():
    os.system("mypy .")


def test():
    subprocess.run(["pytest", "-v", "-ra"], check=True)


def all_checks():
    fmt()
    lint()
    test()