import subprocess


def fmt():
    subprocess.run(
        ["ruff", "format", "--check", "."],
        check=True,
    )


def autofix():
    subprocess.run(
        ["ruff", "format", "."],
        check=True,
    )


def lint():
    subprocess.run(["ruff", "check"], check=True)


def typecheck():
    subprocess.run(["mypy", "."], check=True)


def test():
    subprocess.run(["pytest", "-v", "-ra"], check=True)


def all_checks():
    fmt()
    lint()
    typecheck()
    test()
