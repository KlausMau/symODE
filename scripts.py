import subprocess


def run_ruff(arguments: list[str]):
    subprocess.run(["ruff", *arguments], check=True)


def autofix():
    run_ruff(["format", "."])


def typecheck():
    subprocess.run(["mypy", "."], check=True)


def test():
    subprocess.run(["pytest", "-v", "-ra"], check=True)


def all_checks():
    run_ruff(["format", "--check", "."])
    run_ruff(["check"])
    typecheck()
    test()
