import os
import subprocess


def fmt():
    os.system("ruff format ./examples/ ./symode/ ./tests")

def lint():
    subprocess.run(["ruff", "check"], check=True)
    
def typecheck():
    os.system("mypy .")

def test():
    subprocess.run(["pytest", "-v", "-ra"], check=True)

def all_checks():
    lint()
    test()