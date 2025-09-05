import os

def fmt():
    os.system("ruff format ./examples/ ./symode/ ./tests")

def lint():
    os.system("ruff check ./examples/ ./symode/ ./tests")

def typecheck():
    os.system("mypy .")

def test():
    os.system("pytest -v -ra")
