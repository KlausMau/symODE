import os

def format():
    os.system("black ./examples/ ./symode/ ./tests")

def lint():
    os.system("pylint ./examples/ ./symode/ ./tests")

def typecheck():
    os.system("mypy .")

def test():
    os.system("pytest -v -ra")
