import os

def format():
    os.system("black ./examples/ ./symode/ ./tests")

def lint():
    os.system("pylint ./examples/ ./symode/ ./tests")

def test():
    os.system("pytest")
