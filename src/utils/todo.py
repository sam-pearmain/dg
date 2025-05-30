from .error import TodoError

def todo(str: str):
    raise TodoError(str)