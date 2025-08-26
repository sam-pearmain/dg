from typing import Optional, NoReturn

class TodoError(Exception):
    def __init__(self, message: Optional[str]) -> None:
        message = message if message is not None else "todo error"
        super().__init__(message)

def todo(str: str) -> NoReturn:
    raise TodoError(str)