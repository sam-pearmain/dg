from typing import Optional

class Error(Exception):
    """Base error class"""
    pass

class UninitError(Error):
    """An error arrising from something that is uninitialised"""
    def __init__(self, message: Optional[str]):
        self.message = "used before initialisation" if message is None else message
        super().__init__(self.message)

class TodoError(Error):
    """An error arrising from something which has been marked 'todo'"""
    def __init__(self, message: Optional[str]):
        self.message = "feature not yet implemented" if message is None else message
        super().__init__(self.message)

class MeshReadError(Error):
    """An error arrising when reading in a faulty mesh"""
    def __init__(self, message: Optional[str]):
        self.message = "faulty mesh detected" if message is None else message
        super().__init__(self.message)