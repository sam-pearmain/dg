from typing import Optional

class Error(Exception):
    """Base error class"""
    pass

# -- general errors --

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

class NotSupportedError(Error):
    """An error arrising when something isn't supported"""
    def __init__(self, message: Optional[str]):
        self.message = "feature not supported" if message is None else message
        super().__init__(self.message)

# -- mesh errors -- 

class MeshError(Error):
    """An error arrising when from a faulty mesh"""
    def __init__(self, message: Optional[str]):
        self.message = "faulty mesh detected" if message is None else message
        super().__init__(self.message)

class MeshReadError(MeshError):
    """An error arrising from reading in a faulty mesh"""
    def __init__(self, message):
        super().__init__(message)