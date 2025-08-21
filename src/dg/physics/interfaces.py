from typing import List

class Interface:
    name: str
    marker: int # this relates to the meshes interface tags

    def __init__(self, name: str, marker: int) -> None:
        self.name = name
        self.marker = marker

class InterfaceCollection:
    _interfaces: List[Interface]

    def __init__(self, interfaces: List[Interface]) -> None:
        self._interfaces = interfaces