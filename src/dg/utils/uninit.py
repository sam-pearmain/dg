class Uninit:
    def __repr__(self) -> str:
        return "<uninit>"
    
    def __bool__(self) -> bool:
        return False
    
    def isuninit(self) -> bool:
        return True

def isuninit(obj: object):
    return isinstance(obj, Uninit)

def tests():
    class ABC:
        a: Uninit = Uninit()
    
    print(isuninit(ABC.a))

if __name__ == "__main__":
    tests()