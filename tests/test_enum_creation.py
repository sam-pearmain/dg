import time

from enum import Enum, auto

class Numbers(Enum):
    One = auto()
    Two = auto()

    def num(self):
        match self:
            case self.One: return 1
            case self.Two: return 2

    def number(self):
        # match statements are way way more performant
        return {
            self.One: 1, 
            self.Two: 2,
        }[self]

def one():
    return 1

def two():
    return 2

def one_or_two(num = Numbers.One):
    match num:
        case Numbers.One: return one()
        case Numbers.Two: return two()


start = time.perf_counter()
enums = [Numbers.One.number() for _ in range(20000)]
e_dur = time.perf_counter() - start

start = time.perf_counter()
ints = [one_or_two() for _ in range(20000)]
i_dur = time.perf_counter() - start

print(f"enums: {e_dur}, ints: {i_dur}, performance {i_dur / e_dur * 100}%")

# keeping functions inside enums is actually faster nice