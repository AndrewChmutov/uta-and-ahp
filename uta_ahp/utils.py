from enum import IntEnum, auto


class Order(IntEnum):
    PREFERENCE = auto()
    INDIFFERENCE = auto()

    @classmethod
    def from_str(cls, value: str) -> int:
        match value:
            case "P":
                return cls.PREFERENCE
            case "I":
                return cls.INDIFFERENCE
            case _:
                raise ValueError("Invalid preference value")
