from enum import StrEnum, auto


class Order(StrEnum):
    PREFERENCE = "P"
    INDIFFERENCE = "I"


class Domain(StrEnum):
    CONTINUOUS = "C"
    INTEGER = "I"


class CriterionType(StrEnum):
    GAIN = auto()
    COST = auto()
