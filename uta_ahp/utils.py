from enum import StrEnum, auto


class Order(StrEnum):
    PREFERENCE = "P"
    INDIFFERENCE = "I"


class Domain(StrEnum):
    CONTINUOUS = "C"
    INTEGER = "I"
    BINARY = "B"


class CriterionType(StrEnum):
    GAIN = auto()
    COST = auto()


class Mode(StrEnum):
    CLASSIC = auto()
    DISCRIMINATE = auto()
    CONSISTENT_SUBSETS = auto()
