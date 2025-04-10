from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Self

from typer import Typer

if TYPE_CHECKING:
    import pandas as pd
    from typer import Typer


class Model(ABC):
    types: ClassVar[list[type[Self]]] = []

    def __init_subclass__(cls) -> None:
        cls.types.append(cls)

    @staticmethod
    @abstractmethod
    def command(*args, **kwargs) -> None: ...

    @classmethod
    def name(cls) -> str:
        return cls.__name__.lower()

    @classmethod
    def register_commands(cls, app: Typer) -> None:
        for type in cls.types:
            type.register_command(app)

    @classmethod
    def register_command(cls, app: Typer) -> None:
        app.command(name=cls.name())(cls.command)

    @abstractmethod
    def solve(self, dataset: pd.DataFrame) -> pd.Series: ...
