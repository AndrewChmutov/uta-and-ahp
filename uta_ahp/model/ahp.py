from __future__ import annotations

from typing import TYPE_CHECKING

from uta_ahp.model.base import Model

if TYPE_CHECKING:
    import pandas as pd


class AHP(Model):
    @staticmethod
    def command() -> None:
        """AHP method."""

    def solve(self) -> pd.Series: ...
