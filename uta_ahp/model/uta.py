from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from uta_ahp.model.base import Model
from uta_ahp.utils import Order

if TYPE_CHECKING:
    import pandas as pd


class Uta(Model):
    def __init__(self, pairwise_comparisons: pd.DataFrame) -> None:
        self.pairwise = pairwise_comparisons
        self.pairwise["value"] = self.pairwise["value"].apply(Order.from_str)

    @staticmethod
    def command(
        dataset: Path = Path("data/dataset.csv"),
        pairwise_comparisons: Path = Path("data/pairwise.csv"),
    ) -> None:
        """UTA method."""
        import pandas as pd

        df = pd.read_csv(dataset, index_col=0)
        df_pairwise = pd.read_csv(pairwise_comparisons)
        model = Uta(df_pairwise)
        model.solve(df)

    def solve(self, dataset: pd.DataFrame) -> pd.Series: ...
