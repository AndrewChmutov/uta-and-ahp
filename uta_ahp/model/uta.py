from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

from uta_ahp.model.base import Model
from uta_ahp.utils import CriterionType, Domain, Order

if TYPE_CHECKING:
    import gurobipy as gp
    import numpy as np
    import pandas as pd

    type Residuals = dict[int, tuple[gp.Var, gp.Var]]
    type CharCritPts = list[tuple[gp.Var, np.float64]]
    type CharPts = list[CharCritPts]


class Uta(Model):
    WEIGHT_UPPER_BOUND = .5

    def __init__(
        self,
        df: pd.DataFrame,
        criteria: pd.DataFrame,
        pairwise_comparisons: pd.DataFrame,
        n_characteristic_points: int,
        weight_lower_bound: float,
    ) -> None:
        self.df = df
        self.criteria = criteria
        self.criteria["domain"] = self.criteria["domain"].apply(Domain)
        self.criteria["type"] = self.criteria["type"].apply(CriterionType)
        self.pairwise = pairwise_comparisons
        self.pairwise["value"] = self.pairwise["value"].apply(Order)
        self.n_cp = n_characteristic_points
        self.lb = weight_lower_bound

    @staticmethod
    def command(
        dataset: Path = Path("data/dataset.csv"),
        criteria: Path = Path("data/criteria.csv"),
        pairwise_comparisons: Path = Path("data/pairwise.csv"),
        n_characteristic_points: int = 3,
        weight_lower_bound: float = 0.00
    ) -> None:
        """UTA method."""
        import pandas as pd

        df = pd.read_csv(dataset, index_col=0)
        df_criteria = pd.read_csv(criteria, index_col=0)
        df_pairwise = pd.read_csv(pairwise_comparisons)
        Uta(
            df=df,
            criteria=df_criteria,
            pairwise_comparisons=df_pairwise,
            n_characteristic_points=n_characteristic_points,
            weight_lower_bound=weight_lower_bound
        ).solve()

    def solve(self) -> pd.Series | None:
        import gurobipy as gp

        with self._model() as m:
            rs = self.add_residuals(m)
            self.set_objective(m, rs)
            cps = self.add_characteristic_points(m)

            self.add_preference_constraints(m, cps, rs)
            self.add_normalization_constraints(m, cps)
            self.add_monotonicity_constraints(m, cps)

            m.optimize()
            if m.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
                print("No feasible solutions")
                return

            self.print_table(m)
            print()
            self.print_values(m)
            print()
            self.print_char_points_corresponding_values(cps)
        # Non negativity constraints are covered by the lib by default

    @staticmethod
    @contextmanager
    def _model() -> Iterator[gp.Model]:
        """Disables output of `gurobipy`.

        Yields:
            Model with changed environment
        """
        import gurobipy as gp

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 1)
            env.start()
            with gp.Model(env=env) as m:
                yield m

    def add_residuals(self, m: gp.Model) -> Residuals:
        def var_factory(i: int, suffix: str) -> gp.Var:
            return m.addVar(vtype=Domain.CONTINUOUS, name=f"sig_{i}_{suffix}")

        return {
            i: (var_factory(i, "plus"), var_factory(i, "minus"))
            for i in set(chain(self.pairwise["a"], self.pairwise["b"]))
        }

    def set_objective(self, m: gp.Model, rs: Residuals) -> None:
        import gurobipy as gp

        # Sum over the flattened iterable
        m.setObjective(sum(chain.from_iterable(rs.values())), gp.GRB.MINIMIZE)

    def add_characteristic_points(self, m: gp.Model) -> CharPts:
        cps = []
        for crit, row in self.criteria.iterrows():
            crit_type = row.loc["type"]
            assert isinstance(crit_type, CriterionType)
            min_value = self.df[crit].min()
            max_value = self.df[crit].max()
            n = self.n_cp
            delta = (max_value - min_value) / (n - 1)

            match crit_type:
                case CriterionType.GAIN:
                    vals = [min_value + i * delta for i in range(n)]
                case CriterionType.COST:
                    vals = [max_value - i * delta for i in range(n)]

            # Note: set configured lower bound and upper bound
            cps.append([
                (
                    m.addVar(
                        lb=self.lb,
                        ub=self.WEIGHT_UPPER_BOUND,
                        vtype=Domain.CONTINUOUS,
                        name=f"{crit}_{i}"
                    ),
                    vals[i]
                )
                for i in range(n)
            ])

        return cps

    def add_preference_constraints(
        self,
        m: gp.Model,
        cps: CharPts,
        rs: Residuals,
    ) -> None:
        for _, row in self.pairwise.iterrows():
            a, b, pref = row.tolist()
            assert isinstance(pref, Order)

            def get_handside_expr(alternative: str) -> gp.LinExpr:
                crit_values = self.df.loc[alternative]
                sig_plus, sig_minus = rs[int(alternative)]
                _ = self.get_closest_characteristic_points
                closest_cps = _(cps, crit_values)
                return sum(closest_cps) - sig_plus + sig_minus

            lhs = get_handside_expr(a)
            rhs = get_handside_expr(b)
            m.update()
            print(lhs)
            print(rhs)
            print()
            match pref:
                case Order.PREFERENCE:
                    m.addConstr(lhs >= rhs)
                case Order.INDIFFERENCE:
                    m.addConstr(lhs == rhs)

    @classmethod
    def get_closest_characteristic_points(
        cls,
        cps: CharPts,
        values: list[float]
    ) -> list[gp.Var]:
        return [
            cls.get_closest_characteristic_point(crit_cps, value)
            for crit_cps, value in zip(cps, values)
        ]

    @staticmethod
    def get_closest_characteristic_point(
        crit_cps: CharCritPts,
        value: float
    ) -> gp.Var:
        vars, values = list(zip(*crit_cps))
        closest_index = 0
        for i in range(1, len(crit_cps)):
            if abs(value - values[i]) < abs(value - values[closest_index]):
                closest_index = i

        return vars[closest_index]

    @staticmethod
    def add_normalization_constraints(
        m: gp.Model,
        cps: CharPts
    ) -> None:
        left_cps = [crit_cps[0][0] for crit_cps in cps]
        [m.addConstr(cp == 0) for cp in left_cps]  # map eagerly

        right_cps = [crit_cps[-1][0] for crit_cps in cps]
        m.addConstr(sum(right_cps) == 1)  # pyright: ignore[reportCallIssue, reportArgumentType]

    def add_monotonicity_constraints(
        self,
        m: gp.Model,
        cps: CharPts
    ) -> None:
        for criterion_cps in cps:
            for (higher, _), (lower, _) in zip(
                criterion_cps[1:], criterion_cps[:-1]
            ):
                m.addConstr(higher - lower >= 0)

    @staticmethod
    def print_table(m: gp.Model) -> None:
        m.update()
        for c in m.getConstrs():
            lhs = m.getRow(c)
            sense = c.Sense
            rhs = c.RHS
            name = c.ConstrName
            print(f"{name}: {lhs} {sense} {rhs}")

    @staticmethod
    def print_values(m: gp.Model) -> None:
        m.update()
        for v in m.getVars():
            print(f"{v.VarName}, {v.X:.4f}")

    def print_char_points_corresponding_values(self, cps: CharPts) -> None:
        for crit, crit_cps in zip(self.criteria.index, cps):
            row = [(var.VarName, float(val.round(2))) for var, val in crit_cps]
            print(f"{crit}: {row}")
