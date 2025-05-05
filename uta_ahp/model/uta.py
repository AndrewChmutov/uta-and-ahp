from __future__ import annotations

from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

from uta_ahp.model.base import Model
from uta_ahp.utils import CriterionType, Domain, Mode, Order

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterator
    from typing import Any

    import gurobipy as gp
    import numpy as np
    import pandas as pd

    type ConstraintChoices = dict[Hashable, gp.Var]
    type Residuals = dict[int, tuple[gp.Var, gp.Var]]
    type CharCritPts = list[tuple[gp.Var, np.float64]]
    type CharPts = list[CharCritPts]


class Uta(Model):
    WEIGHT_UPPER_BOUND = 0.5

    def __init__(
        self,
        df: pd.DataFrame,
        criteria: pd.DataFrame,
        pairwise_comparisons: pd.DataFrame,
        n_characteristic_points: int,
        weight_lower_bound: float,
        optimizer_output: bool,
        show_lp: bool,
        mode: Mode,
        plots: Path,
    ) -> None:
        self.df = df
        self.criteria = criteria
        self.criteria["domain"] = self.criteria["domain"].apply(Domain)
        self.criteria["type"] = self.criteria["type"].apply(CriterionType)
        self.pairwise = pairwise_comparisons
        self.pairwise["value"] = self.pairwise["value"].apply(Order)
        self.n_cp = n_characteristic_points
        self.lb = weight_lower_bound
        self.optimizer_output = optimizer_output
        self.show_lp = show_lp
        self.mode = mode
        self.discriminant: gp.Var | None = None
        self.plots = plots
        self.plots.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def command(
        dataset: Path = Path("data/dataset.csv"),
        criteria: Path = Path("data/criteria.csv"),
        pairwise_comparisons: Path = Path("data/pairwise.csv"),
        n_characteristic_points: int = 3,
        weight_lower_bound: float = 0.00,
        optimizer_output: bool = False,
        show_lp: bool = False,
        mode: Mode = Mode.CLASSIC,
        plots: Path = Path("img"),
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
            weight_lower_bound=weight_lower_bound,
            optimizer_output=optimizer_output,
            show_lp=show_lp,
            mode=mode,
            plots=plots,
        ).solve()

    def solve(self) -> list[pd.DataFrame] | pd.Series | None:
        if self.mode == Mode.CONSISTENT_SUBSETS:
            return self.find_consistent_subsets()
        else:
            return self.solve_with_residuals()

    def find_consistent_subsets(
        self, verbose: bool = True
    ) -> list[pd.DataFrame]:
        import gurobipy as gp

        params = {
            "PoolSearchMode": 2,
            "PoolSolutions": 2 ** self.pairwise.shape[0],
        }
        result = []
        with self._model(params) as m:
            vs = self.add_constraint_choices(m)
            self.set_objective_with_selection(m, vs)
            cps = self.add_characteristic_points(m)
            self.add_preference_constraints(m, cps, vs=vs)
            self.add_normalization_constraints(m, cps)
            self.add_monotonicity_constraints(m, cps)

            # Find all solutions
            for ignored_constraints in range(1, self.pairwise.shape[0] + 1):
                m.addConstr(sum(vs.values()) == ignored_constraints)  # pyright: ignore[reportArgumentType, reportCallIssue]
                m.optimize()
                if m.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
                    continue

                for i in range(m.SolCount):
                    m.setParam("SolutionNumber", i)
                    m.update()
                    result.append([i for i, val in vs.items() if val.X == 0])

        dataframes = [self.pairwise.loc[r] for r in result]
        if verbose:
            [print(df) for df in dataframes]
        return dataframes

    def solve_with_residuals(self) -> pd.Series | None:
        import gurobipy as gp

        if self.mode == Mode.DISCRIMINATE:
            # Find largest consistent subset
            self.mode = Mode.CONSISTENT_SUBSETS
            dfs = self.find_consistent_subsets(verbose=False)
            self.pairwise = min(dfs, key=lambda x: x.shape[0])
            self.mode = Mode.DISCRIMINATE

        with self._model() as m:
            rs = self.add_residuals(m)
            self.set_objective_with_residuals(m, rs)
            cps = self.add_characteristic_points(m)
            self.add_preference_constraints(m, cps, rs=rs)
            self.add_normalization_constraints(m, cps)
            self.add_monotonicity_constraints(m, cps)

            m.optimize()

            if m.ObjVal != 0:
                print(f"Inconsistent, Objective: {m.ObjVal}")
            elif self.show_lp:
                print(f"Objective: {m.ObjVal}")

            if m.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
                print("No feasible solutions")
                return

            if self.show_lp:
                self.print_results(m, cps)
                print()

            m.update()
            self.plot(cps)
            ranking = self.rank(cps).sort_values(ascending=False)
            print("Ranking:")
            print(ranking)
            return ranking

        # Non negativity constraints are covered by the lib by default

    @contextmanager
    def _model(self, params: dict[str, Any] = {}) -> Iterator[gp.Model]:
        """Disables output of `gurobipy` if required.

        Yields:
            Model with changed environment
        """
        import gurobipy as gp

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", int(self.optimizer_output))
            for param in params.items():
                env.setParam(*param)
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

    def add_constraint_choices(self, m: gp.Model) -> ConstraintChoices:
        return {
            i: m.addVar(vtype=Domain.BINARY, name=f"v_{a}_{b}")
            for i, (a, b, _) in self.pairwise.iterrows()
        }

    def set_objective_with_residuals(self, m: gp.Model, rs: Residuals) -> None:
        import gurobipy as gp

        assert self.mode in (Mode.CLASSIC, Mode.DISCRIMINATE)
        errors = sum(chain.from_iterable(rs.values()))
        match self.mode:
            case Mode.CLASSIC:
                # Sum over the flattened iterable
                m.setObjective(expr=errors, sense=gp.GRB.MINIMIZE)
            case Mode.DISCRIMINATE:
                self.discriminant = m.addVar(
                    vtype=gp.GRB.CONTINUOUS, name="discriminant"
                )
                m.setObjective(expr=self.discriminant, sense=gp.GRB.MAXIMIZE)

                # Consistency assumption
                m.addConstr(errors == 0)  # pyright: ignore[reportArgumentType, reportCallIssue]

    def set_objective_with_selection(
        self,
        m: gp.Model,
        vs: ConstraintChoices,
    ) -> None:
        import gurobipy as gp

        assert self.mode == Mode.CONSISTENT_SUBSETS
        errors = sum(vs.values())
        m.setObjective(expr=errors, sense=gp.GRB.MINIMIZE)

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
                        name=f"{crit}_{i}",
                    ),
                    vals[i],
                )
                for i in range(n)
            ])

        return cps

    def add_preference_constraints(
        self,
        m: gp.Model,
        cps: CharPts,
        rs: Residuals | None = None,
        vs: ConstraintChoices | None = None,
    ) -> None:
        def get_handside_expr_(alternative: str) -> gp.LinExpr:
            crit_values = self.df.loc[alternative]
            _ = self.get_closest_characteristic_points
            closest_cps = _(cps, crit_values)
            return sum(closest_cps)  # pyright: ignore[reportReturnType]

        match rs, vs:
            case None, None:
                assert False, "Should pass only one: rs | vs"
            case rs, None:

                def get_handside_expr(alternative: str) -> gp.LinExpr:
                    sig_plus, sig_minus = rs[int(alternative)]
                    U = get_handside_expr_(alternative)
                    return U - sig_plus + sig_minus

            case None, vs:
                get_handside_expr = get_handside_expr_
            case _, _:
                assert False, "Should pass only one: rs | vs"

        for i, row in self.pairwise.iterrows():
            a, b, pref = row.tolist()
            assert isinstance(pref, Order)

            lhs = get_handside_expr(a)
            rhs = get_handside_expr(b)
            match pref, self.mode:
                case Order.PREFERENCE, Mode.CLASSIC:
                    m.addConstr(lhs - rhs >= 0)
                case Order.PREFERENCE, Mode.CONSISTENT_SUBSETS:
                    assert vs
                    m.addConstr(lhs >= rhs - vs[i])
                case Order.PREFERENCE, Mode.DISCRIMINATE:
                    assert self.discriminant
                    m.addConstr(lhs - rhs >= self.discriminant)

                case Order.INDIFFERENCE, Mode.CONSISTENT_SUBSETS:
                    assert vs
                    m.addConstr(lhs >= rhs - vs[i])
                    m.addConstr(rhs >= lhs - vs[i])
                case Order.INDIFFERENCE, _:
                    m.addConstr(lhs == rhs)

                case _:
                    assert False, "Invalid internal state"

    @classmethod
    def get_closest_characteristic_points(
        cls, cps: CharPts, values: list[float]
    ) -> list[gp.Var]:
        return [
            cls.get_closest_characteristic_point(crit_cps, value)
            for crit_cps, value in zip(cps, values)
        ]

    @staticmethod
    def get_closest_characteristic_point(
        crit_cps: CharCritPts, value: float
    ) -> gp.Var:
        vars, values = list(zip(*crit_cps))
        closest_index = 0
        for i in range(1, len(crit_cps)):
            if abs(value - values[i]) < abs(value - values[closest_index]):
                closest_index = i

        return vars[closest_index]

    @staticmethod
    def add_normalization_constraints(m: gp.Model, cps: CharPts) -> None:
        left_cps = [crit_cps[0][0] for crit_cps in cps]
        [m.addConstr(cp == 0) for cp in left_cps]  # map eagerly, fancy

        right_cps = [crit_cps[-1][0] for crit_cps in cps]
        m.addConstr(sum(right_cps) == 1)  # pyright: ignore[reportCallIssue, reportArgumentType]

    def add_monotonicity_constraints(self, m: gp.Model, cps: CharPts) -> None:
        for criterion_cps in cps:
            for (higher, _), (lower, _) in zip(
                criterion_cps[1:], criterion_cps[:-1]
            ):
                m.addConstr(higher - lower >= 0)

    def print_results(self, m: gp.Model, cps: CharPts) -> None:
        self.print_char_points_corresponding_values(cps)
        print()
        self.print_table(m)
        print()
        self.print_values(m)

    @staticmethod
    def print_table(m: gp.Model) -> None:
        print("Table:")
        m.update()
        for c in m.getConstrs():
            lhs = m.getRow(c)
            sense = c.Sense
            rhs = c.RHS
            name = c.ConstrName
            print(f"{name}: {lhs} {sense} {rhs}")

    @staticmethod
    def print_values(m: gp.Model) -> None:
        print("Variables:")
        m.update()
        for v in m.getVars():
            print(f"{v.VarName}, {v.X:.4f}")

    def print_char_points_corresponding_values(self, cps: CharPts) -> None:
        print("Characteristic points:")
        for crit, crit_cps in zip(self.criteria.index, cps):
            row = [(var.VarName, float(val.round(2))) for var, val in crit_cps]
            print(f"{crit}: {row}")

    def plot(self, cps: CharPts) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme()
        plt.rc("font", size=18)

        max_value = max(
            chain.from_iterable(
                [var.X for var, _ in crit_ops] for crit_ops in cps
            )
        )  # noqa: E501
        f, axes = plt.subplots(ncols=len(cps), figsize=(20, 6.7))
        for i, (crit, crit_cps) in enumerate(zip(self.criteria.index, cps)):
            xs = [val for _, val in crit_cps]
            ys = [var.X for var, _ in crit_cps]
            axes[i].plot(xs, ys)
            axes[i].set_title(crit)
            axes[i].set_ylim((0 - 0.01, max_value + 0.01))

        f.suptitle("Marginal value functions")
        f.supxlabel("Feature values")
        f.supylabel("Utility values")

        name = f"uta-{self.mode}"
        f.savefig(self.plots / f"{name}.png")

    def rank(self, cps: CharPts) -> pd.Series:
        cols = self.df.columns
        funcs = self.interpolate(cps)
        return sum(self.df[col].apply(func) for col, func in zip(cols, funcs))  # pyright: ignore[reportReturnType]

    def interpolate(self, cps: CharPts) -> Iterator[Callable[[float], float]]:
        for (_, crit), crit_cps in zip(self.criteria.iterrows(), cps):
            step = {
                CriterionType.COST: -1,
                CriterionType.GAIN: 1,
            }[crit.loc["type"]]

            x_values = [val for _, val in crit_cps][::step]
            u_values = [var.X for var, _ in crit_cps][::step]

            def func(
                x: float,
                x_values: list[float] = x_values,  # pyright: ignore[reportArgumentType]
                u_values: list[float] = u_values,
            ) -> float:
                for i, x_char in enumerate(x_values[1:], start=1):
                    if x_char >= x:
                        break
                else:
                    raise RuntimeError

                u_delta = u_values[i] - u_values[i - 1]
                x_delta = x_values[i] - x_values[i - 1]
                prop = (x - x_values[i - i]) / x_delta
                result = u_values[i - 1] + u_delta * prop
                result = max(result, min(u_values))
                result = min(result, max(u_values))
                return result

            yield func  # pyright: ignore[reportReturnType]
