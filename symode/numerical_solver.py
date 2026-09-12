from collections.abc import Callable

import numba as nb
import sympy as sy
from scipy.integrate import cumulative_trapezoid, solve_ivp, trapezoid
from sympy.utilities.lambdify import lambdify


class NumericalSolver:
    def __init__(
        self,
        dynamical_equations: dict[sy.Symbol, sy.Expr],
        variables: list[sy.Symbol],
        parameters: list[sy.Symbol],
    ) -> None:
        self._variables = variables
        self._parameters = parameters
        self._f_odeint = self._compile_integrator(dynamical_equations)

    def _compile_integrator(
        self, dynamical_equations: dict[sy.Symbol, sy.Expr]
    ) -> Callable:
        f_auto = nb.jit(
            lambdify(
                tuple(self._variables + self._parameters),
                tuple(dynamical_equations.values()),
                cse=True,
            ),
            nopython=True,
        )

        def f_odeint(_, state, parameters):
            arguments = list(state) + list(parameters)
            return f_auto(*arguments)

        return f_odeint

    def solve(
        self,
        t_span,
        state0,
        parameter_values,
        max_step=0.01,
        **kwargs,
    ):
        parameter_values_list = [parameter_values[p] for p in self._parameters]
        return solve_ivp(
            self._f_odeint,
            t_span,
            state0,
            args=(parameter_values_list,),
            max_step=max_step,
            **kwargs,
        )

    @staticmethod
    def integrate_trapezoid(values, time):
        return trapezoid(values, time)

    @staticmethod
    def integrate_cumulative_trapezoid(values, time):
        return cumulative_trapezoid(values, time, initial=0)
