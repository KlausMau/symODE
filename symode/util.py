"""This module contains utility functions."""

import sympy as sy
from symode.dynamical_system import DynamicalSystem
from symode.componentwise_expression import ComponentwiseExpression


def update_solution(
    solution: dict[sy.Symbol, sy.Expr], new_solution_part: dict[sy.Symbol, sy.Expr]
) -> dict[sy.Symbol, sy.Expr]:
    """returns a solution updated with new information"""
    for parameter, value in solution.items():
        solution[parameter] = value.subs(new_solution_part).cancel()

    solution.update(new_solution_part)

    return solution


def find_solution_of_equation_by_inserting_values(
    equation: sy.Expr,
    variable: sy.Symbol,
    value_parameter_list: dict[sy.Symbol, sy.Expr],
    show_process: bool = False,
):
    """returns the solution of the equation by inserting the given values"""
    solutions: dict[sy.Symbol, sy.Expr] = {}

    for solvable_parameter, variable_value in value_parameter_list.items():
        sol = sy.solve(equation.subs({variable: variable_value}), solvable_parameter)

        if not sol:
            print(f"No solution found for {solvable_parameter}")
            return solutions

        solved_parameter_expression = sol[0]

        if show_process is True:
            print(f"{solvable_parameter}={solved_parameter_expression}")

        new_solution_part = {solvable_parameter: solved_parameter_expression.simplify()}

        # update the solutions
        solutions = update_solution(solutions, new_solution_part)

        # update equation
        equation = equation.subs(new_solution_part)

    # check whether the solution is complete (complete = prints "0")
    if show_process is True:
        print("remaining terms of equation:")
        print(equation.simplify())

    return solutions


def get_remainder_with_rational_ansatz(
    system: DynamicalSystem, numerator: sy.Expr, denominator: sy.Expr, ld: sy.Symbol
) -> tuple[sy.Expr, ComponentwiseExpression]:
    dt_numerator = system.get_time_derivative_of_observable(numerator)
    dt_denominator = system.get_time_derivative_of_observable(denominator)

    equation = (
        dt_numerator * denominator
        - dt_denominator * numerator
        - ld * denominator * numerator
    )
    observable = numerator / denominator

    return observable, ComponentwiseExpression(equation)


def get_remainder_with_exponential_ansatz(
    system: DynamicalSystem, factor: sy.Expr, exponent: sy.Expr, ld: sy.Symbol
) -> tuple[sy.Expr, ComponentwiseExpression]:
    dt_factor = system.get_time_derivative_of_observable(factor)
    dt_exponent = system.get_time_derivative_of_observable(exponent)

    equation = dt_factor - dt_exponent * factor - ld * factor
    observable = factor * sy.exp(exponent)

    return observable, ComponentwiseExpression(equation)


def get_remainder_with_complex_ansatz(
    system: DynamicalSystem,
    complex_polynomial: sy.Expr,
    real_polynomial: sy.Expr,
    ld: sy.Symbol,
    beta: sy.Symbol,
) -> tuple[sy.Expr, ComponentwiseExpression]:
    dt_complex_polynomial = system.get_time_derivative_of_observable(complex_polynomial)
    dt_real_polynomial = system.get_time_derivative_of_observable(real_polynomial)

    equation = (
        dt_complex_polynomial * real_polynomial
        + beta * dt_real_polynomial * complex_polynomial
        - ld * complex_polynomial * real_polynomial
    )
    observable = complex_polynomial * sy.exp(beta * sy.log(real_polynomial))

    return observable, ComponentwiseExpression(equation)
