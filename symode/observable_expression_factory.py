import sympy as sy

from symode.dynamical_system import DynamicalSystem


def get_remainder_with_rational_ansatz(
    system: DynamicalSystem, numerator: sy.Expr, denominator: sy.Expr, ld: sy.Symbol
) -> tuple[sy.Expr, sy.Expr]:
    """returns the observable and the adjoint equation for a rational ansatz"""
    dt_numerator = system.get_time_derivative_of_observable(numerator)
    dt_denominator = system.get_time_derivative_of_observable(denominator)

    equation = (
        dt_numerator * denominator
        - dt_denominator * numerator
        - ld * denominator * numerator
    )
    observable = numerator / denominator

    return observable, equation


def get_remainder_with_exponential_ansatz(
    system: DynamicalSystem, factor: sy.Expr, exponent: sy.Expr, ld: sy.Symbol
) -> tuple[sy.Expr, sy.Expr]:
    """returns the observable and the adjoint equation for an exponential ansatz"""
    dt_factor = system.get_time_derivative_of_observable(factor)
    dt_exponent = system.get_time_derivative_of_observable(exponent)

    equation = dt_factor - dt_exponent * factor - ld * factor
    observable = factor * sy.exp(exponent)

    return observable, equation


def get_remainder_with_complex_ansatz(
    system: DynamicalSystem,
    complex_polynomial: sy.Expr,
    real_polynomial: sy.Expr,
    ld: sy.Symbol,
    beta: sy.Symbol,
) -> tuple[sy.Expr, sy.Expr]:
    """returns the observable and the adjoint equation for a complex polynomial ansatz"""
    dt_complex_polynomial = system.get_time_derivative_of_observable(complex_polynomial)
    dt_real_polynomial = system.get_time_derivative_of_observable(real_polynomial)

    equation = (
        dt_complex_polynomial * real_polynomial
        + beta * dt_real_polynomial * complex_polynomial
        - ld * complex_polynomial * real_polynomial
    )
    observable = complex_polynomial * sy.exp(beta * sy.log(real_polynomial))

    return observable, equation
