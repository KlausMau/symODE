import pytest
import sympy as sy
import numpy as np
from symode.dynamical_system import DynamicalSystem, SymbolicSubstitution

variable = sy.symbols("x")
parameter = sy.symbols("p")


@pytest.fixture
def test_system() -> DynamicalSystem:
    system = DynamicalSystem(SymbolicSubstitution({variable: parameter * variable}))
    return system


def system_stuart_landau() -> DynamicalSystem:
    system = DynamicalSystem("stuart_landau")
    alpha, mu, omega = system.get_parameters()
    standard_params = SymbolicSubstitution(
        {alpha: sy.Integer(0), mu: sy.Rational(1 / 2), omega: sy.Integer(1)}
    )
    system.set_parameter_value(standard_params)
    return system


def system_harmonic_oscillator() -> DynamicalSystem:
    system = DynamicalSystem("harmonic_oscillator")
    gamma, omega = system.get_parameters()
    standard_params = SymbolicSubstitution({gamma: sy.Integer(0), omega: sy.Integer(1)})
    system.set_parameter_value(standard_params)
    return system


def test_init(test_system):
    assert test_system.get_parameters() == [parameter]
    assert test_system.get_variables() == [variable]
    assert test_system._dynamical_equations == {variable: parameter * variable}


def test_add_term(test_system):
    additional_parameter = sy.symbols("q")
    test_system.add_term([variable], additional_parameter)

    assert test_system.get_parameters() == [parameter, additional_parameter]
    assert test_system.get_variables() == [variable]
    assert test_system._dynamical_equations == {
        variable: parameter * variable + additional_parameter
    }


def test_set_parameter_value(test_system):
    test_system.set_parameter_value({parameter: 1})

    assert test_system.get_parameters() == []
    assert test_system.get_variables() == [variable]
    assert test_system._dynamical_equations == {variable: variable}


@pytest.mark.parametrize(
    "system, expected_circular_frequency, expected_floquet_exponent",
    [
        (system_harmonic_oscillator(), 1, 0),
        (system_stuart_landau(), 1, -0.2),
    ],
)
def test_get_limit_cycle(
    system, expected_circular_frequency, expected_floquet_exponent
):
    def event(t, state, args):
        return state[0]

    event.direction = -1

    _, _, extras = system.get_limit_cycle(
        {},
        event,
        np.array([0, 1]),
        isostable_expansion_order=1,
    )

    tolerance = 1e-7

    assert extras["circular_frequency"] == pytest.approx(
        expected_circular_frequency, abs=tolerance
    )

    assert extras["jacobian_trace_integral"] == pytest.approx(
        expected_floquet_exponent, abs=tolerance
    )

    assert extras["floquet_exponents"] == pytest.approx(
        [0.0, expected_floquet_exponent], abs=tolerance
    )
