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


@pytest.fixture
def test_system_stuart_landau() -> DynamicalSystem:
    system = DynamicalSystem("stuart_landau")
    alpha, mu, omega = system.get_parameters()
    standard_params = SymbolicSubstitution({alpha: 0, mu: 1, omega: 1})
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


def test_get_limit_cycle(test_system_stuart_landau):
    def event(t, state, args):
        return state[0]

    event.direction = -1

    _, _, extras = test_system_stuart_landau.get_limit_cycle(
        {}, event, np.array([0, 1]), isostable_expansion_order=1
    )

    tolerance = 1e-7

    expected_circular_frequency = 1
    assert (
        np.abs(extras["circular_frequency"] - expected_circular_frequency) < tolerance
    )

    expected_floquet_exponent = -2
    assert (
        np.abs(extras["jacobian_trace_integral"] - expected_floquet_exponent)
        < tolerance
    )
    assert (
        np.abs(
            extras["floquet_exponent_by_monodromy_matrix"] - expected_floquet_exponent
        )
        < tolerance
    )
