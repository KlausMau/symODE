from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import sympy as sy

from symode.dynamical_system import DynamicalSystem, SymbolicSubstitution

variable = sy.symbols("x")
parameter = sy.symbols("p")


@pytest.fixture
def numerics_adapter() -> Mock:
    adapter = Mock()
    adapter.create_initial_value_problem_solver.return_value = Mock()
    return adapter


@pytest.fixture
def test_system(numerics_adapter) -> DynamicalSystem:
    system = DynamicalSystem(
        SymbolicSubstitution({variable: parameter * variable}),
        numerics_adapter=numerics_adapter,
    )
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


def test_get_symmetry_equations_uses_full_matrix_ansatz(numerics_adapter):
    x, y = sy.symbols("x y")
    system = DynamicalSystem(
        SymbolicSubstitution({x: x, y: 2 * y}),
        numerics_adapter=numerics_adapter,
    )

    entries = sy.symbols("m0:4")
    ansatz = sy.Matrix(2, 2, entries)
    equations = system.get_commutator_to_linear_transformation(ansatz)

    assert equations == {x: sy.Symbol("m1") * y, y: -sy.Symbol("m2") * x}


def test_get_symmetry_equations_respects_matrix_ansatz(numerics_adapter):
    x, y = sy.symbols("x y")
    a, b = sy.symbols("a b")
    system = DynamicalSystem(
        SymbolicSubstitution({x: x**2, y: y}),
        numerics_adapter=numerics_adapter,
    )
    ansatz = sy.Matrix([[a, 0], [0, b]])

    equations = system.get_commutator_to_linear_transformation(ansatz)

    assert equations == {x: -(a**2) * x**2 + a * x**2, y: 0}


def test_get_all_symmetries_builds_indexed_ansatz_for_any_dimension(numerics_adapter):
    x, y, z = sy.symbols("x y z")
    system = DynamicalSystem(
        SymbolicSubstitution({x: x, y: 2 * y, z: 3 * z}),
        numerics_adapter=numerics_adapter,
    )

    symmetries = system.get_all_symmetries()

    assert len(symmetries) == 1
    assert symmetries[0] == sy.diag(
        sy.Symbol("m_0_0"), sy.Symbol("m_1_1"), sy.Symbol("m_2_2")
    )


def test_get_all_symmtries_van_der_pol_returns_three_symmetries():
    system = DynamicalSystem("van_der_pol")

    symmetries = system.get_all_symmetries()
    expected_symmetries = [sy.zeros(2), sy.eye(2), -sy.eye(2)]

    assert len(symmetries) == 3
    assert all(
        any(symmetry == expected for expected in expected_symmetries)
        for symmetry in symmetries
    )


def test_get_trajectories_delegates_to_numerical_solver():
    numerical_solver = Mock()
    initial_value_problem_solver = Mock()
    expected_solution = object()
    numerical_solver.create_initial_value_problem_solver.return_value = (
        initial_value_problem_solver
    )
    initial_value_problem_solver.solve.return_value = expected_solution
    system = DynamicalSystem(
        SymbolicSubstitution({variable: parameter * variable}),
        numerics_adapter=numerical_solver,
    )

    solution = system.get_trajectories(
        (0.0, 1.0),
        np.array([1.0]),
        {parameter: 2.0},
        max_step=0.5,
        rtol=1e-8,
    )

    assert solution is expected_solution
    numerical_solver.create_initial_value_problem_solver.assert_called_once_with(
        {variable: parameter * variable},
        [variable],
        [parameter],
    )
    initial_value_problem_solver.solve.assert_called_once_with(
        (0.0, 1.0),
        np.array([1.0]),
        {parameter: 2.0},
        max_step=0.5,
        rtol=1e-8,
    )


def test_get_limit_cycle(numerics_adapter):
    period = 2 * np.pi
    samples = 1000
    sampled_period = np.linspace(0, period, samples)
    first_fundamental_matrix_solution = np.zeros((4, samples))
    first_fundamental_matrix_solution[2] = 1.0
    second_fundamental_matrix_solution = np.zeros((4, samples))
    second_fundamental_matrix_solution[3] = 1.0

    transient_solution = SimpleNamespace(
        y_events=[np.array([[1.0, 0.0]])],
        t_events=[np.array([0.0, period])],
    )
    limit_cycle_solution = SimpleNamespace(
        y=np.vstack((np.cos(sampled_period), np.sin(sampled_period)))
    )
    first_fundamental_matrix_solution = SimpleNamespace(
        success=True,
        t=sampled_period,
        y=first_fundamental_matrix_solution,
        message="",
    )
    second_fundamental_matrix_solution = SimpleNamespace(
        success=True,
        t=sampled_period,
        y=second_fundamental_matrix_solution,
        message="",
    )
    initial_value_problem_solver = (
        numerics_adapter.create_initial_value_problem_solver.return_value
    )
    initial_value_problem_solver.solve.side_effect = [
        transient_solution,
        limit_cycle_solution,
        first_fundamental_matrix_solution,
        second_fundamental_matrix_solution,
    ]
    numerics_adapter.integrate_trapezoid.return_value = 0.0
    system = DynamicalSystem(
        SymbolicSubstitution(
            {
                sy.symbols("x"): sy.symbols("y"),
                sy.symbols("y"): -sy.symbols("x"),
            }
        ),
        numerics_adapter=numerics_adapter,
    )

    def event(t, state, args):
        return state[0]

    event.direction = -1

    _, _, extras = system.get_limit_cycle(
        {},
        event,
        np.array([0, 1]),
        isostable_expansion_order=1,
        samples=samples,
    )

    tolerance = 1e-7

    assert extras["circular_frequency"] == pytest.approx(1, abs=tolerance)

    assert extras["jacobian_trace_integral"] == pytest.approx(0, abs=tolerance)

    assert extras["floquet_exponents"] == pytest.approx([0.0, 0.0], abs=tolerance)


def test_get_limit_cycle_raises_when_fundamental_matrix_integration_fails(
    numerics_adapter,
):
    period = 2 * np.pi
    samples = 1000
    sampled_period = np.linspace(0, period, samples)
    transient_solution = SimpleNamespace(
        y_events=[np.array([[1.0, 0.0]])],
        t_events=[np.array([0.0, period])],
    )
    limit_cycle_solution = SimpleNamespace(
        y=np.vstack((np.cos(sampled_period), np.sin(sampled_period)))
    )
    failed_solution = SimpleNamespace(
        success=False,
        t=np.array([]),
        y=np.empty((4, 0)),
        message="integration failed",
    )
    initial_value_problem_solver = (
        numerics_adapter.create_initial_value_problem_solver.return_value
    )
    initial_value_problem_solver.solve.side_effect = [
        transient_solution,
        limit_cycle_solution,
        failed_solution,
    ]
    system = DynamicalSystem(
        SymbolicSubstitution(
            {
                sy.symbols("x"): sy.symbols("y"),
                sy.symbols("y"): -sy.symbols("x"),
            }
        ),
        numerics_adapter=numerics_adapter,
    )

    def event(t, state, args):
        return state[0]

    event.direction = -1

    with pytest.raises(RuntimeError, match="fundamental matrix"):
        system.get_limit_cycle(
            {},
            event,
            np.array([0, 1]),
            isostable_expansion_order=1,
            samples=samples,
        )
