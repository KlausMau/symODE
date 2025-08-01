"""
This module provides a class to conviently deal with dynamical systems given as ordinary
differential equations.
"""

import copy
import itertools
from typing import NewType, Literal

import numpy as np
import sympy as sy
import numba as nb

from numpy.typing import NDArray
from scipy.integrate import solve_ivp, trapezoid, cumulative_trapezoid
from sympy.utilities import lambdify
from symode import systems_catalogue

NumericSubstitution = NewType("NumericSubstitution", dict[sy.Symbol, float])
SymbolicSubstitution = NewType("SymbolicSubstitution", dict[sy.Symbol, sy.Expr])


def get_dynamical_equations_from_catalogue(name: str, **params) -> SymbolicSubstitution:
    """returns a dynamical equations dictionary if "name" is found in the catalogue"""
    if hasattr(systems_catalogue, name) is False:
        print(f"{name} is not in the catalogue.")
        return SymbolicSubstitution({})

    dynamical_equations_builder = getattr(systems_catalogue, name)
    return dynamical_equations_builder(**params)


def get_symbols_with_index(
    symbols: list[sy.Symbol], index: int
) -> dict[sy.Symbol, sy.Symbol]:
    """returns a mapping between symbolds and their indexed counterpart"""
    return {symbol: sy.symbols(f"{symbol}_{index}") for symbol in symbols}


class DynamicalSystem:
    """
    A class to deal with dynamical systems of the form

    dx/dt = F(x,t)      with state variable x € R^N

    in numerical (SciPy) and analytical (SymPy) aspects.

    Naming conventions:
    "get_X" returns an output and DO NOT change the object
    "set_X" can return an output, DO change the object
    """

    def __init__(
        self, dynamical_equations: str | SymbolicSubstitution, **params
    ) -> None:
        """
        dynamical equations are given as "str" or "dict"
        str:  looked up in the catalogue
        dict: taken as is

        The keys of the dictionary are taken as the dynamical variables.
        All other symbols that appear in the values of the dictionary are interpreted as parameters.
        """
        if isinstance(dynamical_equations, str):
            dynamical_equations = get_dynamical_equations_from_catalogue(
                dynamical_equations, **params
            )

        self._dynamical_equations = dynamical_equations
        self._set_attributes_from_dynamical_equations()

    def __str__(self) -> str:
        return self.get_dynamical_equations_in_latex()

    def _set_attributes_from_dynamical_equations(self) -> None:
        """sets the attributes of the dynamical system based on the dynamical equations"""
        self._variables = list(self._dynamical_equations.keys())
        self._variables.sort(key=str)

        self._dimension = len(self._variables)

        all_symbols = sy.Matrix(list(self._dynamical_equations.values())).free_symbols
        self._parameters = list(set(all_symbols) - set(self._variables))
        self._parameters.sort(key=str)

        self._jacobian = self._calculate_jacobian()
        self._hessian = self._calculate_hessian()
        self._f_odeint = self._compile_integrator()

    def get_dynamical_equations_in_latex(self) -> str:
        """returns the LaTeX string of the dynamical equations"""
        equations = [
            rf"\dot {sy.latex(var)} = {sy.latex(self._dynamical_equations[var])} \\"
            for var in self._variables
        ]
        return "".join(equations)

    def get_variables(self) -> list[sy.Symbol]:
        """returns variables of the system"""
        return self._variables

    def get_parameters(self) -> list[sy.Symbol]:
        """returns parameters of the system"""
        return self._parameters

    # symbolic features

    def get_fixed_points(self, jacobian=False):
        """
        this function returns the fixed points of the system

        CAUTION: it does not check whether the system is autonomous or not. Maybe set t=0?

        if Jacobian = True: the returned dictionary is equipped with another key Jacobian
        """
        try:
            fixed_points = sy.solve(
                self._dynamical_equations, self._variables, dict=True
            )
        except NotImplementedError:
            fixed_points = sy.nsolve(
                self._dynamical_equations,
                self._variables,
                np.zeros(self._dimension),
                dict=True,
            )

        if jacobian is True:
            for i, fp in enumerate(fixed_points):
                fixed_points[i].update({"Jacobian": self._jacobian.doit().subs(fp)})

        return fixed_points

    def set_parameter_value(self, parameter_values: SymbolicSubstitution) -> None:
        """modifies the system by replacing parameters"""
        for var in self._variables:
            new_dynamical_equation = self._dynamical_equations[var].subs(
                parameter_values
            )
            self._dynamical_equations.update({var: new_dynamical_equation})

        self._set_attributes_from_dynamical_equations()

    def add_term(self, target_variables: list[sy.Symbol], term: sy.Expr) -> None:
        """modifies the system by adding terms"""

        for target_var in target_variables:
            old_equation = self._dynamical_equations[target_var]
            self._dynamical_equations.update({target_var: sy.Add(old_equation, term)})

        self._set_attributes_from_dynamical_equations()

    def get_time_derivative_of_observable(self, observable: sy.Expr) -> sy.Expr:
        """returns the time derivative of an observable"""
        time_derivative = sy.Integer(0)
        for var in self._variables:
            time_derivative = sy.Add(
                time_derivative,
                sy.Mul(
                    sy.Derivative(observable, var).doit(),
                    self._dynamical_equations[var],
                ),
            )
        return time_derivative

    def _calculate_jacobian(self) -> sy.Matrix:
        """compute Jacobian matrix of system"""
        jacobian = sy.ones(self._dimension)
        for i, j in itertools.product(range(self._dimension), range(self._dimension)):
            var_i = self._variables[i]
            var_j = self._variables[j]
            jacobian[i, j] = sy.Derivative(self._dynamical_equations[var_i], var_j)
        return jacobian

    def _calculate_hessian(self) -> list[sy.Matrix]:
        """compute Hessian tensor of system"""
        hessian = []
        for variable in self._variables:
            temp = sy.ones(self._dimension)
            for i, j in itertools.product(
                range(self._dimension), range(self._dimension)
            ):
                var_i = self._variables[i]
                var_j = self._variables[j]
                temp[i, j] = sy.Derivative(
                    sy.Derivative(self._dynamical_equations[variable], var_j), var_i
                )
            hessian.append(temp)
        return hessian

    def _calculate_jacobian_at_trajectory(
        self, trajectory: NDArray, parameter_values: NumericSubstitution
    ) -> NDArray:
        get_numeric_jacobian = lambdify(
            tuple(self._variables),
            self._jacobian.doit().subs(parameter_values),
            cse=True,
        )

        time_samples = len(trajectory[0, :])
        jacobian_at_limit_cycle = np.zeros(
            (self._dimension, self._dimension, time_samples)
        )

        for t in range(time_samples):
            jacobian_at_limit_cycle[:, :, t] = get_numeric_jacobian(*trajectory[:, t])

        return jacobian_at_limit_cycle

    def _calculate_fundamental_matrix(
        self,
        time_instances: NDArray,
        state0: NDArray,
        parameter_values: NumericSubstitution,
        **kwargs,
    ) -> NDArray:
        system_o1 = self.get_new_system_with_perturbation_variables(order=1)

        fundamental_matrix = np.zeros(
            (self._dimension, self._dimension, len(time_instances))
        )
        fundamental_matrix[:, :, 0] = np.eye(self._dimension)

        for n in range(self._dimension):
            # design the initial state
            state0_o1 = np.zeros(2 * self._dimension)
            state0_o1[: self._dimension] = state0
            state0_o1[self._dimension + n] = 1.0

            solution_o1 = system_o1.get_trajectories(
                (0, time_instances[-1]),
                state0_o1,
                parameter_values,
                t_eval=time_instances,
                **kwargs,
            )

            fundamental_matrix[:, n, :] = solution_o1.y[self._dimension :, :]

        return fundamental_matrix

    def _calculate_isostable_expansion_o2(
        self,
        time_instances: NDArray,
        state0_on_limit_cycle: NDArray,
        fundamental_matrix: NDArray,
        floquet_exponent: float,
        floquet_eigenvector: NDArray,
        parameter_values: NumericSubstitution,
    ) -> NDArray:
        system_o2 = self.get_new_system_with_perturbation_variables(order=2)

        state0_o2 = np.zeros(3 * self._dimension)
        state0_o2[: self._dimension] = state0_on_limit_cycle
        state0_o2[self._dimension : 2 * self._dimension] = floquet_eigenvector

        solution_o2 = system_o2.get_trajectories(
            (0, time_instances[-1]),
            state0_o2,
            parameter_values,
            t_eval=time_instances,
        )

        special_o2_solution = solution_o2.y[2 * self._dimension :, :]

        isostable_expansion_o2_at_state0 = np.matmul(
            np.linalg.inv(
                np.exp(2.0 * floquet_exponent * time_instances[-1]) * np.eye(2)
                - fundamental_matrix[:, :, -1]
            ),
            special_o2_solution[:, -1],
        )
        isostable_expansion_o2 = np.array(
            [
                np.exp(-2.0 * floquet_exponent * time_instance)
                * (
                    np.matmul(
                        fundamental_matrix[:, :, t], isostable_expansion_o2_at_state0[:]
                    )
                    + special_o2_solution[:, t]
                )
                for t, time_instance in enumerate(time_instances)
            ]
        ).transpose()

        return isostable_expansion_o2

    # return a new DynamicalSystem object

    def get_new_system_with_fixed_parameters(
        self, parameter_values: SymbolicSubstitution
    ):
        """returns a new dynamical system with fixed parameter values"""
        new_system = DynamicalSystem(self._dynamical_equations)
        new_system.set_parameter_value(parameter_values)
        return new_system

    def get_new_system_with_inverted_time(self):
        """returns a new dynamical system with inverted time"""
        new_dynamical_equations = {}
        for var in self._variables:
            new_dynamical_equations.update({var: -self._dynamical_equations[var]})
        return DynamicalSystem(new_dynamical_equations)

    def get_new_system_after_transformation(
        self, new_variables, equations: SymbolicSubstitution, **kwargs
    ):
        """
        expects list:
        new_variables = [x_new, y_new]
        equations = {x_old: f(x_new, y_new), y_old: g(x_new, y_new)}
        """
        # calculate the Jacobian
        jacobian_list = []
        for x_old in self._variables:
            row = [sy.Derivative(equations[x_old], x_new) for x_new in new_variables]
            jacobian_list.append(row)
        jacobian = sy.Matrix(jacobian_list)

        # invert Jacobian
        # try:, except
        jacobian_inv = jacobian.inv().doit()

        substituted_dynamical_equations = {}
        for var in self._variables:
            substituted_dynamical_equations.update(
                {var: self._dynamical_equations[var].subs(equations)}
            )

        ode_new = jacobian_inv * substituted_dynamical_equations

        # put into dictionary
        new_dynamical_equations = {}
        for i, x_new in enumerate(new_variables):
            new_dynamical_equations[x_new] = ode_new[i].cancel()

        return DynamicalSystem(SymbolicSubstitution(new_dynamical_equations), **kwargs)

    def get_new_system_with_perturbation_variables(self, order: int = 1):
        """construct perturbed dynamical system up to order N
        (0) dot x = F(x)
        (1) dot d_1 = J_F(x)*d_1
        (2) dot d_2 = J_F(x)*d_2 + H(x, d_1)
        """

        new_dynamical_equations = SymbolicSubstitution({})

        # terms of order "0"
        if order < 0:
            return DynamicalSystem(new_dynamical_equations)

        new_dynamical_equations = self._dynamical_equations

        if order < 1:
            return DynamicalSystem(new_dynamical_equations)

        # add terms of order "1"
        self._calculate_jacobian()
        perturbation_vars_order_1 = [sy.symbols(f"d_1{var}") for var in self._variables]
        for i in range(self._dimension):
            perturbation_equation = sum(
                self._jacobian.doit()[i, j] * perturbation_vars_order_1[j]
                for j in range(self._dimension)
            )
            new_dynamical_equations.update(
                {perturbation_vars_order_1[i]: perturbation_equation}
            )

        if order < 2:
            return DynamicalSystem(new_dynamical_equations)

        # add terms of order "2"
        self._calculate_hessian()
        perturbation_vars_order_2 = [sy.symbols(f"d_2{var}") for var in self._variables]
        for i in range(self._dimension):
            jacobian_matrix_product = sum(
                self._jacobian.doit()[i, j] * perturbation_vars_order_2[j]
                for j in range(self._dimension)
            )
            offset = (
                sy.transpose(perturbation_vars_order_1)
                * self._hessian[i].doit()
                * perturbation_vars_order_1
            )
            new_dynamical_equations.update(
                {perturbation_vars_order_2[i]: jacobian_matrix_product + offset[0]}
            )

        return DynamicalSystem(new_dynamical_equations)

    def get_new_system_with_coupling(
        self,
        coupling_function: SymbolicSubstitution,
        coupling_matrix: sy.Matrix = sy.ones(3, 3),
        non_identical_parameters=None,
    ):
        """
        x_i = F_i(x_i) - sum_{j=1}^{N} C_ij G(x_j)
        coupling_matrix:   square matrix with dimension N
        coupling function: string directing to a DynamicalSystem object
        (Sympy expressions with variables of unit system)
        """
        if non_identical_parameters is None:
            non_identical_parameters = []

        number_of_units = len(np.array(coupling_matrix)[0])

        # write ODEs for new indexed variables
        new_dynamical_equation = {}
        for i in range(number_of_units):
            indexed_symbols = get_symbols_with_index(self._variables, i + 1)
            indexed_parameters = get_symbols_with_index(non_identical_parameters, i + 1)

            for var in self._variables:
                # substitute variables with index
                new_equation = sy.Expr(
                    self._dynamical_equations[var].subs(indexed_symbols)
                )

                # add coupling terms
                for j in range(number_of_units):
                    new_equation = sy.Add(
                        new_equation,
                        -sy.Mul(coupling_matrix[i, j], coupling_function[var]),
                    )

                # substitute nonidentical parameters with index
                new_equation = sy.Expr(new_equation.subs(indexed_parameters))

                # write into dictionary for new indexed variable
                new_dynamical_equation[indexed_symbols[var]] = new_equation

        return DynamicalSystem(SymbolicSubstitution(new_dynamical_equation))

    # numerical features

    def _compile_integrator(self):
        """
        compiles the integrator and returns a function with the
        signature to fit into "scipy.integrate.solve_ivp"

        Still to add and verify:
        - add external "stimulation" as time-dependent function
        - vectorize functions
        - precompile "f_ODEINT" with Numba
        """

        # get numba-precompiled functions (maximum number of arguments is 255 ...)
        f_auto = nb.jit(
            lambdify(
                tuple(self._variables + self._parameters),
                tuple(self._dynamical_equations.values()),
                cse=True,
            ),
            nopython=True,
        )

        def f_odeint(_, state, parameters):
            # combine "state" and "parameters" to new "arguments" list variable
            arguments = list(state) + list(parameters)
            return f_auto(*arguments)

        return f_odeint

    def get_trajectories(
        self, t_span, state0, parameter_values, max_step=0.01, **kwargs
    ):
        """
        Return the trajectories of the dynamical system using "scipy.integrate.solve_ivp".

        Comment on "max_step":
        The integration seems to give spurious and non-reliable results,
        if the choice of "max_step" is left to "solve_ivp"

        """

        # ordered parameter values
        parameter_values_list = [parameter_values[p] for p in self._parameters]

        # integrate with SciPy
        states = solve_ivp(
            self._f_odeint,
            t_span,
            state0,
            args=(parameter_values_list,),
            max_step=max_step,
            **kwargs,
        )

        return states

    def get_event_based_evolution(
        self,
        state0,
        parameter_values,
        event,
        event_settings,
        t_start=0,
        n_events=10,
        time_max=100,
        **kwargs,
    ):
        """
        returns the trajectories for an evolution whose parameters change every time an event is
        found

        N_events:       maximum number of event findings
        event_settings: list of tuples of duration "T_event" and parameters "parameter_values_event"
        t_max:          maximum time
        """
        # make a copy of "parameter_values"
        params = copy.deepcopy(parameter_values)

        # store times of events by index
        events_idx = []

        # initialize "states"
        states = np.reshape(np.array(state0), (len(state0), 1))

        # initialize "Time"
        time = np.array([t_start])

        # intialize event counter
        n = 0

        while n <= n_events and time[-1] < t_start + time_max:
            # integrate until event
            sol = self.get_trajectories(
                (time[-1], t_start + time_max),
                states[:, -1],
                parameter_values=params,
                events=event,
                **kwargs,
            )

            # store results
            states = np.concatenate((states, sol.y), axis=1)
            time = np.concatenate((time, sol.t))

            # print('event found at t=' + str(Time[-1]))
            n += 1

            # store event index
            events_idx.append(len(time))

            # iterate through the triggered sequence
            for t_event, parameters_values_event, state_action in event_settings:
                # update parameters
                params.update(parameters_values_event)

                # change state
                if state_action is None:

                    def state_action(state):
                        return state

                # integrate with parameter values changed by the event
                sol = self.get_trajectories(
                    (time[-1], time[-1] + t_event),
                    state_action(states[:, -1]),
                    parameter_values=params,
                    **kwargs,
                )

                # store results
                states = np.concatenate((states, sol.y), axis=1)
                time = np.concatenate((time, sol.t))

            # reset parameters
            params.update(parameter_values)

        # drop the last event
        # because it might be computed by reaching "t_max", not by finding the event

        return states[:, : events_idx[-1]], time[: events_idx[-1]], events_idx[:-1]

    def get_limit_cycle(
        self,
        parameter_values: NumericSubstitution,
        event,
        state0: NDArray,
        t_eq: float = 100,
        samples: int = 1000,
        isostable_expansion_order: Literal[0, 1, 2] = 0,
        **kwargs,
    ) -> tuple[NDArray, NDArray, dict]:
        """
        currently only for 2D systems
        returns:
        Time:       instances of time, Time[-1] is the period
        y:          limit cycle expansion;
                    y[0] is the limit cycle
                    y[1], y[2]
        extras:     additional information
        """
        event.terminal = False
        extras = {}

        transient_solution = self.get_trajectories(
            (0.0, t_eq), state0, parameter_values, t_eval=[t_eq], events=event, **kwargs
        )
        if transient_solution.y_events is None or transient_solution.t_events is None:
            raise NotImplementedError(
                "No stable limit cycle found for given system and parameter values. If you are confident that a stable limit cycle should exist, check the event function and the initial state.")

        state0_on_limit_cycle = transient_solution.y_events[0][-1]

        periods_between_events = [
            transient_solution.t_events[0][i] - transient_solution.t_events[0][i - 1]
            for i in range(1, len(transient_solution.t_events[0]))
        ]
        for period_between_events in periods_between_events:
            print(period_between_events)

        period = periods_between_events[-1]
        extras.update({"period": period})
        print(f"period = {period}")

        circular_frequency = 2 * np.pi / period
        extras.update({"circular_frequency": circular_frequency})
        print(f"frequency = {circular_frequency}")

        sampled_period = np.linspace(0, period, samples)

        limit_cycle_solution = self.get_trajectories(
            (0.0, period),
            state0_on_limit_cycle,
            parameter_values,
            t_eval=sampled_period,
            **kwargs,
        )

        # prepare return array
        isostable_expansion = np.zeros((isostable_expansion_order + 1, 2, samples))
        isostable_expansion[0] = limit_cycle_solution.y

        if isostable_expansion_order == 0 or self._dimension != 2:
            return sampled_period, isostable_expansion, extras

        jacobian_at_limit_cycle = self._calculate_jacobian_at_trajectory(
            isostable_expansion[0], parameter_values
        )
        extras.update({"jacobian": jacobian_at_limit_cycle})
        jacobian_trace_integral = (
            trapezoid(
                np.trace(jacobian_at_limit_cycle, axis1=0, axis2=1), sampled_period
            )
            / period
        )
        extras.update({"jacobian_trace_integral": jacobian_trace_integral})
        print(f"integral of Jacobian trace = {jacobian_trace_integral}")

        fundamental_matrix = self._calculate_fundamental_matrix(
            sampled_period, state0_on_limit_cycle, parameter_values
        )
        extras.update({"fundamental_matrix": fundamental_matrix})

        # this is numerical unstable for large |kappa|
        # eigenvalues/-vectors of monodromy matrix (this is for N=2 only!!)
        # this selection process has to be revisited!
        eigenvalues, eigenvectors = np.linalg.eig(fundamental_matrix[:, :, -1])
        floquet_eigenvectors = eigenvectors.transpose()
        floquet_exponents = np.log(np.abs(eigenvalues)) / period

        extras.update({"floquet_exponents": floquet_exponents})
        print(f"Floquet exponents = {floquet_exponents}")

        isostable_expansion[1] = np.array(
            [
                np.exp(-jacobian_trace_integral * sampled_period[t])
                * np.matmul(fundamental_matrix[:, :, t], floquet_eigenvectors)
                for t in range(len(sampled_period))
            ]
        ).transpose()
        # y[1] = np.array([np.power(np.min(w),-Time[t]/Time[-1])*np.matmul(fund_matrix[:,:,t],
        # non_unity_eigenvec) for t in range(len(Time))]).transpose()

        if isostable_expansion_order == 1:
            return sampled_period, isostable_expansion, extras

        isostable_expansion[2] = self._calculate_isostable_expansion_o2(
            sampled_period,
            state0_on_limit_cycle,
            fundamental_matrix,
            jacobian_trace_integral,
            floquet_eigenvectors,
            parameter_values,
        )

        return sampled_period, isostable_expansion, extras

    def get_isochrones_isostables(
        self, params, event, r=1e-7, t_max=None, kwargs_limit_cycle=None, **kwargs_int
    ):
        """infer phase-isostable structure for 2D system by backward integration"""

        if t_max is None:
            t_max = [20, 20]

        if kwargs_limit_cycle is None:
            kwargs_limit_cycle = {}

        # obtain limit cycle
        time, y, _ = self.get_limit_cycle(
            params,
            event,
            isostable_expansion_order=1,
            **kwargs_limit_cycle,
            # t_eq=250, state0=[1.,1.],
            # samples=samples,
            **kwargs_int,
        )

        # time-inverted system
        system_inv = self.get_new_system_with_inverted_time()

        # initial curve
        samples = len(time) - 1

        t = time[-1]
        time_samples = [np.arange(0, t_max, t / samples) for t_max in t_max]

        states = {}

        for z, sign in enumerate([-1, 1]):

            states.update({sign: np.zeros((samples, len(time_samples[z]), 2))})

            x0 = y[0, 0] + sign * r * y[1, 0]
            y0 = y[0, 1] + sign * r * y[1, 1]

            for s in range(samples):

                # integrate backwards in time
                sol = system_inv.get_trajectories(
                    t_span=(0.0, time_samples[z][-1]),
                    t_eval=time_samples[z],
                    state0=np.array([x0[s], y0[s]]),
                    parameter_values=params,
                    **kwargs_int,
                )

                # bring solution into correct shape to account for exploding integration

                temp = np.empty((len(time_samples[z]), 2))
                temp.fill(np.inf)

                if len(sol.y) > 0:
                    y_t = sol.y.T
                    temp[: len(y_t[:, 0]), : len(y_t[0, :])] = y_t

                states[sign][s, :, :] = temp

            # rolling the states times further in time to form isochrones

            for t, _ in enumerate(time_samples[z][:]):
                for i in [0, 1]:
                    states[sign][:, t, i] = np.roll(states[sign][:, t, i], -t)

        return states

    def get_isostable_around_focus(self, params):
        """returns a function to compute an ellipse around a focus fixed point

        Parameters
        ----------

        params : dict
            The parameter values for the system

        Returns
        -------
        fp : list
            The fixed point coordinates

        ellipse_radius: func
            A function that calculates the ellipses radius

        """

        if self._dimension != 2:
            print("This works only for 2-dimensional systems!")
            return None

        # compute fixed points
        fp = self.get_fixed_points()

        # x0 = [self.VARIABLES[i].subs(fp) for i in range(self.Dim)

        # compute Jacobian at fixed point "fp[0]"
        df = self._jacobian.doit().subs(fp[0]).subs(params)
        print(df)
        df_np = np.array(df).astype(np.float64)

        # compute eigenvalues and left eigenvector numerically
        eigenvalues, eigenvectors = np.linalg.eig(np.transpose(df_np))

        if np.imag(eigenvalues[0]) == 0:
            print("This function works only for fixed points of focus type!")
            return None

        p = eigenvectors[0, 0]
        q = eigenvectors[1, 0]

        def ellipse_radius(x):
            term_1 = np.abs(p) ** 2 * np.cos(x) ** 2
            term_2 = np.abs(q) ** 2 * np.sin(x) ** 2
            term_3 = 2 * np.real(p * np.conj(q)) * np.cos(x) * np.sin(x)
            return 1.0 / np.sqrt(term_1 + term_2 + term_3)

        return fp[0], ellipse_radius

    def get_isostable_map_at_fixed_points(self, params) -> list:
        """returns a function to compute the coordinates for given isostables

        Parameters
        ----------

        params : dict
            The parameter values for the system

        Returns
        -------
        output : list
            A list of functions that map from isostable to original coordinates
        """

        if self._dimension != 2:
            print("This function is still only implemented for 2-dimensional systems.")
            return []

        # new DynamicalSystem with fixed parameters
        system = self.get_new_system_with_fixed_parameters(params)

        # compute fixed points
        fixed_points = system.get_fixed_points()

        output = []

        for fp in fixed_points:
            # Jacobian at fixed point
            df = system._jacobian.doit().subs(fp)

            df_np = np.array(df).astype(np.float64)
            fp_np = np.array(list(fp.values())).astype(np.float64)

            # compute eigenvalues and right eigenvectors numerically
            eigenvalues, eigenvectors_right = np.linalg.eig(df_np)

            c1 = eigenvectors_right[:, 0]
            c2 = eigenvectors_right[:, 1]

            # compute eigenvalues and left eigenvectors numerically
            # here, we use the orthogonality of left/right eigenvectors and 2D
            d1_raw = np.array([c2[1], -c2[0]])
            d2_raw = np.array([c1[1], -c1[0]])

            d1 = d1_raw / np.dot(c1, d1_raw)
            d2 = d2_raw / np.dot(c2, d2_raw)

            # construct map
            def isostable_from_x(x, y):
                return np.matmul(np.array([d1, d2]), (np.array([x, y]) - fp_np))

            def x_from_isostable_by_a1_a2(a1, a2):
                """complex-conjugate eigenvalues

                Parameters
                ----------
                a1 : float
                    isostable coordinate a1

                a2 : float
                    isostable coordinate a2
                """

                return fp_np + c1 * a1 + c2 * a2

            def x_from_isostable_by_r_psi(r, psi):
                """complex-conjugate eigenvalues

                Parameters
                ----------
                r : float
                    absolute value (non-negative)

                psi : float
                    complex argument (e.g. from 0 to 2*pi)
                """
                # np.matmul(eigenvectors, a)
                return fp_np + 2 * r * np.real(c1 * np.exp(1j * psi))

            if np.imag(eigenvalues[0]) == 0:
                output.append(
                    [eigenvalues, x_from_isostable_by_a1_a2, isostable_from_x]
                )
            else:
                output.append(
                    [eigenvalues, x_from_isostable_by_r_psi, isostable_from_x]
                )

        return output

    def get_time_averaged_jacobian(
        self, t_span, state0, parameter_values: NumericSubstitution, **kwargs
    ):
        """returns the time-averaged Jacobian matrix of the system"""

        solution = self.get_trajectories(t_span, state0, parameter_values, **kwargs)
        jacobian_at_trajectory = self._calculate_jacobian_at_trajectory(
            solution.y, parameter_values
        )
        total_time = solution.t[-1] - solution.t[0]

        time_averaged_jacobian = np.zeros(
            (self._dimension, self._dimension, len(solution.t))
        )

        for i, j in itertools.product(range(self._dimension), range(self._dimension)):
            time_averaged_jacobian[i, j] = (
                cumulative_trapezoid(
                    jacobian_at_trajectory[i, j], solution.t, initial=0
                )
                / total_time
            )

        return time_averaged_jacobian
