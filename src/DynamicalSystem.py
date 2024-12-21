import copy
import itertools
import numpy as np
import scipy as sc
import sympy as sy
import numba as nb
from IPython.core.display import display, Math

from scipy import integrate
import src.SystemsCatalogue as SystemsCatalogue

rng = np.random.default_rng(12345)

def get_dynamical_equations_from_catalogue(name: str, **params) -> dict:
    '''returns a dynamical equations dictionary if "name" is found in the catalogue'''
    if hasattr(SystemsCatalogue, name) is False:
        print(f'{name} is not in the catalogue.')
        return {}

    dynamical_equations_builder = getattr(SystemsCatalogue, name)
    return dynamical_equations_builder(**params)

class DynamicalSystem:
    '''
    A class to deal with dynamical systems of the form

    dx/dt = F(x,t)      with state variable x € R^N

    in numerical (SciPy) and analytical (SymPy) aspects.

    Naming conventions:
    "get_X" returns an output and DO NOT change the object
    "set_X" can return an output, DO change the object
    '''

    def __init__(self, dynamical_equations: str | dict, **params) -> None:
        '''
        dynamical equations are given as "str" or "dict"
        str:  looked up in the catalogue
        dict: taken as is

        The keys of the dictionary are taken as the dynamical variables.
        All other symbols that appear in the values of the dictionary are interpreted as parameters.
        '''
        if isinstance(dynamical_equations, str):
            dynamical_equations = get_dynamical_equations_from_catalogue(dynamical_equations,
                                                                         **params)

        self._dynamical_equations = dynamical_equations
        self.set_attributes_from_dynamical_equations(compile_integrator=True)


    def set_attributes_from_dynamical_equations(self, compile_integrator=True) -> None:
        '''sets the attributes of the dynamical system based on the dynamical equations'''
        self._variables = self._dynamical_equations.keys()
        self._dimension = len(self._variables)

        all_symbols = sy.Matrix(list(self._dynamical_equations.values())).free_symbols
        self._parameters = list(set(all_symbols)-set(self._variables))

        self._jacobian = sy.Matrix()
        self._hessian : list[sy.Matrix] = []

        if compile_integrator is True:
            self.compile_integrator()

    # convenience features

    def show(self) -> None:
        '''prints the LaTeX code'''
        display(Math(rf'{self.get_dynamical_equations_in_latex()}'))

    def get_dynamical_equations_in_latex(self) -> str:
        '''returns the LaTeX string of the dynamical equations'''
        ode_latex = '$'
        for var in self._variables:
            ode_latex += rf'\dot {sy.latex(var)} = {sy.latex(self._dynamical_equations[var])} \\\\'
        ode_latex += '$'
        return ode_latex

    def get_indexing_dict(self, index):
        '''
        returns a dictionary mapping the variables to an indexed version of themselves
        '''
        return dict(zip(self._variables, [sy.symbols(str(var) + '_' + str(index))
                                          for var in self._variables]))

    # symbolic features

    def get_fixed_points(self, jacobian=False):
        '''
        this function returns the fixed points of the system

        CAUTION: it does not check whether the system is autonomous or not. Maybe set t=0?

        if Jacobian = True: the returned dictionary is equipped with another key Jacobian
        '''
        try:
            fixed_points = sy.solve(self._dynamical_equations, self._variables, dict=True)
        except NotImplementedError:
            fixed_points = sy.nsolve(self._dynamical_equations, self._variables, np.zeros(self._dimension),
                                     dict=True)

        if jacobian is True:
            self.calculate_jacobian()
            for i, fp in enumerate(fixed_points):
                fixed_points[i].update({'Jacobian' : self._jacobian.doit().subs(fp)})

        return fixed_points

    def set_parameter_value(self, parameter_values: dict) -> None:
        '''fixes given parameter values in the system'''
        for var in self._variables:
            new_dynamical_equation = self._dynamical_equations[var].subs(parameter_values)
            self._dynamical_equations.update({var: new_dynamical_equation})

        self.set_attributes_from_dynamical_equations(compile_integrator=True)

    def add_term(self, target_variables, term) -> None:
        '''adds a term to the dynamical system'''

        for x in target_variables:
            # find index of target variables
            indeces = [i for i, variable in enumerate(self._variables) if variable == x]
            idx = indeces[0]
            # add to ODE
            self._dynamical_equations[idx] += term

        # add new parameters
        new_parameters = list(set(term.free_symbols)-set(self._variables)-set(self._parameters))
        self._parameters += new_parameters

        # recompile integrator
        self.compile_integrator()

    def calculate_jacobian(self) -> None:
        '''compute Jacobian of system'''
        self._jacobian = sy.ones(self._dimension)
        for i, j in itertools.product(range(self._dimension), range(self._dimension)):
            var_i = self._variables[i]
            var_j = self._variables[j]
            self._jacobian[i,j] = sy.Derivative(self._dynamical_equations[var_i], var_j)

    def calculate_hessian(self) -> None:
        '''compute Hessian tensor of system'''
        n = self._dimension
        self._hessian = []
        for v in range(len(self._variables)):
            temp = sy.ones(n)
            for i in range(n):
                for j in range(n):
                    temp[i,j] = sy.Derivative(sy.Derivative(self._dynamical_equations[v], self._variables[j]),
                                              self._variables[i])
            self._hessian.append(temp)
        return

    # return a new DynamicalSystem object

    def get_new_system_with_fixed_parameters(self, parameter_values: dict):
        ''' returns a new dynamical system with fixed parameter values'''
        new_system = DynamicalSystem(self._dynamical_equations)
        new_system.set_parameter_value(parameter_values)
        return new_system

    def get_new_system_with_inverted_time(self):
        ''' returns a new dynamical system with inverted time'''
        new_dynamical_equations = {}
        for var in self._variables:
            new_dynamical_equations.update({var: -self._dynamical_equations[var]})
        return DynamicalSystem(new_dynamical_equations)

    def new_transformed(self, new_variables, equations, **kwargs):
        '''
        expects list:
        new_variables = [x_new, y_new]
        equations = {x_old: f(x_new, y_new), y_old: g(x_new, y_new)}
        '''
        # calculate the Jacobian
        jacobian_list = []
        for x_old in self._variables:
            row = [sy.Derivative(equations[x_old], x_new) for x_new in new_variables]
            jacobian_list.append(row)
        jacobian = sy.Matrix(jacobian_list)

        # invert Jacobian
        #try:, except
        jacobian_inv = jacobian.inv().doit()
        ode_new = jacobian_inv*self._dynamical_equations.subs(equations)

        # put into dictionary
        ode_dict = {}
        for i, x_new in enumerate(new_variables):
            ode_dict[x_new] = ode_new[i].cancel()

        return DynamicalSystem(ode_dict, **kwargs)

    def new_perturbed(self, order=1):
        ''' construct perturbed dynamical system up to order N
        (0) dot x = F(x)
        (1) dot d_1 = J_F(x)*d_1
        (2) dot d_2 = J_F(x)*d_2 + H(x, d_1)
        '''
        n = self._dimension

        # terms of order "0"
        ode_pert = {}
        for i in range(n):
            ode_pert.update({self._variables[i]: self._dynamical_equations[i]})

        # terms of order "1"
        self.calculate_jacobian()
        d1 = sy.Matrix([sy.symbols(f'd_1{x}') for x in self._variables])
        j_d1 = self._jacobian.doit()*d1

        for i in range(n):
            ode_pert.update({d1[i]: j_d1[i]})

        # terms of order "2"
        if order == 2:
            self.calculate_hessian()
            d2 = sy.Matrix([sy.symbols(f'd_2{x}') for x in self._variables])
            j_d2 = self._jacobian.doit()*d2
            for i in range(n):
                c = sy.transpose(d1)*self._hessian[i].doit()*d1
                ode_pert.update({d2[i]: j_d2[i] + c[0]})

        return DynamicalSystem(dynamical_equations = ode_pert)

    def new_coupled(self, coupling_matrix = sy.ones(3,3),
                    coupling_function = 'linear',
                    non_identical_parameters=[]):
        '''
        coupling_matrix:   square matrix with dimension N
        coupling function: string directing to a DynamicalSystem object
        (Sympy expressions with variables of unit system)
        '''

        # number of units is given by the coupling matrix
        n = len(np.array(coupling_matrix)[0])

        # create DynamicalSystem for copuling function
        coupling_function_system = DynamicalSystem(dynamical_equations=coupling_function,
                                                   variables = self._variables)

        # write ODEs for new indexed variables
        ode_new = {}
        for i in range(n):
            for var_index in range(self._dimension):
                # copy autonomous ODE and substitute variables with index
                ode_new_temp = self._dynamical_equations[var_index].subs(self.get_indexing_dict(i+1))

                # substitute nonidentical parameters with index
                for parameter in non_identical_parameters:
                    parameter_i = sy.symbols(str(parameter) + '_' + str(i+1))
                    ode_new_temp = ode_new_temp.subs({parameter: parameter_i})

                # add coupling terms
                for j in range(n):
                    term = coupling_function_system._dynamical_equations[var_index].subs(self.get_indexing_dict(j+1))
                    ode_new_temp -= coupling_matrix[i,j]*term

                # write into dictionary for new indexed variable
                ode_new[sy.symbols(str(self._variables[var_index]) + '_' + str(i+1))] = ode_new_temp

        return DynamicalSystem(dynamical_equations=ode_new)

    # numerical features

    def get_precompiled_integrator(self):
        '''
        compiles the integrator and returns a function with the
        signature to fit into "scipy.integrate.solve_ivp"

        Still to add and verify:
        - add external "stimulation" as time-dependent function
        - vectorize functions
        - precompile "f_ODEINT" with Numba
        '''

        # get numba-precompiled functions (maximum number of arguments is 255 ...)
        f_auto = nb.jit(sy.utilities.lambdify(tuple(self._variables + self._parameters),
                                              tuple(self._dynamical_equations), cse=True),
                                              nopython=True)

        def f_odeint(t, state, parameters):
            # combine "state" and "parameters" to new "arguments" list variable
            arguments = list(state) + list(parameters)
            return f_auto(*arguments)

        return f_odeint

    def compile_integrator(self, **kwargs) -> None:
        '''compiles the integrator before it is used'''
        self._f_odeint = self.get_precompiled_integrator(**kwargs)

    def get_trajectories(self, t_span, state0 = None, parameter_values={}, max_step=0.01, **kwargs):
        '''
        Return the trajectories of the dynamical system using "scipy.integrate.solve_ivp".

        Comment on "max_step":
        The integration seems to give spurious and non-reliable results,
        if the choice of "max_step" is left to "solve_ivp"

        '''
        # if no state is given, choose randomly
        if state0 is None:
            state0 = rng.standard_normal(size=self._dimension)

        # create a properly ordered list from the dictionary "parameter_values"
        parameter_list = [parameter_values[p] for p in self._parameters]

        # integrate
        states = sc.integrate.solve_ivp(self._f_odeint, t_span, state0,
                                        args=(parameter_list, ), max_step=max_step, **kwargs)

        return states

    def get_event_based_evolution(self, state0, parameter_values, event, event_settings,
                                  t_start=0, n_events=10, time_max=100, **kwargs):
        '''
        returns the trajectories for an evolution whose parameters change every time an event is
        found

        N_events:       maximum number of event findings
        event_settings: list of tuples of duration "T_event" and parameters "parameter_values_event"
        t_max:          maximum time
        '''
        # make a copy of "parameter_values"
        params = copy.deepcopy(parameter_values)

        # store times of events by index
        events_idx = []

        # initialize "states"
        states = np.reshape(np.array(state0), (len(state0),1))

        # initialize "Time"
        time = np.array([t_start])

        # intialize event counter
        n = 0

        while n <= n_events and time[-1] < t_start + time_max:
            # integrate until event
            sol = self.get_trajectories((time[-1], t_start + time_max),
                                        states[:,-1],
                                        parameter_values=params,
                                        events=event, **kwargs)

            # store results
            states = np.concatenate((states, sol.y), axis=1)
            time = np.concatenate((time, sol.t))

            #print('event found at t=' + str(Time[-1]))
            n += 1

            # store event index
            events_idx.append(len(time))

            # iterate through the triggered sequence
            for t_event, parameters_values_event, state_action in event_settings:
                # update parameters
                params.update(parameters_values_event)

                # change state
                if state_action is None:
                    state_action = lambda state: state

                # integrate with parameter values changed by the event
                sol = self.get_trajectories((time[-1], time[-1] + t_event),
                                            state_action(states[:,-1]),
                                            parameter_values=params, **kwargs)

                # store results
                states = np.concatenate((states, sol.y), axis=1)
                time = np.concatenate((time, sol.t))

            # reset parameters
            params.update(parameter_values)

        # drop the last event
        # because it might be computed by reaching "t_max", not by finding the event

        return states[:,:events_idx[-1]], time[:events_idx[-1]], events_idx[:-1]

    def get_limit_cycle(self, params, event,
                        state0=None, t_eq=100, samples=1000, isostable_expansion_order=0,
                        show_results = True,
                        **kwargs):
        '''
        returns:
        Time:       instances of time, Time[-1] is the period
        y:          limit cycle expansion;
                    y[0] is the limit cycle
                    y[1], y[2]
        extra[0]:   Jacobian
        extra[1]:   fundamental matrix
        extra[2]:   kappa_trace
        extra[3]:   kappa_monod
        extra[4]:   d2_special
        '''
        event.terminal = False

        # get to equilibrium
        sol_eq = self.get_trajectories(t_span=(0., t_eq),
                                       t_eval = [t_eq],
                                       state0=state0, events=event,
                                       parameter_values=params,
                                       **kwargs)

        # integrate from last event one period
        #for i in range(1, len(sol_eq.t_events[0])):
        #    print(sol_eq.t_events[0][i]-sol_eq.t_events[0][i-1])

        # period
        t = sol_eq.t_events[0][-1]-sol_eq.t_events[0][-2]

        # frequency
        omega = 2*np.pi/t

        time = np.linspace(0, t, samples)

        sol_lc  = self.get_trajectories(t_span=(0., t),
                                        t_eval = time,
                                        state0 = sol_eq.y_events[0][-1],
                                        parameter_values=params,
                                        **kwargs)

        # prepare return array
        y = np.zeros((isostable_expansion_order+1, 2, samples))

        # grab limit-cycle orbit
        y[0] = sol_lc.y

        extras = []
        if isostable_expansion_order>=1:

            ### calculate Jacobian at limit cycle ###

            self.calculate_jacobian()
            j_np = sy.utilities.lambdify(tuple(self._variables), self._jacobian.doit().subs(params),
                                         cse=True)

            j = np.zeros((self._dimension, self._dimension, samples))

            for t in range(samples):
                j[:,:,t] = j_np(*y[0,:,t])

            ### EXTRA 0 ### --> "extra" to dictionary
            extras.append(j)

            ### calculate fundamental solution matrix ###

            system_o1 = self.new_perturbed(order=1)

            fund_matrix = np.zeros((self._dimension, self._dimension, len(time)))
            fund_matrix[:,:,0] = np.eye(self._dimension)

            for n in range(self._dimension):
                # design the initial state
                state0 = np.zeros(2*self._dimension)
                state0[:self._dimension] = y[0,:,0]
                state0[self._dimension + n] = 1.

                sol = system_o1.get_trajectories(t_span=(0,time[-1]),
                                                t_eval=time,
                                                state0=state0,
                                                parameter_values=params,
                                                **kwargs)

                fund_matrix[:,n,:] = sol.y[self._dimension:,:]

            ### EXTRA 1 ###
            extras.append(fund_matrix)

            # eigenvalues/-vectors of monodromy matrix (this is for N=2 only!!)
            # this selection process has to be revisited!
            eigenvals, eigenvecs = np.linalg.eig(fund_matrix[:,:,-1])
            non_unity_eigenvec = eigenvecs.transpose()[np.abs(eigenvals-1) > 1e-4][0]

            # this is numerical unstable for large |kappa|, consider changing to trace formula
            kappa_trace = integrate.trapezoid(np.trace(j, axis1=0, axis2=1), time)/t
            kappa_monod = np.log(np.min(eigenvals))/t

            ### EXTRA 2 & 3 ###
            extras.append(kappa_trace)
            extras.append(kappa_monod)

            y[1] = np.array([np.exp(-kappa_trace*time[t])*np.matmul(fund_matrix[:,:,t], non_unity_eigenvec) for t in range(len(time))]).transpose()
            #y[1] = np.array([np.power(np.min(w),-Time[t]/Time[-1])*np.matmul(fund_matrix[:,:,t],
            # non_unity_eigenvec) for t in range(len(Time))]).transpose()

            if isostable_expansion_order>=2:

                ### calculate special solution for d2 ###

                system_o2 = self.new_perturbed(order=2)

                state0 = np.zeros(3*self._dimension)
                state0[:self._dimension] = y[0,:,0]
                state0[self._dimension: 2*self._dimension] = non_unity_eigenvec

                sol = system_o2.get_trajectories(t_span=(0,time[-1]),
                                                t_eval=time,
                                                state0=state0,
                                                parameter_values=params)

                d2_special = sol.y[2*self._dimension:,:]

                y2_data_ini = np.matmul(np.linalg.inv(np.exp(2.*kappa_trace*time[-1])*np.eye(2)-fund_matrix[:,:,-1]), d2_special[:,-1])
                y[2] = np.array([np.exp(-2.*kappa_trace*time[t])*(np.matmul(fund_matrix[:,:,t], y2_data_ini[:]) + d2_special[:,t]) for t in range(len(time))]).transpose()

                ### EXTRA 4 ###
                extras.append(d2_special)

        if show_results is True:
            print(f'period = {t}')
            print(f'frequency = {omega}')
            print(f'Floquet exponent (calculated by Jacobian trace)   = {kappa_trace}')
            print(f'Floquet exponent (calculated by Monodromy matrix) = {kappa_monod}')

        return time, y, extras

    def get_isochrones_isostables(self, params, event, r = 1e-7,
                                  t_max = [20,20], kwargs_limit_cycle={}, **kwargs_int):
        '''infer phase-isostable structure for 2D system by backward integration'''

        # obtain limit cycle
        time, y, extras = self.get_limit_cycle(params, event, isostable_expansion_order=1,
                                         **kwargs_limit_cycle,
                                         #t_eq=250, state0=[1.,1.],
                                         #samples=samples,
                                         **kwargs_int)

        # time-inverted system
        system_inv = self.get_new_system_with_inverted_time()

        # initial curve
        samples = len(time)-1

        t = time[-1]
        time_samples = [np.arange(0, t_max, t/samples) for t_max in t_max]

        states = dict()

        for z, sign in enumerate([-1,1]):

            states.update({sign: np.zeros((samples, len(time_samples[z]), 2))})

            x0 = y[0,0] + sign*r*y[1,0]
            y0 = y[0,1] + sign*r*y[1,1]

            for s in range(samples):

                # integrate backwards in time
                sol = system_inv.get_trajectories(t_span=(0., time_samples[z][-1]),
                                            t_eval = time_samples[z],
                                            state0 = np.array([x0[s], y0[s]]),
                                            parameter_values=params,
                                            **kwargs_int)

                # bring solution into correct shape to account for exploding integration

                temp = np.empty((len(time_samples[z]),2))
                temp.fill(np.inf)

                if len(sol.y) > 0:
                    y_t = sol.y.T
                    temp[:len(y_t[:,0]), :len(y_t[0,:])] = y_t

                states[sign][s,:,:] = temp

            # rolling the states times further in time to form isochrones

            for t,t in enumerate(time_samples[z][:]):
                for i in [0,1]:
                    states[sign][:,t,i] = np.roll(states[sign][:,t,i],-t)

        return states

    def get_isostable_around_focus(self, params):
        ''' returns a function to compute an ellipse around a focus fixed point

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

        '''

        if self._dimension != 2:
            print('This works only for 2-dimensional systems!')
            return

        # compute fixed points
        fp = self.get_fixed_points()

        #x0 = [self.VARIABLES[i].subs(fp) for i in range(self.Dim)

        # compute Jacobian at fixed point "fp[0]"
        self.calculate_jacobian()
        df = self._jacobian.doit().subs(fp[0]).subs(params)
        print(df)
        df_np = np.array(df).astype(np.float64)

        # compute eigenvalues and left eigenvector numerically
        eigenvalues, eigenvectors = np.linalg.eig(np.transpose(df_np))

        if np.imag(eigenvalues[0]) == 0:
            print('This function works only for fixed points of focus type!')
            return

        p = eigenvectors[0,0]
        q = eigenvectors[1,0]

        def ellipse_radius(x):
            term_1 = np.abs(p)**2*np.cos(x)**2
            term_2 = np.abs(q)**2*np.sin(x)**2
            term_3 = 2*np.real(p*np.conj(q))*np.cos(x)*np.sin(x)
            return 1./np.sqrt(term_1 + term_2 + term_3)

        return fp[0], ellipse_radius

    def get_isostable_map_at_fixed_points(self, params) -> list:
        ''' returns a function to compute the coordinates for given isostables

        Parameters
        ----------

        params : dict
            The parameter values for the system

        Returns
        -------
        output : list
            A list of functions that map from isostable to original coordinates
        '''

        if self._dimension != 2:
            print('This function is still only implemented for 2-dimensional systems.')
            return list()

        # new DynamicalSystem with fixed parameters
        system = self.get_new_system_with_fixed_parameters(params)

        # compute Jacobian
        system.calculate_jacobian()

        # compute fixed points
        fixed_points = system.get_fixed_points()

        output = list()

        for fp in fixed_points:
            # Jacobian at fixed point
            df = system._jacobian.doit().subs(fp)

            df_np = np.array(df).astype(np.float64)
            fp_np = np.array(list(fp.values())).astype(np.float64)

            # compute eigenvalues and right eigenvectors numerically
            eigenvalues, eigenvectors_right = np.linalg.eig(df_np)

            c1 = eigenvectors_right[:,0]
            c2 = eigenvectors_right[:,1]

            # compute eigenvalues and left eigenvectors numerically
            # here, we use the orthogonality of left/right eigenvectors and 2D
            d1_raw = np.array([c2[1], -c2[0]])
            d2_raw = np.array([c1[1], -c1[0]])

            d1 = d1_raw/np.dot(c1,d1_raw)
            d2 = d2_raw/np.dot(c2,d2_raw)

            # construct map
            def isostable_from_x(x, y):
                return np.matmul(np.array([d1, d2]),(np.array([x,y]) - fp_np))

            if np.imag(eigenvalues[0]) == 0:
                def x_from_isostable(a1, a2):
                    ''' complex-conjugate eigenvalues

                    Parameters
                    ----------
                    a1 : float
                        isostable coordinate a1

                    a2 : float
                        isostable coordinate a2
                    '''

                    return fp_np + c1*a1 + c2*a2

            else:
                def x_from_isostable(r, psi):
                    ''' complex-conjugate eigenvalues

                    Parameters
                    ----------
                    r : float
                        absolute value (non-negative)

                    psi : float
                        complex argument (e.g. from 0 to 2*pi)
                    '''
                    #np.matmul(eigenvectors, a)
                    return fp_np + 2*r*np.real(c1*np.exp(1j*psi))

            output.append([eigenvalues, x_from_isostable, isostable_from_x])

        return output

    def get_time_averaged_jacobian(self, params, **kwargs):
        #'''this function calculates the angular frequency and the Floquet exponent
        # of a 2D limit cycle system'''

        self.calculate_jacobian()
        j_np = sy.utilities.lambdify(tuple(self._variables), self._jacobian.doit().subs(params),
                                     cse=True)

        sol = self.get_trajectories(parameter_values=params, **kwargs)

        j_int = np.zeros((self._dimension, self._dimension, len(sol.t)))

        for i in range(self._dimension):
            for j in range(self._dimension):
                j_ij_at_lc = [j_np(sol.y[0,t], sol.y[1,t])[i,j] for t in range(len(sol.t))]
                j_int[i,j] = integrate.cumulative_trapezoid(j_ij_at_lc, sol.t,
                                                            initial=0)/(sol.t[-1]-sol.t[0])

        return j_int
