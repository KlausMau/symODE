#from os import system
import numpy as np
import scipy as sc
import sympy as sy
import numba as nb
#from numbalsoda import lsoda_sig, lsoda, dop853

#import sys
import copy
from IPython.core.display import display, Math

from scipy import integrate #, interpolate, optimize
#from matplotlib import cm
import SystemsCatalogue
import CircularRealFunction as cf

rng = np.random.default_rng(12345)

# Conventions:
# "get_X" returns an output and DO NOT change the object 
# "set_X" can return an output, DO change the object

class DynamicalSystem:
    '''
    This is a class to deal in numerical and analytical aspect with dynamical systems of the form

    dx/dt = F(x)      with state variable x € R^N

    N-dimensional function F is given as dictionary:
    ----------------------------------------------
    F   autonomous      autonomous dynamics

    Future features:
    - adapt functions to non-autonmous F(x,t)
    - store ODE also as dictionary
    '''

    def __init__(self, autonomous, autocompile_integrator=True, **params) -> None:
        
        ### set autonomous dynamics by "string" or "dictionary"

        if type(autonomous) == str:
            # initialize by a special name    
            # search for function matching the name    
            try:
                #Builder = getattr(DynamicalSystem_SympyBuilder, autonomous)
                Builder = getattr(SystemsCatalogue, autonomous)
            except AttributeError:
                print('Unknown name. Using "van_der_Pol" instead.')
                Builder = getattr(SystemsCatalogue, 'van_der_Pol')

            autonomous = Builder(**params)
             
        # keys -> dynamical variables (list introduces an order!)
        self.VARIABLES = list(autonomous.keys())

        # dimensionality
        self.DIMENSION = len(self.VARIABLES)

        # values -> ODEs (Sympy matrix, fixes the order of variables!)
        self.ODE = sy.Matrix(list(autonomous.values()))

        # every other symbol appearing in the ODEs apart from variables
        self.PARAMETERS = list(set(self.ODE.free_symbols)-set(self.VARIABLES))

        # compile integrator function
        if autocompile_integrator == True:
            self.f_ODEINT = self.get_precompiled_integrator()

    # convenience features

    def show(self) -> None:
        display(Math(r'{}'.format(self.get_ODE_LaTeX())))

    def get_ODE_LaTeX(self):
        ODE_latex = '$'
        for i in range(self.DIMENSION):
            ODE_latex += '\dot {} = {} \\\\ '.format(sy.latex(self.VARIABLES[i]), sy.latex(self.ODE[i]))
        ODE_latex += '$'

        return ODE_latex[:]

    def get_indexing_dict(self, index):
        '''
        returns a dictionary mapping the variables to an indexed version of themselves
        '''
        return dict(zip(self.VARIABLES, [sy.symbols(str(var) + '_' + str(index)) for var in self.VARIABLES]))

    # symbolic features

    def get_fixed_points(self, Jacobian=False):
        '''
        this function returns the fixed points of the system

        CAUTION: it does not check whether the system is autonomous or not. Maybe set t=0?
        
        if Jacobian = True: the returned dictionary is equipped with another key Jacobian
        '''
        try:
            fixed_points = sy.solve(self.ODE, self.VARIABLES, dict=True)
        except NotImplementedError:
            fixed_points = sy.nsolve(self.ODE, self.VARIABLES, np.zeros(self.DIMENSION), dict=True)

        if Jacobian == True:
            self.calculate_Jacobian()
            for i, fp in enumerate(fixed_points):
                fixed_points[i].update({'Jacobian' : self.JACOBIAN.doit().subs(fp)})

        return fixed_points

    def set_parameter_value(self, param_values) -> None:
        # insert values into ODE
        self.ODE = self.ODE.subs(param_values)

        # recheck for the parameters
        self.PARAMETERS = list(set(self.ODE.free_symbols)-set(self.VARIABLES))

        # recompile integrator 
        self.compile_integrator()

    def add_term(self, target_variables, term) -> None:
        # this function is preliminary  

        for x in target_variables:

            # find index of target variables
            indeces = [i for i, variable in enumerate(self.VARIABLES) if variable == x]
            idx = indeces[0]

            # add to ODE
            self.ODE[idx] += term
        
        # add new parameters        
        new_parameters = list(set(term.free_symbols)-set(self.VARIABLES)-set(self.PARAMETERS))
        self.PARAMETERS += new_parameters

        # recompile integrator
        self.compile_integrator()

    def calculate_Jacobian(self) -> None:
        '''compute Jacobian of system'''
        N = self.DIMENSION
        self.JACOBIAN = sy.ones(N)
        for i in range(N):
            for j in range(N):
                self.JACOBIAN[i,j] = sy.Derivative(self.ODE[i], self.VARIABLES[j])
        return

    def calculate_Hessian(self) -> None:
        '''compute Hessian tensor of system'''
        N = self.DIMENSION
        self.HESSIAN = []
        for v in range(len(self.VARIABLES)):
            temp = sy.ones(N)
            for i in range(N):
                for j in range(N):
                    temp[i,j] = sy.Derivative(sy.Derivative(self.ODE[v], self.VARIABLES[j]), self.VARIABLES[i])
            self.HESSIAN.append(temp)
        return

    # return a new DynamicalSystem object

    def new_parameter_set(self, params):
        ''' returns a new DynamicalSystem with fixed parameter values
        
        TODO : this function would benefit from "ODE" being a dictionary
        '''
        
        ODE_new = {}
        for i in range(len(self.VARIABLES)):
            ODE_new.update({self.VARIABLES[i]: -self.ODE[i]})
        
        system_new = DynamicalSystem(autonomous = ODE_new)
        system_new.set_parameter_value(params)
        return system_new

    def new_time_inverted(self):
        ODE_new = {}
        for i in range(len(self.VARIABLES)):
            ODE_new.update({self.VARIABLES[i]: -self.ODE[i]})
        return DynamicalSystem(autonomous = ODE_new)

    def new_transformed(self, new_variables, equations, **kwargs):
        ''' 
        expects list:
        new_variables = [x_new, y_new]
        equations = {x_old: f(x_new, y_new), y_old: g(x_new, y_new)}
        '''
        # calculate the Jacobian
        Jacobian_list = []
        for x_old in self.VARIABLES:
            row = [sy.Derivative(equations[x_old], x_new) for x_new in new_variables]            
            Jacobian_list.append(row)
        Jacobian = sy.Matrix(Jacobian_list)

        # invert Jacobian
        #try:, except
        Jacobian_inv = Jacobian.inv().doit()
        ODE_new = Jacobian_inv*self.ODE.subs(equations)

        # put into dictionary    
        ODE_dict = {}        
        for i, x_new in enumerate(new_variables):
            ODE_dict[x_new] = ODE_new[i].cancel()

        return DynamicalSystem(ODE_dict, **kwargs) 
    
    def new_perturbed(self, order=1):
        ''' construct perturbed dynamical system up to order N 
        (0) dot x = F(x)
        (1) dot d_1 = J_F(x)*d_1
        (2) dot d_2 = J_F(x)*d_2 + H(x, d_1)
        '''
        N = self.DIMENSION

        # terms of order "0"
        ODE_pert = {}
        for i in range(N):
            ODE_pert.update({self.VARIABLES[i]: self.ODE[i]})
        
        # terms of order "1"
        self.calculate_Jacobian()
        d1 = sy.Matrix([sy.symbols(f'd_1{x}') for x in self.VARIABLES])
        J_d1 = self.JACOBIAN.doit()*d1
        
        for i in range(N):
            ODE_pert.update({d1[i]: J_d1[i]})

        # terms of order "2"
        if order == 2:
            self.calculate_Hessian()
            d2 = sy.Matrix([sy.symbols(f'd_2{x}') for x in self.VARIABLES])
            J_d2 = self.JACOBIAN.doit()*d2
            for i in range(N):
                C = sy.transpose(d1)*self.HESSIAN[i].doit()*d1
                ODE_pert.update({d2[i]: J_d2[i] + C[0]})

        return DynamicalSystem(autonomous = ODE_pert)

    def new_coupled(self, coupling_matrix = sy.ones(3,3), coupling_function = 'linear', non_identical_parameters=[]):
        '''
        coupling_matrix:   square matrix with dimension N
        coupling function: string directing to a DynamicalSystem object (Sympy expressions with variables of unit system)

        To-Do:
        if no specifications are given, impose mean field coupling in first variable with coupling parameter "epsilon"
        '''

        # number of units is given by the coupling matrix
        N = len(np.array(coupling_matrix)[0])
        
        # create DynamicalSystem for copuling function
        coupling_function_System = DynamicalSystem(autonomous=coupling_function, variables = self.VARIABLES)

        # write ODEs for new indexed variables
        ODE_new = {}
        for i in range(N):
            for var_index in range(self.DIMENSION):
                # copy autonomous ODE and substitute variables with index
                ODE_new_temp = self.ODE[var_index].subs(self.get_indexing_dict(i+1)) 

                # substitute nonidentical parameters with index
                for parameter in non_identical_parameters:
                    parameter_i = sy.symbols(str(parameter) + '_' + str(i+1))
                    ODE_new_temp = ODE_new_temp.subs({parameter: parameter_i})

                # add coupling terms
                for j in range(N):
                    term = coupling_function_System.ODE[var_index].subs(self.get_indexing_dict(j+1))
                    ODE_new_temp -= coupling_matrix[i,j]*term

                # write into dictionary for new indexed variable
                ODE_new[sy.symbols(str(self.VARIABLES[var_index]) + '_' + str(i+1))] = ODE_new_temp

        return DynamicalSystem(autonomous=ODE_new)

    # numerical features

    def get_precompiled_integrator(self, stimulation = nb.njit(lambda t : 0), clean_sympy_expressions=False):
        '''
        Compile the integrator, ready to be fed into "sc.integrate.solve_ivp"

        Still to add and verify:
        - add external "stimulation" as time-dependent function
        - vectorize functions
        - precompile "f_ODEINT" with Numba
        '''

        # simplify evaluates remaining formal derivatives
        #if clean_sympy_expressions == True:
        #    f_auto_sy = sy.cancel(sy.simplify(f_auto_sy))
        #    f_ext_sy  = sy.cancel(sy.simplify(f_ext_sy))

        # get numba-precompiled functions
        # maximum number of arguments = 255 ... 

        f_auto = nb.jit(sy.utilities.lambdify(tuple(self.VARIABLES + self.PARAMETERS), tuple(self.ODE), cse=True), nopython=True)

        # function with the signature to fit into "scipy.integrate.solve_ivp"
        def f_ODEINT(t, state, parameters):            
            # combine "state" and "parameters" to new "arguments" list variable
            arguments = list(state) + list(parameters)
            #state_I = list(state) + [I_ext] + list(parameters)
            #print(type(f_ext(*arguments)))
            #print(type(f_auto(*arguments)))

            #return tuple(map(sum, zip(f_auto(*arguments), tuple(val_ext*stimulation(t) for val_ext in f_ext(*arguments)))))
            return f_auto(*arguments)#, )

        return f_ODEINT

    def compile_integrator(self, **kwargs) -> None:
        self.f_ODEINT = self.get_precompiled_integrator(**kwargs)

    def get_trajectories(self, t_span, state0 = None, parameter_values={}, max_step=0.01, **kwargs):
        '''
        Return the trajectories of the dynamical system using "scipy.integrate.solve_ivp".

        Comment on "max_step":
        The integration seems to give spurious and non-reliable results, if the choice of "max_step" is left to "solve_ivp"

        '''
        # if no state is given, choose randomly
        if state0 is None:
            state0 = rng.standard_normal(size=self.DIMENSION)

        # create a properly ordered list from the dictionary "parameter_values"
        parameter_list = [parameter_values[p] for p in self.PARAMETERS]

        # integrate 
        states = sc.integrate.solve_ivp(self.f_ODEINT, t_span, state0, args=(parameter_list, ), max_step=max_step, **kwargs)

        return states

    # def get_trajectories_numbalsoda(self):
        
    #     f_auto = nb.jit(sy.utilities.lambdify(tuple(self.VARIABLES + self.PARAMETERS), tuple(self.ODE)), nopython=True)

    #     @nb.cfunc(lsoda_sig)
    #     def rhs(t, u, du, p):
    #         arguments = list(u) + list(p)
    #         du = f_auto(*arguments)
    #         #du[0] = u[0]-u[0]*u[1]
    #         #du[1] = u[0]*u[1]-u[1]*p[0]

    #     #rhs = nb.cfunc()

    #     funcptr = rhs.address # address to ODE function
    #     u0 = np.array([5.,0.8]) # Initial conditions
    #     data = np.array([1.0]) # data you want to pass to rhs (data == p in the rhs).
    #     t_eval = np.linspace(0.0, 50.0, 1000) # times to evaluate solution

    #     # integrate with lsoda method
    #     usol, success = lsoda(funcptr, u0, t_eval, data = data)
    #     print(success)
    #     # integrate with dop853 method
    #     #usol1, success1 = dop853(funcptr, u0, t_eval, data = data)

    #     # usol = solution
    #     # success = True/False

    #     return usol

    def get_event_based_evolution(self, state0, parameter_values, event, event_settings, 
                                  t_start=0, N_events=10, T_max=100, **kwargs):
        '''
        returns the trajectories for an evolution whose parameters change every time an event is found

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
        Time = np.array([t_start])

        # intialize event counter 
        n = 0

        while n <= N_events and Time[-1] < t_start + T_max:
            # integrate until event
            sol = self.get_trajectories((Time[-1], t_start + T_max), 
                                        states[:,-1], 
                                        parameter_values=params, 
                                        events=event, **kwargs)
          
            # store results
            states = np.concatenate((states, sol.y), axis=1)
            Time = np.concatenate((Time, sol.t))

            #print('event found at t=' + str(Time[-1]))
            n += 1

            # store event index
            events_idx.append(len(Time))

            # iterate through the triggered sequence
            for T_event, parameters_values_event, state_action in event_settings:
                # update parameters
                params.update(parameters_values_event)

                # change state
                if state_action is None:
                    state_action = lambda state: state
 
                # integrate with parameter values changed by the event
                sol = self.get_trajectories((Time[-1], Time[-1] + T_event), 
                                            state_action(states[:,-1]),
                                            parameter_values=params, **kwargs)

                # store results
                states = np.concatenate((states, sol.y), axis=1)
                Time = np.concatenate((Time, sol.t))

            # reset parameters
            params.update(parameter_values)

        # drop the last event
        # because it might be computed by reaching "t_max", not by finding the event

        return states[:,:events_idx[-1]], Time[:events_idx[-1]], events_idx[:-1]

    def get_limit_cycle(self, params, event, state0=None, t_eq=100, samples=1000, isostable_expansion_order=0, **kwargs):
        '''
        returns:
        Time:       instances of time, Time[-1] is the period
        y:          limit cycle expansion; 
                    y[0] is the limit cycle
                    y[1], y[2]
        extra[0]:   Jacobian
        extra[1]:   fundamental matrix   
        extra[2]:   d2_special
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

        T = sol_eq.t_events[0][-1]-sol_eq.t_events[0][-2]

        Time = np.linspace(0, T, samples)

        sol_LC  = self.get_trajectories(t_span=(0., T), 
                                        t_eval = Time, 
                                        state0 = sol_eq.y_events[0][-1],
                                        parameter_values=params,
                                        **kwargs)

        # prepare return array
        y = np.zeros((isostable_expansion_order+1, 2, samples))

        # grab limit-cycle orbit
        y[0] = sol_LC.y

        extras = []
        if isostable_expansion_order>=1:

            ### calculate Jacobian at limit cycle ###

            self.calculate_Jacobian()
            J_np = sy.utilities.lambdify(tuple(self.VARIABLES), self.JACOBIAN.doit().subs(params), cse=True)
            
            J = np.zeros((self.DIMENSION, self.DIMENSION, samples))

            for t in range(samples):
                J[:,:,t] = J_np(*y[0,:,t])

            ### EXTRA 1 ### --> "extra" to dictionary
            extras.append(J)

            ### calculate fundamental solution matrix ###

            system_O1 = self.new_perturbed(order=1)

            fund_matrix = np.zeros((self.DIMENSION, self.DIMENSION, len(Time))) 
            fund_matrix[:,:,0] = np.eye(self.DIMENSION)

            for n in range(self.DIMENSION):
                # design the initial state
                state0 = np.zeros(2*self.DIMENSION)
                state0[:self.DIMENSION] = y[0,:,0]
                state0[self.DIMENSION + n] = 1.

                sol = system_O1.get_trajectories(t_span=(0,Time[-1]),
                                                t_eval=Time,
                                                state0=state0,
                                                parameter_values=params,
                                                **kwargs)

                fund_matrix[:,n,:] = sol.y[self.DIMENSION:,:]
            
            ### EXTRA 2 ###
            extras.append(fund_matrix)

            # eigenvalues/-vectors of monodromy matrix (this is for N=2 only!!)
            # this selection process has to be revisited!
            w, v = np.linalg.eig(fund_matrix[:,:,-1])
            non_unity_eigenvec = v.transpose()[np.abs(w-1) > 1e-4][0]

            # this is numerical unstable for large |kappa|, consider changing to trace formula
            #kappa = np.log(np.min(w))/Time[-1]
            kappa = integrate.trapezoid(np.trace(J, axis1=0, axis2=1), Time)/Time[-1]

            y[1] = np.array([np.exp(-kappa*Time[t])*np.matmul(fund_matrix[:,:,t], non_unity_eigenvec) for t in range(len(Time))]).transpose()
            #y[1] = np.array([np.power(np.min(w),-Time[t]/Time[-1])*np.matmul(fund_matrix[:,:,t], non_unity_eigenvec) for t in range(len(Time))]).transpose()

            if isostable_expansion_order>=2:

                ### calculate special solution for d2 ###

                system_O2 = self.new_perturbed(order=2)

                state0 = np.zeros(3*self.DIMENSION)
                state0[:self.DIMENSION] = y[0,:,0]
                state0[self.DIMENSION: 2*self.DIMENSION] = non_unity_eigenvec

                sol = system_O2.get_trajectories(t_span=(0,Time[-1]),
                                                t_eval=Time,
                                                state0=state0,
                                                parameter_values=params)
                
                d2_special = sol.y[2*self.DIMENSION:,:]

                y2_data_ini = np.matmul(np.linalg.inv(np.exp(2.*kappa*Time[-1])*np.eye(2)-fund_matrix[:,:,-1]), d2_special[:,-1])
                y[2] = np.array([np.exp(-2.*kappa*Time[t])*(np.matmul(fund_matrix[:,:,t], y2_data_ini[:]) + d2_special[:,t]) for t in range(len(Time))]).transpose()

                ### EXTRA 3 ###
                extras.append(d2_special)

        return Time, y, extras

    def get_isochrones_isostables(self, params, event, r = 1e-7, t_max = [20,20], kwargs_limit_cycle={}, **kwargs_int):
        '''infer phase-isostable structure for 2D system by backward integration'''

        # obtain limit cycle
        Time, y, extras = self.get_limit_cycle(params, event, isostable_expansion_order=1,
                                         **kwargs_limit_cycle,
                                         #t_eq=250, state0=[1.,1.], 
                                         #samples=samples, 
                                         **kwargs_int)

        # time-inverted system
        system_inv = self.new_time_inverted()

        # initial curve
        samples = len(Time)-1

        T = Time[-1]
        time_samples = [np.arange(0, t_max, T/samples) for t_max in t_max]

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
                    y_T = sol.y.T
                    temp[:len(y_T[:,0]), :len(y_T[0,:])] = y_T
                
                states[sign][s,:,:] = temp

            # rolling the states times further in time to form isochrones
            
            for t,T in enumerate(time_samples[z][:]):
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

        if self.DIMENSION != 2:
            print('This works only for 2-dimensional systems!')
            return 

        # compute fixed points
        fp = self.get_fixed_points()
        
        #x0 = [self.VARIABLES[i].subs(fp) for i in range(self.Dim)

        # compute Jacobian at fixed point "fp[0]"
        self.calculate_Jacobian()
        DF = self.JACOBIAN.doit().subs(fp[0]).subs(params)
        print(DF)
        DF_np = np.array(DF).astype(np.float64)

        # compute eigenvalues and left eigenvector numerically
        eigenvalues, eigenvectors = np.linalg.eig(np.transpose(DF_np))
        
        if np.imag(eigenvalues[0]) == 0:
            print('This function works only for fixed points of focus type!')
            return
        
        p = eigenvectors[0,0]
        q = eigenvectors[1,0]

        def ellipse_radius(x):
            return 1./np.sqrt(np.abs(p)**2*np.cos(x)**2 + np.abs(q)**2*np.sin(x)**2 + 2*np.real(p*np.conj(q))*np.cos(x)*np.sin(x))

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

            
        TODO    review whether the new DynamicalSystem is really necessary
        TODO    check for degenerate case of eigenvalues with multipl. > 1
        '''

        if self.DIMENSION != 2:
            print('This function is still only implemented for 2-dimensional systems.')
            return list()
        
        # new DynamicalSystem with fixed parameters
        system = self.new_parameter_set(params)

        # compute Jacobian
        system.calculate_Jacobian()

        # compute fixed points
        fixed_points = system.get_fixed_points()
        
        output = list()

        for fp in fixed_points:
            # Jacobian at fixed point 
            DF = system.JACOBIAN.doit().subs(fp)
            
            DF_np = np.array(DF).astype(np.float64)
            fp_np = np.array(list(fp.values())).astype(np.float64)

            # compute eigenvalues and right eigenvectors numerically
            eigenvalues, eigenvectors_right = np.linalg.eig(DF_np)

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

                    return fp_np + 2*r*np.real(c1*np.exp(1j*psi)) #np.matmul(eigenvectors, a)

            output.append([eigenvalues, x_from_isostable, isostable_from_x])

        return output

    def get_time_averaged_Jacobian(self, params, **kwargs):
        #'''this function calculates the angular frequency and the Floquet exponent of a 2D limit cycle system'''
        
        #T, x0, y0 = self.get_limit_cycle(params, **kwargs_LC)
        
        self.calculate_Jacobian()
        J_np = sy.utilities.lambdify(tuple(self.VARIABLES), self.JACOBIAN.doit().subs(params), cse=True)
        
        sol = self.get_trajectories(parameter_values=params, **kwargs)

        J_int = np.zeros((self.DIMENSION, self.DIMENSION, len(sol.t)))

        for i in range(self.DIMENSION):
            for j in range(self.DIMENSION):
                J_ij_at_LC = [J_np(sol.y[0,t], sol.y[1,t])[i,j] for t in range(len(sol.t))]
                J_int[i,j] = integrate.cumulative_trapezoid(J_ij_at_LC, sol.t, initial=0)/(sol.t[-1]-sol.t[0]) 

        return J_int