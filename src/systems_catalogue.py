'''
This module provides a catalogue for well-known dynamical systems.
'''

import sympy as sy

# arbitrary dimension #

def linear(variables=[sy.symbols('x')], param_name = 'a'):
    # "variables" is a list of Sympy symbols
    ODE = {}
    for i in range(len(variables)):
        f_i = 0
        for j in range(len(variables)):
            f_i += sy.symbols(param_name + '_' + str(variables[i]) + '_' + str(variables[j]))*variables[j]
        ODE.update({variables[i]: f_i})
    return  ODE

def Kuramoto(number_of_oscillators=2):
    phi = sy.symbols('phi')
    omega, epsilon = sy.symbols('omega epsilon')

    ODE = {}
    for i in range(number_of_oscillators):
        phi_i = sy.symbols(str(phi) + '_' + str(i+1))
        om_i  = sy.symbols(str(omega)  + '_' + str(i+1))

        # calculate coupling term
        coupling_term = 0
        for j in range(number_of_oscillators):
            phi_j = sy.symbols(str(phi) + '_' + str(j+1))
            coupling_term += sy.sin(phi_j-phi_i)
        coupling_term *= epsilon/number_of_oscillators

        ODE.update({phi_i: om_i + coupling_term})
    return ODE

# 1D #

def Adler():
    delta = sy.symbols('Delta')
    eta, epsilon = sy.symbols('eta epsilon')

    return  {delta: eta + epsilon*sy.sin(delta)}

def linear_1D():
    x= sy.symbols('x')
    kappa = sy.symbols('kappa')

    return  {x: kappa*x}

def constant():
    x = sy.symbols('x')

    return {x: 1}

# 2D #

def Stuart_Landau():
    x, y = sy.symbols('x y')
    mu, omega, alpha = sy.symbols('mu omega alpha')

    return {x: mu*x - omega*y - (x**2+y**2)*(x-alpha*y),
            y: mu*y + omega*x - (x**2+y**2)*(y+alpha*x)}

def van_der_Pol():
    x, y = sy.symbols('x y')
    epsilon = sy.symbols('epsilon')

    return {x: y,
            y: -x + epsilon*(1-x**2)*y}

def van_der_Pol_Lienard():
    # should be replaced by transformations soon
    x, y = sy.symbols('x y')
    epsilon = sy.symbols('epsilon')

    return {x: epsilon*(x-x**3/3)-y,
            y: x}

def FitzHugh_Nagumo():
    '''FitzHugh-Nagumo model (https://doi.org/10.1016/S0006-3495(61)86902-6)'''
    x, y = sy.symbols('x y', real=True)
    a, b, i, tau = sy.symbols('a b I tau')

    return {x: x - x**3/3 - y + i,
            y: (x + a - b*y)/tau}

def harmonic_oscillator():
    x, y = sy.symbols('x y')
    gamma, omega = sy.symbols('gamma omega')

    return {x: y,
            y: -2*gamma*omega*y - omega**2*x}

def rayleigh():
    '''Rayleigh oscillator (www.google.com)'''
    x, y = sy.symbols('x y')
    mu, omega = sy.symbols('mu omega')

    return {x: y,
            y: mu*(1-y**2)*y - omega*omega*x}

def homoclinic():
    x, y = sy.symbols('x y')
    mu1, mu2 = sy.symbols('mu_1 mu_2')

    return {x: mu1*x + y,
            y: (mu2-mu1)*y + x**2 - x*y}

def sin_cos_LC():
    # should be replaced by transformations soon
    x, y = sy.symbols('x y')

    return {x: y - sy.sin(y)*x/2,
            y: -x + sy.cos(x)*y/2}

def infinity_oscillator():
    x, y = sy.symbols('x y')
    omega, kappa, r, alpha = sy.symbols('omega kappa r alpha')

    # functions
    C = -2*x*y/((r+2)*x**2 + r*y**2)
    D = (x**2+y**2)**2/((r+2)*x**2 + r*y**2)

    return {x: omega*(x*C-y) + kappa/2*(D-1)*(x+alpha*(x*C-y)),
            y: omega*(y*C+x) + kappa/2*(D-1)*(y+alpha*(y*C+x))}

def isostable_2D():
    phi, psi = sy.symbols('phi psi')
    omega, kappa = sy.symbols('omega kappa')

    return {phi: omega,
            psi: kappa*psi}

def coupled_phase_oscillators():
    phi_1, phi_2 = sy.symbols('phi_1 phi_2')
    omega_1, omega_2, epsilon, alpha, i_ext = sy.symbols('omega_1 omega_2 epsilon alpha I_ext')

    return {phi_1: omega_1 + epsilon*sy.sin(phi_1-phi_2+alpha) + i_ext*sy.sin(phi_1),
            phi_2: omega_2 + epsilon*sy.sin(phi_2-phi_1+alpha)}

def coupled_oscillators_isostable():
    # not needed anymore #
    phi, psi = sy.symbols('varphi psi')
    w, eta, eps, i_ext = sy.symbols('Omega eta epsilon I_ext')

    c = eta/eps
    aux_a = -2*sy.sqrt(1-c**2)
    aux_b = sy.tan(sy.asin(c)/2)

    phi_2 = phi - sy.atan((1-psi/aux_a*aux_b)/(psi/aux_a - aux_b))
    d_psi = 1/(4*(1-c**2))*psi**2 + c/sy.sqrt(1-c**2)*psi + 1

    # PRC = sin
    Z = sy.sin

    return {phi: w + i_ext*Z(phi_2)/2,
            psi: -2*eps*sy.sqrt(1-c**2)*psi - i_ext*d_psi*Z(phi_2)}

def coupled_phase_oscillators_harmonics():
    phi_1, phi_2 = sy.symbols('phi_1 phi_2')
    omega_1, omega_2, epsilon, sigma, beta, i_ext = sy.symbols('omega_1 omega_2 epsilon sigma beta I_ext')

    return {phi_1: omega_1 + epsilon*sy.sin(phi_2-phi_1) + sigma*sy.sin(2*(phi_2-phi_1)) + i_ext*sy.sin(phi_1),
            phi_2: omega_2 + epsilon*sy.sin(phi_1-phi_2) + beta*sy.sin(3*(phi_1-phi_2))}

def linear_2D(variables=[sy.symbols('x'), sy.symbols('y')]):
    ''' "variables" is a list of Sympy symbols '''
    x = variables[0]
    y = variables[1]
    c = sy.symbols('c')

    return  {x: x - c*y,
             y: c*x + y}

def oscillator_Rok():
    x, y = sy.symbols('x y')
    a, b = sy.symbols('a b')

    return {x: y - a*sy.sin(y)*x,
            y: -x + b*sy.cos(x)*y}

# 3D #

def Lorenz():
    x, y, z = sy.symbols('x y z')
    sigma, rho, beta = sy.symbols('sigma rho beta')

    return {x: sigma * (y - x),
            y: x * (rho - z) - y,
            z: x * y - beta * z }

def Thomas():
    x, y, z = sy.symbols('x y z')
    b = sy.symbols('b')

    return {x: sy.sin(y) - b*x,
            y: sy.sin(z) - b*y,
            z: sy.sin(x) - b*z}

def Roessler():
    x, y, z = sy.symbols('x y z')
    a, b, c = sy.symbols('a b c')

    return {x: -y-z,
            y: x + a*y,
            z: b+z*(x-c)}

def Hindmarsh_Rose():
    x, y, z = sy.symbols('x y z')
    a, b, c, d, r, s, i, x_r = sy.symbols('a b c d r s I x_R')

    return {x: y - a*x**3 + b*x**2 - z + i,
            y: c - d*x**2 - y,
            z: r*(s*(x-x_r)-z)}

def isostable_3D():
    phi, psi_1, psi_2 = sy.symbols('phi psi_1 psi_2')
    omega, kappa_1, kappa_2 = sy.symbols('omega kappa_1 kappa_2')

    return {phi: omega,
            psi_1: kappa_1*psi_1,
            psi_2: kappa_2*psi_2}
