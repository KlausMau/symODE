import sympy as sy

'''
Catalogue of dynamical systems:

- Reserved name for external stimulation: "I_ext"
- use only keyword arguments

Documentation:
- future: make functions possible (also visually)
'''

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

def Kuramoto(N=2):
    phi = sy.symbols('phi')
    om = sy.symbols('omega')

    eps = sy.symbols('epsilon')

    
    ODE = {}
    for i in range(N):
        phi_i = sy.symbols(str(phi) + '_' + str(i+1))
        om_i  = sy.symbols(str(om)  + '_' + str(i+1))

        # calculate coupling term
        coupling_term = 0
        for j in range(N):
            phi_j = sy.symbols(str(phi) + '_' + str(j+1))
            coupling_term += sy.sin(phi_j-phi_i)
        coupling_term *= eps/N

        ODE.update({phi_i: om_i + coupling_term})
    return ODE

# 1D #

def Adler():
    D = sy.symbols('Delta')
    
    eta = sy.symbols('eta')
    eps = sy.symbols('epsilon')
    
    return  {D: eta + eps*sy.sin(D)}

def linear_1D():
    x= sy.symbols('x')
    
    k = sy.symbols('kappa')
    
    return  {x: k*x}

def constant():
    x = sy.symbols('x')
    return {x: 1}

# 2D #

def Stuart_Landau():
    x = sy.symbols('x')
    y = sy.symbols('y')
    
    mu = sy.symbols('mu')
    w = sy.symbols('omega')
    a = sy.symbols('alpha')

    return {x: mu*x -w*y - (x**2+y**2)*(x-a*y), y: mu*y + w*x - (x**2+y**2)*(y+a*x)}  

def van_der_Pol():
    x = sy.symbols('x')
    y = sy.symbols('y')
    eps = sy.symbols('epsilon')

    return {x: y, y: -x + eps*(1-x**2)*y}  

def van_der_Pol_Lienard():
    # should be replaced by transformations soon
    x = sy.symbols('x')
    y = sy.symbols('y')
    eps = sy.symbols('epsilon')

    return {x: -y + eps*(x-x**3/3), y: x}

def FitzHugh_Nagumo():
    x = sy.symbols('x')
    y = sy.symbols('y')
    
    I = sy.symbols('I')
    tau = sy.symbols('tau')
    a = sy.symbols('a')
    b = sy.symbols('b')
    
    return {x: x - x**3/3 - y + I, y: (x + a - b*y)/tau}  

def harmonic_oscillator():
    x = sy.symbols('x')
    y = sy.symbols('y')
    
    g = sy.symbols('gamma')
    w = sy.symbols('omega')
    
    return {x: y, y: -2*g*w*y - w**2*x}

def rayleigh():
    x = sy.symbols('x')
    y = sy.symbols('y')
    mu = sy.symbols('mu')
    w = sy.symbols('omega') 

    return {x: y, y: mu*(1-y**2)*y - w*w*x} 

def homoclinic():
    x,y = sy.symbols('x,y')
    mu1,mu2 = sy.symbols('mu_1, mu_2')
    return {x: mu1*x + y, y: (mu2-mu1)*y + x**2 - x*y}

def sin_cos_LC():
    # should be replaced by transformations soon
    x = sy.symbols('x')
    y = sy.symbols('y')

    return {x: y - sy.sin(y)*x/2, y: -x + sy.cos(x)*y/2}

def infinity_oscillator():
    x = sy.symbols('x')
    y = sy.symbols('y')
    
    w = sy.symbols('omega')
    k = sy.symbols('kappa')
    r = sy.symbols('r')
    a = sy.symbols('alpha')

    # functions
    C = -2*x*y/((r+2)*x**2 + r*y**2)
    D = (x**2+y**2)**2/((r+2)*x**2 + r*y**2)

    return {x: w*(x*C-y) + k/2*(D-1)*(x+a*(x*C-y)), 
            y: w*(y*C+x) + k/2*(D-1)*(y+a*(y*C+x))}

def isostable_2D():
    psi = sy.symbols('psi')
    phi = sy.symbols('phi')
    
    w = sy.symbols('omega')
    k = sy.symbols('kappa')

    return {phi: w, psi: k*psi}

def coupled_phase_oscillators():
    p1 = sy.symbols('phi_1')
    p2 = sy.symbols('phi_2')
    
    w1 = sy.symbols('omega_1')
    w2 = sy.symbols('omega_2')
    eps = sy.symbols('epsilon')
    alpha = sy.symbols('alpha')
    I = sy.symbols('I_ext')

    return {p1: w1 + eps*sy.sin(p1-p2+alpha) + I*sy.sin(p1), p2: w2 + eps*sy.sin(p2-p1+alpha)}  

def coupled_oscillators_isostable():
    # not needed anymore #

    phi = sy.symbols('varphi')
    psi = sy.symbols('psi')

    w = sy.symbols('Omega')
    eta = sy.symbols('eta')
    eps = sy.symbols('epsilon')
    I = sy.symbols('I_ext')

    C = eta/eps
    A = -2*sy.sqrt(1-C**2)
    B = sy.tan(sy.asin(C)/2)

    phi_2 = phi - sy.atan((1-psi/A*B)/(psi/A - B))

    d_psi = 1/(4*(1-C**2))*psi**2 + C/sy.sqrt(1-C**2)*psi + 1

    # PRC = sin
    Z = sy.sin

    return {phi: w + I*Z(phi_2)/2, psi: -2*eps*sy.sqrt(1-C**2)*psi - I*d_psi*Z(phi_2)} 

def coupled_phase_oscillators_harmonics():
    p1 = sy.symbols('phi_1')
    p2 = sy.symbols('phi_2')
    
    w1 = sy.symbols('omega_1')
    w2 = sy.symbols('omega_2')
    eps = sy.symbols('epsilon')
    #alpha = sy.symbols('alpha')
    sigma = sy.symbols('sigma')
    beta = sy.symbols('beta')
    I = sy.symbols('I_ext')

    return {p1: w1 + eps*sy.sin(p2-p1) + sigma*sy.sin(2*(p2-p1)) + I*sy.sin(p1), 
            p2: w2 + eps*sy.sin(p1-p2) + beta*sy.sin(3*(p1-p2))}  

def linear_2D(variables=[sy.symbols('x'), sy.symbols('y')]):
    ''' "variables" is a list of Sympy symbols '''

    x = variables[0]
    y = variables[1]

    c= sy.symbols('c')
    return  {x: x - c*y, y: c*x + y}

def oscillator_Rok():
    x = sy.symbols('x')
    y = sy.symbols('y')
    a = sy.symbols('a')
    b = sy.symbols('b') 

    return {x: y - a*sy.sin(y)*x, y: -x + b*sy.cos(x)*y} 

# 3D #

def Lorenz():
    x = sy.symbols('x')
    y = sy.symbols('y')
    z = sy.symbols('z')
    
    sigma = sy.symbols('sigma')
    rho = sy.symbols('rho')
    beta = sy.symbols('beta')

    return {x: sigma * (y - x), y: x * (rho - z) - y, z: x * y - beta * z }

def Roessler():
    x = sy.symbols('x')
    y = sy.symbols('y')
    z = sy.symbols('z')
    
    a = sy.symbols('a')
    b = sy.symbols('b')
    c = sy.symbols('c')

    return {x: -y-z, y: x + a*y, z: b+z*(x-c)}

def Hindmarsh_Rose():
    x = sy.symbols('x')
    y = sy.symbols('y')
    z = sy.symbols('z')

    a = sy.symbols('a')
    b = sy.symbols('b')
    c = sy.symbols('c')
    d = sy.symbols('d')
    r = sy.symbols('r')
    s = sy.symbols('s')
    I = sy.symbols('I')
    x_R = sy.symbols('x_R')

    return {x: y - a*x**3 + b*x**2 - z + I, 
            y: c - d*x**2 - y, 
            z: r*(s*(x-x_R)-z)}

def isostable_3D():
        psi_1 = sy.symbols('psi_1')
        psi_2 = sy.symbols('psi_2')
        phi = sy.symbols('phi')
        
        w = sy.symbols('omega')
        k_1 = sy.symbols('kappa_1')
        k_2 = sy.symbols('kappa_2')

        return {phi: w, psi_1: k_1*psi_1, psi_2: k_2*psi_2}