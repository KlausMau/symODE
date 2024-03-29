import numpy as np
import scipy as sc

from scipy import optimize

def min_sqsum(A,b):
    ''' 
    returns c from overdetermined linear equation A*c = b by method of least squares 
    '''
    return np.linalg.solve(A.transpose().dot(A), A.transpose().dot(b))

class CircularRealFunction:
    
    # initialization
    def __init__(self, **kwargs):
        self.set_shape(**kwargs)
    
    # overwrite the Fourier modes by predefined ones (centred at 0); default: "f=0"
    def set_shape(self, FOURIER_MODES = None, MAX_MODE_NUMBER = 10, TYPE = None, **properties) -> None:
        if FOURIER_MODES is not None:
            self.FOURIER_MODES = np.array(FOURIER_MODES, dtype=complex)
        else:
            # if not, choose a predefined type with given "MAX_MODE_NUMBER"

            # first check for valid "MAX_MODE_NUMBER"
            if MAX_MODE_NUMBER < 0:
                raise ValueError('maximum Fourier mode number has to be a positive integer.')

            # evaluate "TYPE"
            if TYPE == None:
                self.FOURIER_MODES = np.array([0, 0], dtype=complex)
            #elif TYPE == 'type-I':
            #    self.set_from_function(PRC_type_I, MAX_MODE_NUMBER = MAX_MODE_NUMBER, **properties)
            #elif TYPE == 'type-II':
            #    self.set_from_function(PRC_type_II, MAX_MODE_NUMBER = MAX_MODE_NUMBER, **properties)
            #elif TYPE == 'Stuart-Landau_PRC':
            #    self.FOURIER_MODES = PRC_Stuart_Landau(**properties)
            #elif TYPE == 'Stuart-Landau_IRC':
            #    self.FOURIER_MODES = IRC_Stuart_Landau(**properties)
            #elif TYPE == 'van-der-Pol':
            #    self.FOURIER_MODES = PRC_van_der_Pol()
            else:
                raise ValueError('unrecognized value for "TYPE"')
   
    # shift the Fourier modes by "phi0"
    def shift_by(self, phi0: float) -> None:
        for n in range(len(self.FOURIER_MODES)):
            # f_k |-> f_k * e^(i*k*phi0))
            self.FOURIER_MODES[n] *= np.exp(1j*n*phi0)

    # centre the distribution at "phi0"
    def centre_at(self, phi0: float) -> None:
        theta = np.angle(self.FOURIER_MODES[1])
        self.shift_by(phi0-theta)

    # set Fourier modes from data by fitting (method of least squares)
    def set_from_data(self, x, y, MAX_MODE_NUMBER = 10):
        modes = np.zeros(MAX_MODE_NUMBER + 1, dtype = complex)
        np_x = np.array(x)
        np_y = np.array(y)

        if np_x.size < 2*MAX_MODE_NUMBER + 1:
            raise ValueError('Number of desired Fourier modes too large for sample number!') 
        else:
            L = np.zeros((np_x.size, 2*MAX_MODE_NUMBER + 1))
            for i in range(np_x.size):
                L[i,0] = 1.
                for j in range(MAX_MODE_NUMBER):
                    L[i,2*j+1] = 2.*np.cos((j+1)*np_x[i])
                    L[i,2*j+2] = 2.*np.sin((j+1)*np_x[i])
    
            modes_sincos = min_sqsum(L,np_y)
    
            # translate sin and cos to complex modes  
            modes[0] = modes_sincos[0]
            for n in range(1,MAX_MODE_NUMBER + 1):
                modes[n] = modes_sincos[2*n-1] + 1j*modes_sincos[2*n]

        self.FOURIER_MODES = modes           

    # set Fourier modes by generating data from function 
    def set_from_function(self, f, MAX_MODE_NUMBER = 10, samples = 100, **properties) -> None:
        x = np.linspace(0, 2*np.pi, samples)
        y = f(x, **properties)
        self.set_from_data(x, y, MAX_MODE_NUMBER = MAX_MODE_NUMBER)

    def set_from_sincos_modes(self, SINCOS_MODES) -> None:
        '''sets the function from modes '''
        MAX_MODE_NUMBER = int((len(SINCOS_MODES)+1)/2)
        self.FOURIER_MODES = np.array([SINCOS_MODES[0]] + [0.5*(SINCOS_MODES[2*i+1] + 1j*SINCOS_MODES[2*i+2]) for i in range(MAX_MODE_NUMBER-1)])

    def set_zero_at(self, value, direction=1, guesses = [0., np.pi]) -> None:
        '''shift f such that f(0) = value and sign(f'(0)) = dir'''

        # search for all arguments x0 with f(x0)=value
        x0 = []
        for guess in guesses:
            roots = optimize.root(lambda x: self.get_values_at(x)-value, guess)
            x0.extend(roots.x)

        # select first that matches the direction
        df = self.get_derivative()
        i = 0
        stop = False
        while stop==False and i<len(x0):
            if np.sign(df.get_values_at(x0[i])) == direction:
                stop = True
            else:
                i += 1

        # shift f
        self.shift_by(-x0[i])

    def get_values_at(self, x):
        ''' 
        returns Numpy array with the function values of "x" (same size)
        f(x) = f_0 + 2*sum_i=1^N Re(f_i*exp(-ikx)) 
        '''

        f_modes = self.FOURIER_MODES
        np_x = np.array(x)
        return np.real(f_modes[0]*np.ones(np_x.size) + 2*np.sum([f_modes[k]*np.exp(-1j*k*np_x) for k in range(1, f_modes.size)], axis=0))

    # return a "CircularFunction" being the derivative of "self"
    def get_derivative(self):
        f_modes = self.FOURIER_MODES
        df = CircularRealFunction()
        df.FOURIER_MODES = np.array([(-1j*n)*f_modes[n] for n in range(len(f_modes))], dtype=complex)
        return df

    # return a "CircularRealFunction" being the multiplication "self" with another "CircularRealFunction"
    def get_multiplication(self, g): 
        f_modes = self.FOURIER_MODES
        g_modes = g.FOURIER_MODES

        N_f = f_modes.size - 1
        N_g = g_modes.size - 1
        N_h = N_f + N_g
        h_modes = np.zeros(N_h + 1, dtype=complex)

        for k in range(h_modes.size):
            l = 1
            while l <= np.min([N_f-k, N_g]):
                h_modes[k] += f_modes[k+l]*np.conj(g_modes[l])
                l+=1
        
            l = np.max([0, k-N_f])
            while l <= np.min([k, N_g]):
                h_modes[k] += f_modes[k-l]*g_modes[l]
                l+=1

            l = k+1
            while l <= np.min([N_f+k, N_g]):
                h_modes[k] += np.conj(f_modes[l-k])*g_modes[l]
                l+=1

        h = CircularRealFunction()
        h.FOURIER_MODES = h_modes
        return h 

    # return argument and value for minimum  
    def get_min(self, samples = 100):
        x = np.linspace(0, 2.*np.pi, samples)
        y = self.get_values_at(x)

        # index, that yields maximum value
        min_index = np.argmin(y)

        argument = x[min_index]
        value = y[min_index]

        return argument, value

    # return argument and value for maximum  
    def get_max(self, samples = 100):
        x = np.linspace(0, 2.*np.pi, samples)
        y = self.get_values_at(x)
        
        # index, that yields maximum value
        max_index = np.argmax(y)

        argument = x[max_index]
        value = y[max_index]

        return argument, value

    # return value and sign of derivative for given argument  
    def get_threshold(self, x):
        df = self.get_derivative()
        return np.array([self.get_values_at(x), np.sign(df.get_values_at(x))])
