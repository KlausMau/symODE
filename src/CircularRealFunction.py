import numpy as np
from scipy import optimize

def min_sqsum(a,b):
    '''
    returns c from overdetermined linear equation a*c = b by method of least squares
    '''
    return np.linalg.solve(a.transpose().dot(a), a.transpose().dot(b))

class CircularRealFunction:

    # initialization
    def __init__(self, fourier_modes = np.array([0.])) -> None:
        '''initialize a CircularRealFunction (default: zero function)'''
        self._fourier_modes = np.array(fourier_modes, dtype=complex)

    # shift the Fourier modes by "phi0"
    def shift_by(self, phi0: float) -> None:
        for n in range(len(self._fourier_modes)):
            # f_k |-> f_k * e^(i*k*phi0))
            self._fourier_modes[n] *= np.exp(1j*n*phi0)

    # centre the distribution at "phi0"
    def centre_at(self, phi0: float) -> None:
        theta = np.angle(self._fourier_modes[1])
        self.shift_by(phi0-theta)

    # set Fourier modes from data by fitting (method of least squares)
    def set_from_data(self, x, y, maximum_mode_number = 10):
        modes = np.zeros(maximum_mode_number + 1, dtype = complex)
        np_x = np.array(x)
        np_y = np.array(y)

        if np_x.size < 2*maximum_mode_number + 1:
            raise ValueError('Number of desired Fourier modes too large for sample number!')
        else:
            l = np.zeros((np_x.size, 2*maximum_mode_number + 1))
            for i in range(np_x.size):
                l[i,0] = 1.
                for j in range(maximum_mode_number):
                    l[i,2*j+1] = 2.*np.cos((j+1)*np_x[i])
                    l[i,2*j+2] = 2.*np.sin((j+1)*np_x[i])

            modes_sincos = min_sqsum(l,np_y)

            # translate sin and cos to complex modes
            modes[0] = modes_sincos[0]
            for n in range(1,maximum_mode_number + 1):
                modes[n] = modes_sincos[2*n-1] + 1j*modes_sincos[2*n]

        self._fourier_modes = modes

    # set Fourier modes by generating data from function
    def set_from_function(self, f, maximum_mode_number = 10, samples = 100, **properties) -> None:
        x = np.linspace(0, 2*np.pi, samples)
        y = f(x, **properties)
        self.set_from_data(x, y, maximum_mode_number = maximum_mode_number)

    def set_from_sincos_modes(self, sincos_modes) -> None:
        '''sets the function from modes '''
        maximum_mode_number = int((len(sincos_modes)+1)/2)
        self._fourier_modes = np.array([sincos_modes[0]] + [0.5*(sincos_modes[2*i+1] + 1j*sincos_modes[2*i+2]) for i in range(maximum_mode_number-1)])

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

        f_modes = self._fourier_modes
        np_x = np.array(x)
        return np.real(f_modes[0]*np.ones(np_x.size) + 2*np.sum([f_modes[k]*np.exp(-1j*k*np_x) for k in range(1, f_modes.size)], axis=0))

    # return a "CircularFunction" being the derivative of "self"
    def get_derivative(self):
        f_modes = self._fourier_modes
        df = CircularRealFunction()
        df._fourier_modes = np.array([(-1j*n)*f_modes[n] for n in range(len(f_modes))], dtype=complex)
        return df

    # return a "CircularRealFunction" being the multiplication "self" with another "CircularRealFunction"
    def get_multiplication(self, g):
        f_modes = self._fourier_modes
        g_modes = g.FOURIER_MODES

        n_f = f_modes.size - 1
        n_g = g_modes.size - 1
        n_h = n_f + n_g
        h_modes = np.zeros(n_h + 1, dtype=complex)

        for k in range(h_modes.size):
            l = 1
            while l <= np.min([n_f-k, n_g]):
                h_modes[k] += f_modes[k+l]*np.conj(g_modes[l])
                l+=1

            l = np.max([0, k-n_f])
            while l <= np.min([k, n_g]):
                h_modes[k] += f_modes[k-l]*g_modes[l]
                l+=1

            l = k+1
            while l <= np.min([n_f+k, n_g]):
                h_modes[k] += np.conj(f_modes[l-k])*g_modes[l]
                l+=1

        h = CircularRealFunction()
        h._fourier_modes = h_modes
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
