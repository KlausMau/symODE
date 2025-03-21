"""Module for circular real functions f: [0,2pi) -> R"""

import numpy as np
from scipy import optimize


def min_sqsum(matrix, y):
    """returns x from overdetermined linear equation matrix*x = y by method of least squares"""
    return np.linalg.solve(matrix.transpose().dot(matrix), matrix.transpose().dot(y))


class CircularRealFunction:
    """class for circular real functions f: [0,2pi) -> R"""

    def __init__(self, fourier_modes=np.array([0.0])) -> None:
        """initialize a CircularRealFunction (default: zero function)"""
        self._fourier_modes = np.array(fourier_modes, dtype=complex)

    def shift_by(self, phi0: float) -> None:
        """shift the Fourier modes by "phi0" """
        for n, _ in enumerate(self._fourier_modes):
            # f_k |-> f_k * e^(i*k*phi0))
            self._fourier_modes[n] *= np.exp(1j * n * phi0)

    def shift_with_zero_at(self, value, direction=1, guesses=None) -> None:
        """shift f such that f(0) = value and sign(f'(0)) = dir"""
        if guesses is None:
            guesses = [0.0, np.pi]

        # search for all arguments x0 with f(x0)=value
        x0 = []
        for guess in guesses:
            roots = optimize.root(lambda x: self.get_values_at(x) - value, guess)
            x0.extend(roots.x)

        # select first that matches the direction
        df = self.get_derivative()
        i = 0
        stop = False
        while stop is False and i < len(x0):
            if np.sign(df.get_values_at(x0[i])) == direction:
                stop = True
            else:
                i += 1

        # shift f
        self.shift_by(-x0[i])

    def shift_with_mean_at(self, phi0: float) -> None:
        """shift the function such the mean is at "phi0" """
        theta = np.angle(self._fourier_modes[1])
        self.shift_by(phi0 - theta)

    def set_from_data(self, x, y, maximum_mode_number=10):
        """set Fourier modes from data by fitting (method of least squares)"""
        modes = np.zeros(maximum_mode_number + 1, dtype=complex)
        np_x = np.array(x)
        np_y = np.array(y)

        if np_x.size < 2 * maximum_mode_number + 1:
            raise ValueError(
                "Number of desired Fourier modes too large for sample number!"
            )
        l = np.zeros((np_x.size, 2 * maximum_mode_number + 1))
        for i in range(np_x.size):
            l[i, 0] = 1.0
            for j in range(maximum_mode_number):
                l[i, 2 * j + 1] = 2.0 * np.cos((j + 1) * np_x[i])
                l[i, 2 * j + 2] = 2.0 * np.sin((j + 1) * np_x[i])

        modes_sincos = min_sqsum(l, np_y)

        # translate sin and cos to complex modes
        modes[0] = modes_sincos[0]
        for n in range(1, maximum_mode_number + 1):
            modes[n] = modes_sincos[2 * n - 1] + 1j * modes_sincos[2 * n]

        self._fourier_modes = modes

    def set_from_function(
        self, f, maximum_mode_number=10, samples=100, **properties
    ) -> None:
        """set Fourier modes by generating data from function"""
        x = np.linspace(0, 2 * np.pi, samples)
        y = f(x, **properties)
        self.set_from_data(x, y, maximum_mode_number=maximum_mode_number)

    def set_from_sincos_modes(self, sincos_modes) -> None:
        """sets the function from modes"""
        maximum_mode_number = int((len(sincos_modes) + 1) / 2)
        non_constant_modes = [
            0.5 * (sincos_modes[2 * i + 1] + 1j * sincos_modes[2 * i + 2])
            for i in range(maximum_mode_number - 1)
        ]
        self._fourier_modes = np.array([sincos_modes[0]] + non_constant_modes)

    def get_fourier_modes(self) -> np.ndarray:
        """returns the Fourier modes of the function"""
        return self._fourier_modes

    def get_values_at(self, x):
        """
        returns Numpy array with the function values of "x" (same size)
        f(x) = f_0 + 2*sum_i=1^N Re(f_i*exp(-ikx))
        """
        f_modes = self._fourier_modes
        np_x = np.array(x)
        temporary_name = 2 * np.sum(
            [f_modes[k] * np.exp(-1j * k * np_x) for k in range(1, f_modes.size)],
            axis=0,
        )
        return np.real(f_modes[0] * np.ones(np_x.size) + temporary_name)

    def get_derivative(self) -> "CircularRealFunction":
        """returns the derivative of the function"""
        fourier_modes = np.array(
            [(-1j * n) * fn for n, fn in enumerate(self._fourier_modes)], dtype=complex
        )
        return CircularRealFunction(fourier_modes=fourier_modes)

    def get_minimum(self, samples=100):
        """return argument and value for minimum"""
        x = np.linspace(0, 2.0 * np.pi, samples)
        y = self.get_values_at(x)

        # index, that yields maximum value
        min_index = np.argmin(y)

        argument = x[min_index]
        value = y[min_index]

        return argument, value

    def get_maximum(self, samples=100):
        """return argument and value for maximum"""
        x = np.linspace(0, 2.0 * np.pi, samples)
        y = self.get_values_at(x)

        # index, that yields maximum value
        max_index = np.argmax(y)

        argument = x[max_index]
        value = y[max_index]

        return argument, value

    def get_threshold(self, x):
        """return value and sign of derivative for given argument"""
        df = self.get_derivative()
        return np.array([self.get_values_at(x), np.sign(df.get_values_at(x))])


def multiply(
    f: CircularRealFunction, g: CircularRealFunction
) -> "CircularRealFunction":
    """returns the product of the function with another function g"""
    f_fourier_modes = f.get_fourier_modes()
    g_fourier_modes = g.get_fourier_modes()

    n_f = f_fourier_modes.size - 1
    n_g = g_fourier_modes.size - 1
    n_h = n_f + n_g
    h_modes = np.zeros(n_h + 1, dtype=complex)

    for k in range(h_modes.size):
        l = 1
        while l <= np.min([n_f - k, n_g]):
            h_modes[k] += f_fourier_modes[k + l] * np.conj(g_fourier_modes[l])
            l += 1

        l = np.max([0, k - n_f])
        while l <= np.min([k, n_g]):
            h_modes[k] += f_fourier_modes[k - l] * g_fourier_modes[l]
            l += 1

        l = k + 1
        while l <= np.min([n_f + k, n_g]):
            h_modes[k] += np.conj(f_fourier_modes[l - k]) * g_fourier_modes[l]
            l += 1

    return CircularRealFunction(fourier_modes=h_modes)
