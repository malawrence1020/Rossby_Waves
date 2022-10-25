"""Test file for plotting in terminal."""
import matplotlib.pyplot as plt
import numpy as np


# scale for amplitude
alpha = 1e11

# scale of sphericity for Rossby waves in Ocean
beta = 2e-11

# scaling for typical wavelength
n = 2e-6

# Rossby deformation radius
Rd = 1e5


def plot_test():
    plt.plot([1, 2, 3])
    plt.savefig('myfig.png')


def plot_streamfunction(xlim=(-2500, 2500, 5000),
                        ylim=(-2500, 2500, 5000),
                        t=0,
                        lines=50,
                        filled=True):
    """
    Contour plot of the streamfunction of a Rossby wave.

    Parameters
    ----------
    xlim : array_like
        (x start, x end, x points)
    ylim : array_like
        (y start, y end, y points)
    t : float
        time
    lines : float
        scale of number of lines
    filled : bool
        if false, plot contour and if true, plot filled contour

    Returns
    -------
    """
    # assumed t units were 0.1 microseconds
    # now assume t units are 1 days
    time = t
    x = np.linspace(*xlim)
    y = np.linspace(*ylim)
    X, Y = np.meshgrid(x, y)
    phase = 0
    omega = -beta * X / (X**2 + Y**2 + Rd**-2)
    amplitude = (np.exp(-X**2 / n**2 - Y**2 /
                            n**2) * (X**2 + Y**2)) * alpha
    Z = amplitude * np.cos(
            X * x + Y * y - omega * t + phase)
    if filled:
        plt.contourf(X, Y, Z, lines, cmap="coolwarm")
    else:
        plt.contour(X, Y, Z, lines, cmap="coolwarm")
    plt.xlabel('X')
    plt.ylabel('Y')
    cbar = plt.colorbar(pad=0.1)
    cbar.ax.set_ylabel("Stream Function value")
    plt.title(f"t={time} days")
    plt.savefig('test.png')
