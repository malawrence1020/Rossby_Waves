"""Implementation of Rossby waves."""
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from math import floor

# scale for amplitude
alpha = 1e11

# scale of sphericity for Rossby waves in Ocean
beta = 2e-11

# scaling for typical wavelength
n = 2e-6

# Rossby deformation radius
Rd = 1e5


def amplitude(wavevector):
    """
    Return amplitude of a Rossby wave.

    Parameters
    ----------
    wavevector : np.ndarray
        wavevector np.array([k, l]) of non-dimensional wavenumbers

    Returns
    -------
    amplitude : float
        amplitude of Rossby wave
    """
    amplitude = np.exp(-wavevector[0]**2 - wavevector[1]**2) * (
        wavevector[0]**2 + wavevector[1]**2)
    return amplitude


def dispersion(wavevector):
    """
    Return frequency from wavevector according to Rossby waves.

    Parameters
    ----------
    wavevector : np.ndarray
        wavevector np.array([k, l]) of wavenumbers

    Returns
    -------
    omega : float
        frequency of Rossby wave
    """
    omega = -beta * wavevector[0] / (wavevector[0]**2 + wavevector[1]**2 + Rd**-2)
    return omega

def plot_amplitude2D(xlim=(-5e-6, 5e-6, 256),
                     ylim=(-5e-6, 5e-6, 256),
                     levels=50,
                     filename="amplitude2D"):
    """
    2D Contour plot of the amplitude of a Rossby wave.

    Parameters
    ----------
    xlim : array_like
        (x start, x end, x points)
    ylim : array_like
        (y start, y end, y points)
    levels : float
        number of contours
    filename : str
        file saved as {filename}.png

    Returns
    -------
    """
    x = np.linspace(*xlim)
    y = np.linspace(*ylim)
    X, Y = np.meshgrid(x, y)
    amp = amplitude([X, Y])
    fig = plt.figure()
    ax = fig.add_subplot()
    plot = ax.contourf(X, Y, amp, levels, cmap="coolwarm")
    cbar = fig.colorbar(plot, pad=0.05)
    cbar.ax.set_ylabel('amplitude, m^2/s')
    ax.set_xlabel('k, 1/m')
    ax.set_ylabel('l, 1/m')
    plt.title('Amplitude in spectral space')
    plt.savefig(f'{filename}.png')

def plot_amplitude3D(xlim=(-5e-6, 5e-6, 256),
                     ylim=(-5e-6, 5e-6, 256),
                     levels=100,
                     filename="amplitude3D"):
    """
    3D Contour plot of the amplitude of a Rossby wave.

    Parameters
    ----------
    xlim : array_like
        (x start, x end, x points)
    ylim : array_like
        (y start, y end, y points)
    levels : float
        number of contours
    filename : str
        file saved as {filename}.png

    Returns
    -------
    """
    x = np.linspace(*xlim)
    y = np.linspace(*ylim)
    X, Y = np.meshgrid(x, y)
    amp = amplitude([X, Y])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.contourf(X, Y, amp, levels, cmap="coolwarm")
    ax.set_xlabel('k, 1/m')
    ax.set_ylabel('l, 1/m')
    ax.set_zlabel('amplitude, m^2/s')
    plt.title('Amplitude in spectral space')
    plt.savefig(f'{filename}.png')


class RossbyWave:
    """
    Class to represent a Rossby wave.

    Attributes
    ----------
    wavevector : array_like
        wavevector (k, l) of wavenumbers
    k : float
        1st component of wavevector
    l : float
        2nd component of wavevector
    phase1 : float
        phase of the wave (solenoidal)
    phase2 : float
        phase of the wave (irrotational)

    Methods
    -------
    __str__(self):
        Return string representation: RossbyWave([k, l], phase).
    __repr__(self):
        Return canonical string representation: RossbyWave([k, l], phase, beta).
    streamfunction(x, t):
        Return streamfunction of Rossby wave.
    plot_streamfunction(self, xlim=(-1, 1, 100), ylim=(-1, 1, 100), t=0, lines=50, filled=True):
        Contour plot of the streamfunction of a Rossby wave.
    animate_streamfunction(self, xlim=(-1, 1, 100), ylim=(-1, 1, 100), tlim=(0, 1000, 101), lines=50, filled=True, filename="streamfunction"):
        Animate the contour plot of the streamfunction of a Rossby wave.
    potentialfunction(self, x, y, t, eps=0.1):
        Return streamfunction of Rossby wave.
    velocity(self, x, y, t, eps=0.1, irrotational=False, solenoidal=False):
        Return velocity of Rossby wave at x at time t.
    plot_velocity(self, xlim=(-1, 1, 100), ylim=(-1, 1, 100), t=0, density=1, eps=0.1, irrotational=False, solenoidal=False):
        Quiverplot the velocity of the Rossby wave.
    plot_stream_velocity(self, xlim=(-np.pi, np.pi, 20, 100), ylim=(-np.pi, np.pi, 20, 100), t=0, eps=0.1, irrotational=False, solenoidal=False, lines=50):
        Contourplot of streamfunction with quiverplot of velocity field of a Rossby wave.
    animate_velocity(self, xlim=(-np.pi, np.pi, 20), ylim=(-np.pi, np.pi, 20), tlim=(0, 3e13, 100), eps=0.1, irrotational=False, solenoidal=False, filename="velocityfield"):
        Animate the quiver plot of the velocity field of a Rossby wave.
    velocity_divergence(self, x, y, t, eps=0.1):
        Return the velocity divergence at (x, y) at time t.
    plot_velocity_divergence(self, xlim=(-np.pi, np.pi, 100), ylim=(-np.pi, np.pi, 100), t=0, eps=0.1, lines=50):
        Contour plot of the streamfunction of a Rossby wave.
    animate_velocity_divergence(self, xlim=(-np.pi, np.pi, 100), ylim=(-np.pi, np.pi, 100), tlim=(0, 3e13, 100), eps=0.1, lines=50, filename="velocity_divergence"):
        Animate the contour plot of the streamfunction of a Rossby wave.
    animate_trajectory(self, x0, xlim=(-np.pi, np.pi, 20, 100), ylim=(-np.pi, np.pi, 20, 100), tlim=(0, 3e13, 100), lines=50, markersize=2, eps=0.1, irrotational=False, solenoidal=False, filename="trajectory"):
        Animate the quiver plot of the velocity field of a Rossby wave.
    """

    def __init__(self, wavevector, phase1=0, phase2=0):
        self.wavevector = list(wavevector)
        self.k = wavevector[0]
        self.l = wavevector[1]
        self.phase1 = phase1
        self.phase2 = phase2
        self.omega = -beta * wavevector[0] / (wavevector[0]**2 +
                                              wavevector[1]**2 + Rd**-2)
        self.amplitude = (np.exp(-wavevector[0]**2 / n**2 - wavevector[1]**2 /
                                n**2) * (wavevector[0]**2 + wavevector[1]**2)) * alpha

    def __str__(self):
        """Return string representation: RossbyWave([k, l], phase1, phase2)."""
        return self.__class__.__name__ + "(" + str(
            self.wavevector) + ", " + str(self.phase1) + ", " + str(self.phase2) + ")"

    def __repr__(self):
        """Return canonical string representation: RossbyWave([k, l], phase1, phase2)."""
        return self.__class__.__name__ + "(" + repr(
            self.wavevector) + ", " + repr(self.phase1) + ", " + repr(self.phase2) + ")"


    def streamfunction(self, x, y, t):
        """
        Return streamfunction of Rossby wave.

        Parameters
        ----------
        x : float
            x position coordinate
        y : float
            y position coordinate
        t : float
            time

        Returns
        -------
        psi : float
            streamfunction at x at time t
        """
        psi = self.amplitude * np.cos(
            self.k * x + self.l * y - self.omega * t + self.phase1)
        return psi

    def plot_streamfunction(self,
                            xlim=(-2000, 2000, 256),
                            ylim=(-2000, 2000, 256),
                            t=0,
                            lines=50,
                            filled=True,
                            filename="streamfunction"):
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
            number of contours
        filled : bool
            if false, plot contour and if true, plot filled contour
        filename : str
            file saved as {filename}.png

        Returns
        -------
        """
        # assumed t units were 0.1 microseconds
        # now assume t units are 1 days
        time = t
        x = np.linspace(*xlim)
        y = np.linspace(*ylim)
        X, Y = np.meshgrid(x, y)
        # Pre-true-scaling modification for plotting purposes
        Z = self.streamfunction(X*1e4, Y*1e4, t*1e6)
        plt.figure()
        if filled:
            plt.contourf(X, Y, Z, lines, cmap="coolwarm")
        else:
            plt.contour(X, Y, Z, lines, cmap="coolwarm")
        plt.xlabel('X')
        plt.ylabel('Y')
        cbar = plt.colorbar(pad=0.1)
        cbar.ax.set_ylabel("Stream Function value")
        plt.title(f"t={time} days")
        plt.savefig(f'{filename}.png')

    def animate_streamfunction(self,
                               xlim=(-2000, 2000, 256),
                               ylim=(-2000, 2000, 256),
                               tlim=(0, 100, 100),
                               lines=50,
                               filled=True,
                               filename="streamfunction"):
        """
        Animate the contour plot of the streamfunction of a Rossby wave.

        Parameters
        ----------
        xlim : array_like
            (x start, x end, x points)
        ylim : array_like
            (y start, y end, y points)
        tlim : array_like
            (time start, time end, time points)
        lines : float
            number of contours
        filled : bool
            if false, plot contour and if true, plot filled contour
        filename : str
            file saved as {filename}.gif

        Returns
        -------
        """
        x = np.linspace(*xlim)
        y = np.linspace(*ylim)
        t = np.linspace(*tlim)
        xx, yy = np.meshgrid(x, y)
        Y, T, X = np.meshgrid(y, t, x)
        fig, ax = plt.subplots(1)
        # Pre-true-scaling for plotting purposes
        stream = self.streamfunction(X*1e4, Y*1e4, T*1e6)

        def init_func():
            plt.cla()

        def update_plot(i):
            plt.cla()
            plt.xlabel('X')
            plt.ylabel('Y')
            if not isinstance(self, RossbyOcean):
                plt.title(
                    f"RossbyWave: k={self.k}, l={self.l}, phase={self.phase1}")
            else:
                plt.title("RossbyOcean")
            if filled:
                plt.contourf(xx, yy, stream[i], lines, cmap="coolwarm")
            else:
                plt.contour(xx, yy, stream[i], lines, cmap="coolwarm")

        anim = FuncAnimation(fig,
                             update_plot,
                             frames=np.arange(0, len(t)),
                             init_func=init_func)

        writergif = PillowWriter(fps=30)
        anim.save(f'{filename}.gif', writer=writergif)

    def potentialfunction(self, x, y, t, eps=0.01):
        """
        Return streamfunction of Rossby wave.

        Parameters
        ----------
        x : float
            x position coordinate
        y : float
            y position coordinate
        t : float
            time
        eps : float
            ratio of stream to potential function

        Returns
        -------
        phi : float
            potentialfunction at x at time t
        """
        phi = eps * self.amplitude * np.cos(
            self.k * x + self.l * y - self.omega * t + self.phase2)
        return phi

    def velocity(self, x, y, t, eps=0.01, irrotational=False, solenoidal=False):
        """
        Return velocity of Rossby wave at x at time t.

        Parameters:
        x : float
            x position coordinate
        y : float
            y position coordinate
        t : float
            time
        eps : float
            ratio of stream to potential function
        irrotational : bool
            curl-free wave
        solenoidal : bool
            divergence-free wave

        Returns
        -------
        v : np.ndarray
            velocity at x at time t
        """
        # v = (-dpsi/dy, dpsi/dx) + (dphi/dx, dphi/dy)
        #   = (1-eps)(-dpsi/dy, dpsi/dx) + eps(dpsi/dx, dpsi/dy)
        # psi = 1/eps * phi = Re[A * exp[i(kx + ly - omega * t + phase)]]

        # dpsi/dx = Re[A * ik * exp[i(kx + ly - omega * t + phase)]]
        #         = A * -k * sin(kx + ly - omega * t + phase)
        # dpsi/dy = Re[A * il * exp[i(kx + ly - omega * t + phase)]]
        #         = A * -l * sin(kx + ly - omega * t + phase)

        v = [0, 0]
        if irrotational and solenoidal:
            raise ValueError(
                "Wave cannot be both irrotational and solenoidal.")
        elif irrotational:
            eps = 1
        elif solenoidal:
            eps = 0
        v[0] = (1 - eps) * amplitude(self.wavevector) * self.l * np.sin(
            self.k * x + self.l * y -
            dispersion(self.wavevector, self.beta) * t +
            self.phase) - eps * amplitude(self.wavevector) * self.k * np.sin(
                self.k * x + self.l * y -
                dispersion(self.wavevector, self.beta) * t + self.phase)
        v[1] = -(1 - eps) * amplitude(self.wavevector) * self.k * np.sin(
            self.k * x + self.l * y -
            dispersion(self.wavevector, self.beta) * t +
            self.phase) - eps * amplitude(self.wavevector) * self.l * np.sin(
                self.k * x + self.l * y -
                dispersion(self.wavevector, self.beta) * t + self.phase)
        return np.array(v)

    def plot_velocity(self,
                      xlim=(-np.pi, np.pi, 20),
                      ylim=(-np.pi, np.pi, 20),
                      t=0,
                      eps=0.1,
                      irrotational=False,
                      solenoidal=False):
        """
        Quiverplot the velocity of the Rossby wave.

        Parameters
        ----------
        xlim : array_like
            (x start, x end, x points)
        ylim : array_like
            (y start, y end, y points)
        t : float
            time
        eps : float
            ratio of stream to potential function
        irrotational : bool
            curl-free wave
        solenoidal : bool
            divergence-free wave

        Returns
        -------
        """
        time = round(t / (10_000_000 * 60 * 60 * 24), 1)
        x = np.linspace(*xlim)
        y = np.linspace(*ylim)
        X, Y = np.meshgrid(x, y)
        u, v = self.velocity(X,
                             Y,
                             t,
                             eps=eps,
                             irrotational=irrotational,
                             solenoidal=solenoidal)
        plt.quiver(X, Y, u, v)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f"t={time} days")

    def animate_velocity(self,
                         xlim=(-np.pi, np.pi, 20),
                         ylim=(-np.pi, np.pi, 20),
                         tlim=(0, 3e13, 100),
                         eps=0.1,
                         irrotational=False,
                         solenoidal=False,
                         filename="velocityfield"):
        """
        Animate the quiver plot of the velocity field of a Rossby wave.

        Parameters
        ----------
        xlim : array_like
            (x start, x end, x points)
        ylim : array_like
            (y start, y end, y points)
        tlim : array_like
            (time start, time end, time points)
        eps : float
            ratio of stream to potential function
        irrotational : bool
            curl-free wave
        solenoidal : bool
            divergence-free wave
        filename : str
            file saved as {filename}.gif

        Returns
        -------
        """
        x = np.linspace(*xlim)
        y = np.linspace(*ylim)
        t = np.linspace(*tlim)
        xx, yy = np.meshgrid(x, y)
        Y, T, X = np.meshgrid(y, t, x)
        fig, ax = plt.subplots(1)
        u, v = self.velocity(X,
                             Y,
                             T,
                             eps=eps,
                             irrotational=irrotational,
                             solenoidal=solenoidal)

        def init_func():
            plt.cla()

        def update_plot(i):
            plt.cla()
            plt.xlabel('X')
            plt.ylabel('Y')
            if not isinstance(self, RossbyOcean):
                plt.title(
                    f"RossbyWave: k={self.k}, l={self.l}, phase={self.phase}")
            else:
                plt.title("RossbyOcean Velocity Field")
            plt.quiver(xx, yy, u[i], v[i])

        anim = FuncAnimation(fig,
                             update_plot,
                             frames=np.arange(0, len(t)),
                             init_func=init_func)

        writergif = PillowWriter(fps=30)
        anim.save(f'{filename}.gif', writer=writergif)

    def plot_stream_velocity(self,
                             xlim=(-np.pi, np.pi, 20, 100),
                             ylim=(-np.pi, np.pi, 20, 100),
                             t=0,
                             eps=0.1,
                             irrotational=False,
                             solenoidal=False,
                             lines=50):
        """
        Contourplot of streamfunction with quiverplot of velocity field of a Rossby wave.

        Parameters
        ----------
        t : float
            time
        xlim : array_like
            (x start, x end, x velocity points, x stream points)
        ylim : array_like
            (y start, y end, y velocity points, y stream points)
        density : float
            density of streamplot arrows
        eps : float
            ratio of stream to potential function
        irrotational : bool
            curl-free wave
        solenoidal : bool
            divergence-free wave
        lines : float
            scale of number of lines

        Returns
        -------
        """
        time = round(t / (10_000_000 * 60 * 60 * 24), 1)
        xlim, ylim = list(xlim), list(ylim)
        x_vel, y_vel = np.linspace(*xlim[0:-1]), np.linspace(*ylim[0:-1])
        xlim.pop(2)
        ylim.pop(2)
        x_str, y_str = np.linspace(*xlim), np.linspace(*ylim)

        # contour plot
        X_str, Y_str = np.meshgrid(x_str, y_str)
        Z_str = self.streamfunction(X_str, Y_str, t)
        cplot = plt.contourf(X_str, Y_str, Z_str, lines, cmap="coolwarm")
        cbar = plt.colorbar(cplot)
        cbar.ax.set_ylabel("Stream Function value")

        # quiver plot
        X_vel, Y_vel = np.meshgrid(x_vel, y_vel)
        u_vel, v_vel = self.velocity(X_vel,
                                     Y_vel,
                                     t,
                                     eps=eps,
                                     irrotational=irrotational,
                                     solenoidal=solenoidal)
        quiv = plt.quiver(X_vel, Y_vel, u_vel, v_vel)

        # labels
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f"t={time} days, eps={eps}")

    def animate_stream_velocity(self,
                                xlim=(-np.pi, np.pi, 20, 100),
                                ylim=(-np.pi, np.pi, 20, 100),
                                tlim=(0, 3e13, 100),
                                lines=50,
                                eps=0.1,
                                irrotational=False,
                                solenoidal=False,
                                filename="streamvelocityfield"):
        """
        Animate the quiver plot of the velocity field of a Rossby wave.

        Parameters
        ----------
        xlim : array_like
            (x start, x end, x points)
        ylim : array_like
            (y start, y end, y points)
        tlim : array_like
            (time start, time end, time points)
        lines : float
            scale of number of lines
        eps : float
            ratio of stream to potential function
        irrotational : bool
            curl-free wave
        solenoidal : bool
            divergence-free wave
        filename : str
            file saved as {filename}.gif

        Returns
        -------
        """
        xlim, ylim = list(xlim), list(ylim)
        x_vel, y_vel = np.linspace(*xlim[0:-1]), np.linspace(*ylim[0:-1])
        xlim.pop(2)
        ylim.pop(2)
        x_str, y_str = np.linspace(*xlim), np.linspace(*ylim)
        t = np.linspace(*tlim)

        xx_vel, yy_vel = np.meshgrid(x_vel, y_vel)
        Y_vel, T_vel, X_vel = np.meshgrid(y_vel, t, x_vel)
        xx_str, yy_str = np.meshgrid(x_str, y_str)
        Y_str, T_str, X_str = np.meshgrid(y_str, t, x_str)
        fig, ax = plt.subplots(1)
        u, v = self.velocity(X_vel,
                             Y_vel,
                             T_vel,
                             eps=eps,
                             irrotational=irrotational,
                             solenoidal=solenoidal)
        stream = self.streamfunction(X_str, Y_str, T_str)

        def init_func():
            plt.cla()

        def update_plot(i):
            plt.cla()
            plt.xlabel('X')
            plt.ylabel('Y')
            if not isinstance(self, RossbyOcean):
                plt.title(
                    f"RossbyWave: k={self.k}, l={self.l}, phase={self.phase}")
            else:
                plt.title("RossbyOcean Streamfunction on Velocity Field")
            plt.contourf(xx_str, yy_str, stream[i], lines, cmap="coolwarm")
            plt.quiver(xx_vel, yy_vel, u[i], v[i])

        anim = FuncAnimation(fig,
                             update_plot,
                             frames=np.arange(0, len(t)),
                             init_func=init_func)

        writergif = PillowWriter(fps=30)
        anim.save(f'{filename}.gif', writer=writergif)

    def velocity_divergence(self, x, y, t, eps=0.1):
        """
        Return the velocity divergence at (x, y) at time t.

        Parameters
        ----------
        x : float
            x coordinate
        y : float
            y coordinate
        t : float
            time
        eps : float
            ratio of stream to potential function

        Returns
        -------
        div : float
            divergence of velocity potential
        """
        # v = (-dpsi/dy, dpsi/dx) + (dphi/dx, dphi/dy)
        # psi = 1/eps * phi = Re[A * exp[i(kx + ly - omega * t + phase)]]
        # div(v) = d^2phi/dx^2 + d^2phi/dy^2
        #        = eps * [Re(d/dx A * ik * exp[i(kx + ly - omega * t + phase)] + d/dy A * il * exp[i(kx + ly - omega * t + phase)])]
        #        = eps * [Re[(A * -k^2 + A * -l^2)exp[i(kx + ly - omega * t + phase)]]
        #        = eps * A * (-l^2-k^2) * cos(kx + ly - omega * t + phase)

        div = eps * amplitude(
            self.wavevector) * (-self.l**2 - self.k**2) * np.cos(
                self.k * x + self.l * y -
                dispersion(self.wavevector, self.beta) * t + self.phase)
        return div

    def plot_velocity_divergence(self,
                                 xlim=(-np.pi, np.pi, 100),
                                 ylim=(-np.pi, np.pi, 100),
                                 t=0,
                                 eps=0.1,
                                 lines=50):
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
        eps : float
            ratio of stream to potential function
        lines : float
            scale of number of lines

        Returns
        -------
        """
        x = np.linspace(*xlim)
        y = np.linspace(*ylim)
        X, Y = np.meshgrid(x, y)
        Z = self.velocity_divergence(X, Y, t, eps)
        plt.contourf(X, Y, Z, lines, cmap="coolwarm")
        plt.xlabel('X')
        plt.ylabel('Y')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel("Divergence value")
        if not isinstance(self, RossbyOcean):
            plt.title(
                f"RossbyWave: k={self.k}, l={self.l}, phase={self.phase}")
        else:
            plt.title("RossbyOcean Divergence")

    def animate_velocity_divergence(self,
                                    xlim=(-np.pi, np.pi, 100),
                                    ylim=(-np.pi, np.pi, 100),
                                    tlim=(0, 3e13, 100),
                                    eps=0.1,
                                    lines=50,
                                    filename="velocity_divergence"):
        """
        Animate the contour plot of the streamfunction of a Rossby wave.

        Parameters
        ----------
        xlim : array_like
            (x start, x end, x points)
        ylim : array_like
            (y start, y end, y points)
        tlim : array_like
            (time start, time end, time points)
        eps : float
            ratio of stream to potential function
        lines : float
            scale of number of lines
        filename : str
            file saved as {filename}.gif

        Returns
        -------
        """
        x = np.linspace(*xlim)
        y = np.linspace(*ylim)
        t = np.linspace(*tlim)
        xx, yy = np.meshgrid(x, y)
        Y, T, X = np.meshgrid(y, t, x)
        fig, ax = plt.subplots(1)
        div = self.velocity_divergence(X, Y, T)

        def init_func():
            plt.cla()

        def update_plot(i):
            plt.cla()
            plt.xlabel('X')
            plt.ylabel('Y')
            if not isinstance(self, RossbyOcean):
                plt.title(
                    f"RossbyWave: k={self.k}, l={self.l}, phase={self.phase}")
            else:
                plt.title("RossbyOcean")
            plt.contourf(xx, yy, div[i], lines, cmap="coolwarm")

        anim = FuncAnimation(fig,
                             update_plot,
                             frames=np.arange(0, len(t)),
                             init_func=init_func)

        writergif = PillowWriter(fps=30)
        anim.save(f'{filename}.gif', writer=writergif)

    def animate_trajectory(self,
                           x0,
                           xlim=(-np.pi, np.pi, 20, 100),
                           ylim=(-np.pi, np.pi, 20, 100),
                           tlim=(0, 10, 100),
                           lines=50,
                           markersize=2,
                           eps=0.1,
                           irrotational=False,
                           solenoidal=False,
                           filename="trajectory"):
        """
        Animate the quiver plot of the velocity field of a Rossby wave.

        Parameters
        ----------
        x0 : np.array
            initial position of particle
        xlim : array_like
            (x start, x end, x points)
        ylim : array_like
            (y start, y end, y points)
        tlim : array_like
            (time start, time end, time points)
        lines : float
            scale of number of lines
        markersize : float
            size of markers plotting trajectory
        eps : float
            ratio of stream to potential function
        irrotational : bool
            curl-free wave
        solenoidal : bool
            divergence-free wave
        filename : str
            file saved as {filename}.gif

        Returns
        -------
        """
        x, y = trajectory(self, x0, tlim[0], tlim[1], tlim[2] * 50, eps,
                          irrotational, solenoidal)

        xlim, ylim = list(xlim), list(ylim)
        x_vel, y_vel = np.linspace(*xlim[0:-1]), np.linspace(*ylim[0:-1])
        xlim.pop(2)
        ylim.pop(2)
        x_str, y_str = np.linspace(*xlim), np.linspace(*ylim)
        t = np.linspace(*tlim)

        xx_vel, yy_vel = np.meshgrid(x_vel, y_vel)
        Y_vel, T_vel, X_vel = np.meshgrid(y_vel, t, x_vel)
        xx_str, yy_str = np.meshgrid(x_str, y_str)
        Y_str, T_str, X_str = np.meshgrid(y_str, t, x_str)
        fig, ax = plt.subplots(1)
        u, v = self.velocity(X_vel,
                             Y_vel,
                             T_vel,
                             eps=eps,
                             irrotational=irrotational,
                             solenoidal=solenoidal)
        stream = self.streamfunction(X_str, Y_str, T_str)

        def init_func():
            plt.cla()

        def update_plot(i):
            x_traj = x[0:i * 50]
            y_traj = y[0:i * 50]
            plt.cla()
            plt.xlabel('X')
            plt.ylabel('Y')
            if not isinstance(self, RossbyOcean):
                plt.title(
                    f"RossbyWave: k={self.k}, l={self.l}, phase={self.phase}")
            else:
                plt.title("RossbyOcean Streamfunction on Velocity Field")
            plt.contourf(xx_str, yy_str, stream[i], lines, cmap="coolwarm")
            plt.quiver(xx_vel, yy_vel, u[i], v[i])
            plt.plot(x_traj, y_traj, 'o', ms=markersize, color="magenta")

        anim = FuncAnimation(fig,
                             update_plot,
                             frames=np.arange(0, len(t)),
                             init_func=init_func)

        writergif = PillowWriter(fps=30)
        anim.save(f'{filename}.gif', writer=writergif)


class RossbyOcean(RossbyWave):
    """Collection of Rossby waves.

    Attributes
    ----------
    waves : list
        list of RossbyWaves in RossbyOcean
    wavevectors : np.ndarray
        array of wavevectors of RossbyWaves
    phases : np.ndarray
        array of phases
    k : np.ndarray
        array of 1st wavevector components
    l : np.ndarray
        array of 2nd wavevector components
    beta : float
        scale of sphericity

    Methods
    -------
    __str__(self):
        Return string representation: RossbyOcean(RossbyWave(wavevector, phase), ...).
    __repr__(self):
        Return canonical string representation: RossbyOcean([RossbyWave(wavevector, phase, beta), ...], beta).
    streamfunction(x, t):
        Return streamfunction of Rossby wave.
    velocity(self, x, y, t, eps=0.1, irrotational=False, solenoidal=False):
        Return velocity of Rossby wave at x at time t.
    velocity_divergence(self, x, y, t, eps=0.1):
        Return the velocity divergence at (x, y) at time t.
    add_wave(wave):
        Add a RossbyWave to the RossbyOcean.
    add_random_wave(self, xlim=(-5, 5), ylim=(-5, 5), plim=(0, 2 * np.pi)):
        Add a RossbyWave to the Rossbyocean with random wavevector.
    add_random_waves(self, n, xlim=(-5, 5), ylim=(-5, 5), plim=(0, 2 * np.pi)):
        Add n random wavevectors.
    normal_wavevectors(xlim=(-5, 5, 10), ylim=(-5, 5, 10)):
        Add RossbyWaves with wavevectors (k, l) in a grid.
    remove_wave(self, index):
        Remove the RossbyWave at index in the RossbyOcean.
    """

    def __init__(self, rossby_waves, beta=beta):
        self.waves = rossby_waves
        self.wavevectors = np.array([wave.wavevector for wave in rossby_waves])
        self.phases = np.array([wave.phase for wave in rossby_waves])
        self.k = self.wavevectors[:, 0]
        self.l = self.wavevectors[:, 1]
        self.beta = beta

    def __str__(self):
        """
        Return string representation:
        RossbyOcean(RossbyWave(wavevector, phase), ...).
        """
        waves = ""
        for wave in self.waves:
            waves += str(wave) + ", "
        return self.__class__.__name__ + "(" + waves[0:-2] + ")"

    def __repr__(self):
        """
        Return canonical string representation:
        RossbyOcean([RossbyWave(wavevector, phase, beta), ...], beta).
        """
        return self.__class__.__name__ + "(" + str(self.waves) + ", " + str(
            self.beta) + ")"

    def streamfunction(self, x, y, t):
        """
        Return streamfunction of Rossby ocean.

        Parameters
        ----------
        x : float
            x position coordinate
        y : float
            y position coordinate
        t : float
            time

        Returns
        -------
        psi : float
            streamfunction at x at time t
        """
        psi = 0
        for wave in self.waves:
            psi += wave.streamfunction(x, y, t)
        return psi

    def velocity(self, x, y, t, eps=0.1, irrotational=False, solenoidal=False):
        """
        Return velocity of Rossby wave at x at time t.

        Parameters:
        x : float
            x position coordinate
        y : float
            y position coordinate
        t : float
            time
        eps : float
            ratio of stream to potential function
        irrotational : bool
            curl-free wave
        solenoidal : bool
            divergence-free wave

        Returns
        -------
        v : np.ndarray
            velocity at x at time t
        """
        # v = (-dpsi/dy, dpsi/dx) + (dphi/dx, dphi/dy)
        # eps*phi = psi = A * exp(k.x - omega * t)
        v = [0, 0]
        if irrotational and solenoidal:
            raise ValueError(
                "Wave cannot be both irrotational and solenoidal.")
        for wave in self.waves:
            ou, ov = wave.velocity(x, y, t, eps, irrotational, solenoidal)
            v[0] += ou
            v[1] += ov
        return np.array(v)

    def velocity_divergence(self, x, y, t, eps=0.1):
        """
        Return the velocity divergence at (x, y) at time t.

        Parameters
        ----------
        x : float
            x coordinate
        y : float
            y coordinate
        t : float
            time
        eps : float
            ratio of stream to potential function

        Returns
        -------
        div : float
            divergence of velocity potential
        """
        div = 0
        for wave in self.waves:
            div += wave.velocity_divergence(x, y, t, eps)
        return div

    def add_wave(self, wave):
        """
        Add a RossbyWave to the RossbyOcean.

        Parameters
        ----------
        wave : RossbyWave
            RossbyWave to be added

        Returns
        -------
        """
        self.waves.append(wave)
        self.wavevectors = np.array([wave.wavevector for wave in self.waves])
        self.phases = np.array([wave.phase for wave in self.waves])
        self.k = self.wavevectors[:, 0]
        self.l = self.wavevectors[:, 1]
        self = RossbyOcean(self.waves, beta=self.beta)

    def remove_wave(self, index):
        """
        Remove the RossbyWave at index in the RossbyOcean.

        Parameters
        ----------
        index : int
            index of RossbyWave to be removed

        Returns
        -------
        """
        self.waves.pop(index)
        self.wavevectors = np.array([wave.wavevector for wave in self.waves])
        self.phases = np.array([wave.phase for wave in self.waves])
        self.k = self.wavevectors[:, 0]
        self.l = self.wavevectors[:, 1]
        self = RossbyOcean(self.waves, beta=self.beta)

    def add_random_wave(self,
                        xlim=(-5, 5),
                        ylim=(-5, 5),
                        plim=(0, 2 * np.pi),
                        beta=beta):
        """
        Add a RossbyWave to the Rossbyocean with random wavevector.

        Parameters
        ----------
        xlim : array_like
            (l, u) lower and upperbounds of k wavevector component
        ylim : array_like
            (l, u) lower and upperbounds of l wavevector component
        plim : array_like
            (l, u) lower and upperbounds of phase
        beta : float
            scale of sphericity

        Returns
        -------
        """
        k = 0
        l = 0
        while k == 0 and l == 0:
            k = (xlim[1] - xlim[0]) * np.random.random() + xlim[0]
            l = (ylim[1] - ylim[0]) * np.random.random() + ylim[0]
        phase = (plim[1] - plim[0]) * np.random.random() + plim[0]
        self.add_wave(RossbyWave([k, l], phase, beta=beta))

    def add_random_waves(self,
                         n,
                         beta=beta,
                         xlim=(-5, 5),
                         ylim=(-5, 5),
                         plim=(0, 2 * np.pi)):
        """
        Add n random wavevectors.

        Parameters
        ----------
        n : int
            number of wavevectors to add
        beta : float
            scale of sphericity
        xlim : array_like
            (l, u) lower and upperbounds of k wavevector component
        ylim : array_like
            (l, u) lower and upperbounds of l wavevector component
        plim : array_like
            (l, u) lower and upperbounds of phase

        Returns
        -------
        """
        for i in range(n):
            self.add_random_wave(xlim, ylim, plim, beta)

    def add_grid_waves(self,
                       xlim=(-5, 5, 11),
                       ylim=(-5, 5, 11),
                       phase=True,
                       beta=2e-11):
        """
        Add RossbyWaves with wavevectors (k, l) in a grid.

        Parameters
        ----------
        xlim : array_like
            (x start, x end, x points)
        ylim : array_like
            (y start, y end, y points)
        phase : bool
            if True, add random phases in (0, 2*np.pi), else phase=0

        Returns
        -------
        """
        np.random.seed(5)
        x, y = np.linspace(*xlim), np.linspace(*ylim)
        for i in x:
            for j in y:
                if i == 0 and j == 0:
                    continue
                if phase:
                    p = 2 * np.pi * np.random.random()
                else:
                    p = 0
                self.add_wave(RossbyWave((i, j), p, beta))


def trajectory(ro, x0, t0, t, h, eps=0.1, xrange=np.pi, yrange=np.pi):
    """
        Return lists of x-coords and y-coords of trajectory of particles with
        initial conditions in the velocity field of the RossbyOcean.

        Parameters
        ----------
        ro : RossbyOcean

        x0 : nx2 np.array
            initial positions of n particles
        t0 : float
            starting time
        t : float
            ending time
        h : float
            step size
        eps : float
            ratio of stream to potential function
        xrange : float
            length of x-axis, e.g. if xrange = 3, the x-axis goes from -3 to 3
        yrange : float
            length of y-axis, e.g. if yrange = 3, the y-axis goes from -3 to 3

        Returns
        -------
        x_coords : list
            list of lists of x-coordinates of particle trajectories, e.g. x[i] gives x-coordinates of the i-1th particle's trajectory
        y_coords : list
            list of lists of y-coordinates of particle trajectories, e.g. y[i] gives y-coordinates of the i-1th particle's trajectory
        """
    n = (t - t0) / h
    x = x0
    t = t0
    i = 0
    num_points = x0.shape[0]
    x_coords = [[] for x in range(num_points)]
    y_coords = [[] for x in range(num_points)]
    j = 0
    for list in x_coords:
        list.append(x[j, 0])
        j += 1
    j = 0
    for list in y_coords:
        list.append(x[j, 1])
        j += 1
    while i < n:
        k_1 = vel(ro, x, t, eps)
        k_2 = vel(ro, x + h * k_1 / 2, t + h / 2, eps)
        k_3 = vel(ro, x + h * k_2 / 2, t + h / 2, eps)
        k_4 = vel(ro, x + h * k_3, t + h, eps)
        x = x + h / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        i += 1
        t += h
        x[:, 0] += xrange
        x[:, 0] = (x[:, 0] % (2 * xrange)) - xrange
        x[:, 1] += yrange
        x[:, 1] = (x[:, 1] % (2 * yrange)) - yrange
        j = 0
        for list in x_coords:
            list.append(x[j, 0])
            j += 1
        j = 0
        for list in y_coords:
            list.append(x[j, 1])
            j += 1

    return x_coords, y_coords


def grid(n, rho=0, xrange=np.pi, yrange=np.pi):
    """
        Return array of nxn evenly spaced points.

        Parameters
        ----------
        n : int

        xrange : float
            length of half of x-axis, e.g. xrange = 1 means the x-axis goes from -1 to 1
        yrange : float
            length of half of y-axis

        Returns
        -------
        np.array
        """
    a, b = -xrange, xrange
    h = (b - a) / n
    x = [a + h * (1 / 2 + i) for i in range(n)]
    a, b = -yrange, yrange
    h = (b - a) / n
    y = [a + h * (1 / 2 + i) for i in range(n)]
    v = []
    for i in x:
        for j in y:
            if rho == 0:
                v.append([i, j])
            else:
                v.append([i, j, rho])
    return np.array(v)


def vel(ro, x, t, eps=0.1):
    """
        Return array of velocity at x and time t of RossbyOcean.

        Parameters
        ----------
        ro : RossbyOcean

        x : nx2 np.array
            n points at which velocity will be evaluated at
        t : float
            time at which velocity will be evaluated at
        eps : float
            ratio of stream to potential function
        Returns
        -------
        v : nx2 np.array
            velocity of the n points at t
        """
    L = np.shape(x)[0]
    v = np.zeros((L, 2))
    for r in ro.waves:
        s = np.sin(r.k * x[:, 0] + r.l * x[:, 1] - r.omega * t + r.phase)
        dpsidy = -r.amplitude * r.l * s
        dpsidx = -r.amplitude * r.k * s
        if eps != 1:
            v[:, 0] += (1 - eps) * -dpsidy
            v[:, 1] += (1 - eps) * dpsidx
        if eps != 0:
            v[:, 0] += eps * dpsidx
            v[:, 1] += eps * dpsidy
    return v


def vel_den(ro, x, t, eps=0.1):
    """
        Return array of velocity at x and time t of RossbyOcean.

        Parameters
        ----------
        ro : RossbyOcean

        x : nx3 np.array

        t : float
            time at which velocity will be evaluated at
        eps : float
            ratio of stream to potential function
        Returns
        -------
        v : nx2 np.array
            velocity of the n points at t
        """
    L = np.shape(x)[0]
    v = np.zeros((L, 3))
    for r in ro.waves:
        s = np.sin(r.k * x[:, 0] + r.l * x[:, 1] - r.omega * t + r.phase)
        c = np.cos(r.k * x[:, 0] + r.l * x[:, 1] - r.omega * t + r.phase)
        dpsidy = -r.amplitude * r.l * s
        dpsidx = -r.amplitude * r.k * s
        d2psidx2 = -r.amplitude * r.k**2 * c
        d2psidy2 = -r.amplitude * r.l**2 * c
        if eps != 1:
            v[:, 0] += (1 - eps) * -dpsidy
            v[:, 1] += (1 - eps) * dpsidx
        if eps != 0:
            v[:, 0] += eps * dpsidx
            v[:, 1] += eps * dpsidy
        v[:, 2] += -eps * (d2psidx2 + d2psidy2)
    v[:, 2] = np.multiply(x[:, 2], v[:, 2])
    return v


def trajectory_den(ro, x0, t0, t, h, eps=0.1, xrange=np.pi, yrange=np.pi):
    n = (t - t0) / h
    x = x0
    t = t0
    i = 0
    num_points = x0.shape[0]
    x_coords = [[] for x in range(num_points)]
    y_coords = [[] for x in range(num_points)]
    rho = [[] for x in range(num_points)]
    j = 0
    for list in x_coords:
        list.append(x[j, 0])
        j += 1
    j = 0
    for list in y_coords:
        list.append(x[j, 1])
        j += 1
    j = 0
    for list in rho:
        list.append(x[j, 2])
        j += 1
    while i < n:
        k_1 = vel_den(ro, x, t, eps)
        k_2 = vel_den(ro, x + h * k_1 / 2, t + h / 2, eps)
        k_3 = vel_den(ro, x + h * k_2 / 2, t + h / 2, eps)
        k_4 = vel_den(ro, x + h * k_3, t + h, eps)
        x = x + h / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        i += 1
        t += h
        x[:, 0] += xrange
        x[:, 0] = (x[:, 0] % (2 * xrange)) - xrange
        x[:, 1] += yrange
        x[:, 1] = (x[:, 1] % (2 * yrange)) - yrange
        j = 0
        for list in x_coords:
            list.append(x[j, 0])
            j += 1
        j = 0
        for list in y_coords:
            list.append(x[j, 1])
            j += 1
        j = 0
        for list in rho:
            list.append(x[j, 2])
            j += 1

    return x_coords, y_coords, rho


def vel_autocor(ro, x, t0, t1, e=0.1):
    xcoords, ycoords = trajectory(ro,x,t0,t1, 2.5e9, eps=e)
    initial = np.vstack((np.array(xcoords)[:,0],np.array(ycoords)[:,0])).T
    final = np.vstack((np.array(xcoords)[:,-1],np.array(ycoords)[:,-1])).T
    u = vel(ro, initial, t0, eps=e)
    v = vel(ro, final, t1, eps=e)
    a = 0
    b = 0
    for i in range(np.shape(v)[0]):
        a += (np.dot(u[i], v[i]))
        b += np.dot(v[i], v[i])
    return (a / b) * np.shape(v)[0]


def dxt(ro, x, t, e=0.1):
    p1, p2 = trajectory(ro, x, 0, t, (t / 2e3), eps=e)
    p3, p4 = np.array(p1), np.array(p2)
    a = 0
    for i in range(len(p1)):
        a += (p3[i, 2000] - p3[i, 00])**2
    return a / len(p1)


def dyt(ro, x, t, e=0.1):
    p1, p2 = trajectory(ro, x, 0, t, (t / 2e3), eps=e)
    p3, p4 = np.array(p1), np.array(p2)
    a = 0
    for i in range(len(p1)):
        a += (p4[i, 2000] - p4[i, 0])**2
    return a / len(p2)


def dt2(data):
    x = np.array(data)
    a = []

    for j in range(len(data[1])):

        y = x[:,j] - x[:,0]
        b = np.dot(y,y)
        a.append(b/10000)
    return a


def traj2(ro, x0, t0, t, h, eps=0.1, xrange=np.pi, yrange=np.pi):
    """
        Return lists of x-coords and y-coords of trajectory of particles with
        initial conditions in the velocity field of the RossbyOcean.

        Parameters
        ----------
        ro : RossbyOcean

        x0 : nx2 np.array
            initial positions of n particles
        t0 : float
            starting time
        t : float
            ending time
        h : float
            step size
        eps : float
            ratio of stream to potential function
        xrange : float
            length of x-axis, e.g. if xrange = 3, the x-axis goes from -3 to 3
        yrange : float
            length of y-axis, e.g. if yrange = 3, the y-axis goes from -3 to 3

        Returns
        -------
        x_coords : list
            list of lists of x-coordinates of particle trajectories, e.g. x[i] gives x-coordinates of the i-1th particle's trajectory
        y_coords : list
            list of lists of y-coordinates of particle trajectories, e.g. y[i] gives y-coordinates of the i-1th particle's trajectory
        """
    n = (t - t0) / h
    x = x0
    t = t0
    i = 0
    num_points = x0.shape[0]
    x_coords = [[] for x in range(num_points)]
    y_coords = [[] for x in range(num_points)]
    j = 0
    for list in x_coords:
        list.append(x[j, 0])
        j += 1
    j = 0
    for list in y_coords:
        list.append(x[j, 1])
        j += 1
    while i < n:
        k_1 = vel(ro, x, t, eps)
        k_2 = vel(ro, x + h * k_1 / 2, t + h / 2, eps)
        k_3 = vel(ro, x + h * k_2 / 2, t + h / 2, eps)
        k_4 = vel(ro, x + h * k_3, t + h, eps)
        x = x + h / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        i += 1
        t += h
        j = 0
        for list in x_coords:
            list.append(x[j, 0])
            j += 1
        j = 0
        for list in y_coords:
            list.append(x[j, 1])
            j += 1

    return x_coords, y_coords
