import pickle as pkl

import numpy as np
import sympy as sym


def pend(y, t, u):
    """
    System of first order equations for a pendulum system
    The policy commands the torque applied to the joint
    (stable equilibrium point with the pole down at [0,0])
    """
    theta, theta_dot = y

    m = 1.0
    l = 1.0
    b = 0.1
    g = 9.81
    I = 1 / 3 * m * l**2

    dydt = [theta_dot, (u - b * theta_dot - 1 / 2 * m * l * g * np.sin(theta)) / I]
    return dydt


def cartpole(y, t, u):
    """
    System of first order equations for a cart-pole system
    The policy commands the force applied to the cart
    (stable equilibrium point with the pole down at [~,0,0,0])
    """

    x, x_dot, theta, theta_dot = y

    m1 = 0.5
    m2 = 0.5
    l = 0.5
    b = 0.1
    g = 9.81

    den = 4 * (m1 + m2) - 3 * m2 * np.cos(theta) ** 2

    dydt = [
        x_dot,
        (
            2 * m2 * l * theta_dot**2 * np.sin(theta)
            + 3 * m2 * g * np.sin(theta) * np.cos(theta)
            + 4 * u
            - 4 * b * x_dot
        )
        / den,
        theta_dot,
        (
            -3 * m2 * l * theta_dot**2 * np.sin(theta) * np.cos(theta)
            - 6 * (m1 + m2) * g * np.sin(theta)
            - 6 * (u - b * x_dot) * np.cos(theta)
        )
        / (l * den),
    ]
    return dydt
