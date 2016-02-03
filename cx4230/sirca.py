#!/usr/bin/env python
"""
CX 4230, Spring 2016

sirca.py: This module implements the susceptible-infected-recovered
(SIR) model of infectious disease spread using a cellular automaton.
"""

import numpy as np
import scipy as sp
import scipy.sparse

import matplotlib.pyplot as plt # Core plotting support

# Possible states:
EMPTY = -1
SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2

# Default probability of getting sick, given any sick neighbors
COND_PROB_ILL = 0.5

def create_new_grid (m, n):
    """
    Returns a new (m+2) x (n+2) grid with an empty boundary
    and an interior full of susceptible individuals.
    """
    G = EMPTY * np.ones ((m+2, n+2), dtype=int)
    G[1:-1, 1:-1] = SUSCEPTIBLE
    return G
    

def set_recovery_time (dt):
    """
    Sets the time to recovery from an initial infection. Returns
    the previous recovery time.
    """
    global RECOVERED
    assert dt > 0
    dt_old = RECOVERED - INFECTED
    RECOVERED = INFECTED + dt
    return dt_old

def show_peeps (G, vmin=EMPTY, vmax=RECOVERED, ticks=range (EMPTY, RECOVERED+1)):
    """
    Displays a checkerboard plot of the world.
    """
    plt.pcolor (G, vmin=vmin, vmax=vmax, edgecolor='black')
    plt.colorbar (ticks=ticks)
    plt.axes ().set_aspect ('equal')

def susceptible (G):
    """
    Given a grid, G, returns a grid S whose (i, j) entry
    equals 1 if G[i, j] is susceptible or 0 otherwise.
    """
    return (G == SUSCEPTIBLE).astype (int)

def infected (G):
    """
    Given a grid G, returns a grid I whose (i, j) entry equals 1 if
    G[i, j] is infected or 0 otherwise.
    """
    return ((G >= INFECTED) & (G < RECOVERED)).astype (int)

def recovered (G):
    """
    Given a grid G, returns a grid I whose (i, j) entry equals 1 if
    G[i, j] has recovered or 0 otherwise.
    """
    return (G == RECOVERED).astype (int)

def exposed (G):
    """
    Returns a grid E whose (i, j) entry is 1 if G[i,j] has at least
    1 infected neighbor, or 0 otherwise.
    """
    E = np.zeros (G.shape, dtype=int)
    I = infected (G)
    E[1:-1, 1:-1] = (I[0:-2, 1:-1] | I[1:-1, 0:-2] | I[2:, 1:-1] | I[1:-1, 2:])
    return E.astype (int)

def spreads (G, tau=COND_PROB_ILL):
    """
    Returns a grid G_s whose (i, j) entry is 1 with probability tau
    if G[i,j] is exposed, or 0 otherwise.
    """
    random_draw = np.random.uniform (size=G.shape)
    
    pr_infection = tau * susceptible (G) * exposed (G)
    G_s = (random_draw <= pr_infection)
    return G_s.astype (int)

def step (G, tau=COND_PROB_ILL):
    """
    Simulates one time step and returns a grid
    of the resulting states.
    """
    return G + infected (G) + spreads (G, tau)

def summarize (G, verbose=True):
    n = (G.shape[0]-2) * (G.shape[1]-2)
    n_s = np.sum (susceptible (G))
    n_i = np.sum (infected (G))
    n_r = np.sum (recovered (G))
    if verbose:
        print ("Total beds:", n)
        print ("Susceptible:", n_s)
        print ("Infected:", n_i)
        print ("Recovered:", n_r)
    return (n_s, n_i, n_r, n)

def sim (G_0, max_steps, tau=COND_PROB_ILL, dt=0):
    """
    Starting from a given initial state, `G_0`, this
    function simulates up to `max_steps` time steps of
    the S-I-R cellular automaton. It returns a tuple
    `(t, G_t)` containing the final time step `t` and
    simulation state `G_t`.
    """
    if dt > 0: # Change global recovery time
      dt_save = set_recovery_time (dt)

    print ("@ recovery time:", RECOVERED - INFECTED)
    print ("@ max steps:", max_steps)

    t = 0
    G_t = G_0
    (_, num_infected, _, _) = summarize (G_t, verbose=False)
    while (num_infected > 0) and (t < max_steps):
        t = t + 1
        G_t = step (G_t, tau)
        (_, num_infected, _, _) = summarize (G_t, verbose=False)

    if dt > 0: # Restore global recovery time
      set_recovery_time (dt_save)

    return (t, G_t)

from ipywidgets import interact

def isim (m, n, max_steps=0, dt=0, tau=COND_PROB_ILL, seed=0):
    np.random.seed (seed)

    # Initial state
    G_0 = EMPTY * np.ones ((m+2, n+2), dtype=int)
    G_0[1:-1, 1:-1] = SUSCEPTIBLE
    i_mid = int ((m+2) / 2)
    j_mid = int ((n+2) / 2)
    G_0[i_mid, j_mid] = INFECTED
    
    if not max_steps:
        max_steps = 10 * max (m, n) * max (dt, 1)
    
    (_, G_t) = sim (G_0, max_steps, tau=tau, dt=dt)
    show_peeps (G_t)

if __name__ == "__main__":
    print ("hello, world!")
   
# eof
