#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pylab as p
from numpy import *
from scipy import integrate


class Gate(object):
  """
  Gate class
  """

  def __init__(self, steady_state_params, time_constant_params):
    """
    Gate constructor

    arguments:
       steady_state_params - tuple containing (Vhalf, k)
      time_constant_params - tuple containing (Vmax, sigma, C_amp, C_base)

    note:
      for transient current, pass None as time_constant_params
    """
    self.Vh, self.k = steady_state_params
    self.transient = time_constant_params is None
    if not self.transient:
      self.Vmax, self.sigma, self.C_amp, self.C_base = time_constant_params

  def m_inf(self, Vm):
    """
    steady-state activation/inactivation curve
    approximated by the Boltzmann function

    parameters:
      Vh - a potential that satisfies m_inf(Vh) = 1/2
       k - slope factor (positive for activation, negative for inactivation)
      Vm - membrane potential
    """
    return 1/(1+exp((self.Vh - Vm) / self.k))

  def tau(self, Vm):
    """
    voltage-sensitive time constant
    approximated by the Gaussian function

    arguments:
        Vmax - a potential at which maximum value is achieved
       sigma - standard deviation
       C_amp - amplifier
      C_base - minimum value
          Vm - membrane potential
    """
    if self.transient:
      raise Exception
    else:
      return self.C_base + self.C_amp * exp(-(self.Vmax-Vm)**2 / self.sigma**2)

  def dm_dt(self, Vm, m):
    return (self.m_inf(Vm) - m) / self.tau(Vm)

class Channel(object):

  def __init__(self, g, E, act_gate, inact_gate):
    """
    
    arguments:
               g - maximum conductance
               E - reversal potential
        act_gate - a tuple containing (activation gate object, num. of gates)
      inact_gate - a tuple containing (inactivation gate object, num. of gates)
    """
    self.g = g
    self.E = E

    self.act_gate, self.a = act_gate if act_gate else (None, 0)
    self.inact_gate, self.b = inact_gate if inact_gate else (None, 0)

  def initial_params(self, Vm):
    m, h = 1, 1
    if self.act_gate:
      m = self.act_gate.m_inf(Vm)
    if self.inact_gate:
      h = self.inact_gate.m_inf(Vm)
    return [m, h]

  def I(self, Vm, params):
    """
    macroscopic current

    arguments:
       g - maximum conductance
       m - prob. of activation gate to be open
       h - prob. of inactivation gate to be open
       a - the num. of activation gates per channel
       b - the num. of inactivation gates per channel
      Vm - membrane potential
       E - reversal potential
  """
    return self.g * params[0]**self.a * params[1]**self.b * (Vm - self.E)

  def dX_dt(self, Vm, params):
    return array([self.act_gate.dm_dt(Vm, params[0]) if self.a > 0 else 0,
                  self.inact_gate.dm_dt(Vm, params[1]) if self.b > 0 else 0])


class Cell(object):
  def __init__(self, channels, current, Vm, C):
    self.channels = channels
    self.current = current
    self.Vm = Vm
    self.C = C

    arr = array([self.Vm])
    for channel in self.channels:
      arr = append(arr, channel.initial_params(self.Vm))
    self.X_0 = arr

  def dX_dt(self, X, t=0):
    # membrane potential
    Vm = X[0]
    net_current = self.current(t)
    for i, channel in enumerate(self.channels):
      net_current -= channel.I(Vm, X[1+2*i:3+2*i])
    res = array([net_current / self.C])

    # channel gates
    for i, channel in enumerate(self.channels):
      res = append(res, channel.dX_dt(Vm, X[1+2*i:3+2*i]))

    return res

  def go(self, to_time):
    t = linspace(0, to_time, 1000)
    print(self.X_0)
    X, infodict = integrate.odeint(self.dX_dt, self.X_0, t, full_output=True)
    f = p.figure()
    p.plot(t, (X.T)[0], 'r-', label='Vm')
    p.grid()
    p.xlabel('time')
    p.ylabel('Vm')
    p.title('Hodkin-Huxley model')
    f.savefig('hoge.png')

I_Na_t = Channel(120, 55, # g, E
                 (Gate((-40, 15), (-38, 30, 0.46, 0.04)), 3), # activation
                 (Gate((-62, -7), (-67, 20, 7.4,  1.2)), 1))  # inactivation
I_K = Channel(36, -77,
              (Gate((-53, 15), (-79, 50, 4.7, 1.1)), 4), # activation
              None) # delayed rectifier has no inactivation gates
I_leak = Channel(0.3, -55, None, None)

def cur(t):
  """stim. current"""
  return 600 if 1 < t < 2 else 0


def main(progname, args):
  # current clamp
  giant_axon = Cell((I_Na_t, I_K, I_leak), cur, -50, 1)
  giant_axon.go(10)
  return 0

if __name__ == '__main__':
  sys.exit(main(sys.argv[0], sys.argv[1:]))

