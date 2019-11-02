# Copyright [2019] [Kevin Abraham]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Contains the simulation required for finding the next state. 
TODO : Structure this file into different class (OOP)
Under development (code for experimentation.) RL will be experimented here
"""

#%%
import numpy as np
import qutip as q

# %%
# Divide the total time frame of operation into n segments of equal duration. 
# # There are two possibilities of going about this. 
# #   1. Use a heuristic to find the time taken for the whole operation. Assume that 
# #      this time is our worst case time. ie our algorithm will defnitely improve the 
#        time. So divide the time required by heuristic into n segments

#     2. Take the segment_duration as a tunable parameter. Start with a random value. And 
#        find the number of time steps such that, after n steps the state obtained will have 
#        the maximum fidelity

#     For computational reasons we go by method 2 [it goes better with RL approach]
# TODO : CUrrently the implemention is limited to single qubits. Make it generic using numpy array for n-qubits

segment_duration =  .25   # Ensure that this value is in comparison with the system parameters
                          # that affect the spin precession

def set_segment_duration(val):
    segment_duration = val

def get_segment_duration():
    return segment_duration

def get_current_segment():
    #TODO : Ensure that all controls has same length
    return len(control_detuning)


# Create control list which can be appended after each segment duration. 

# detuning, Omega_x and Omega_y are the control parameters (or the actions) which can be varied
# for each time segment. Based on these controls the system will evolve into a new state. We start
# with controlling only the amplitude  and not the wave type (ie its always square wave)
# TODO : Have a functionality for changing the wave type

control_detuning = []
control_omega_x = []
control_omega_y =[]


#returns the cotrol at time t chosen by the algorithm

# check whether a segment is valid. True if the control 
# has been determined for the segment
def is_valid_segment(segment):
    return segment <= get_current_segment()

def get_detuning_control(t):
    segment = int(t/get_segment_duration())
    if(is_valid_segment(segment)):
        return control_detuning[segment]
    else:
        return 0

def get_omega_x_control(t):
    segment = int(t/get_segment_duration())
    if(is_valid_segment(segment)):
        return control_omega_x[segment]
    else:
        return 0

def get_omega_y_control(t):
    segment = int(t/get_segment_duration())
    if(is_valid_segment(segment)):
        return control_omega_y[segment]
    else:
        return 0

#update the control list with the latest 
def append_controls(detuning, omega_x, omega_y):
    #TODO : Ensure all the control list has same lenght
    control_detuning.append(detuning)
    control_omega_x.append(omega_x)
    control_omega_y.append(omega_y)

append_controls(1,1,1)
#%%
# Hamiltonian 
h =   1   # 2*np.pi*6.6*1e-36  # planks constatn 
w0 = .5   # static magnetic field in T 
w =  .4   # driving frequency

detuning =  w0 - w 
def detuning(t, args):
    return h/2*(w0 - get_detuning_control(t))
def Omega_x(t, args):
    return get_omega_x_control(t)*h/2     # Rabi rate  along x 

def Omega_y(t, args):
    return get_omega_y_control(t)*h/2     # Rabi rate  along y

def noise(t, args):  
    #TODO :  make this staochastic using random variable for each segment
    return np.sin(10*t)*10*h/2

H_0 = q.sigmaz()
H_noise = q.sigmam()
H_x = q.sigmax()
H_y = q.sigmay() 

#H  = h/2*(detuning*q.sigmaz() + noise*q.sigmaz() + Omega_x*q.sigmax() + Omega_y*q.sigmay())
H = [[H_0, detuning], [H_noise, noise], [H_x, Omega_x], [H_y, Omega_y]]

#%% Simulation 
psi0 = q.basis(2,0) 
TIME_UPPER_BOUND  = 2
segments = TIME_UPPER_BOUND/get_segment_duration()
t = np.arange(0,.5,.1)
c_ops = []
append_controls(.4,5,5)
output = q.mesolve(H, psi0, t, c_ops)

# %% Visualize the evolution 
sphere = q.Bloch()
for states in output.states:
    sphere.clear()
    sphere.add_states(states)
    sphere.save(dirc = 'temp')

# %%

# Algorithm 

# # 1. Define the control states. Each control (action) can take a discrete set of values.
#      The combination of the cotrols (actions) Will take the system to a new state 
#   2. Find the state which has the lowest Cost + Cost to Go
#                 a. Cost = square sum distance between previous pulse amplitude and the current pulse ampli.
#                             (Omega_x_next - Omega_x_current)**2 + (Omega_y_next - Omega_y_current)**2  + (detuning_next - detuning_next)**2
                
#                 b. Cost to go : 1 / Fidelity between next state and the final desired state (Heuristic : Improve using GRAPE if possible)
#                        = (1/F) -1  
#      The inverse of the fidelity ranges from 1 to infinity. This ensures that cost to go is given priority over cost, thus ensuring elimination
#      of random changes. When the cost to go approaches zero priority will be given to cost for fine tuning the pulse amplitudes 

OMEGA_X_UPPER_BOUND = 10
OMEGA_X_CHANGE = 1

OMEGA_Y_UPPER_BOUND = 10
OMEGA_Y_CHANGE = 1

DETUNING_CONTROL_UPPER_BOUND = w0 
DETUNING_CONTROL_CHANGE = 0.1
# Note that when the upper bount for detuning control is w0 (resonance frequency)
# detuning will be zero. See the defnition of detuning(t, args)

## RL actions 
omega_x_actions = np.arange(0, OMEGA_X_UPPER_BOUND, OMEGA_X_CHANGE)
omega_y_actions = np.arange(0, OMEGA_Y_UPPER_BOUND, OMEGA_Y_CHANGE)
detuning_actions = np.arange(0, DETUNING_CONTROL_UPPER_BOUND , DETUNING_CONTROL_CHANGE)

# Since the cost is independent of the state we can precalculate them and arrange them in a priority queue 
# with corresponding cost of the transition in a map for easier computation

##Use a numpy mgrid to store the above actions for compulation of eucledian distance. 

from queue import PriorityQueue


