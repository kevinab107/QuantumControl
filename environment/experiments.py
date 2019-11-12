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
TODO [OOP] : Structure this file into different class (OOP)
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
# TODO [SCALE]: CUrrently the implemention is limited to single qubits. Make it generic using numpy array for n-qubits

segment_duration =  .25   # Ensure that this value is in comparison with the system parameters
                          # that affect the spin precession

def set_segment_duration(val):
    segment_duration = val

def get_segment_duration():
    return segment_duration

def get_current_segment():
    #TODO [STANDARDIZE] : Ensure that all controls has same length
    return len(control_detuning)


# Create control list which can be appended after each segment duration. 

# detuning, Omega_x and Omega_y are the control parameters (or the actions) which can be varied
# for each time segment. Based on these controls the system will evolve into a new state. We start
# with controlling only the amplitude  and not the wave type (ie its always square wave)
# TODO [SCALE] : Have a functionality for changing the wave type

control_detuning = []
control_omega_x = []
control_omega_y =[]


#returns the cotrol at time t chosen by the algorithm

# check whether a segment is valid. True if the control 
# has been determined for the segment
# def is_valid_segment(segment):
#     return segment <= get_current_segment()

# def get_detuning_control(t):
#     segment = int(t/get_segment_duration())
#     if(is_valid_segment(segment)):
#         return control_detuning[segment]
#     else:
#         return 0

# def get_omega_x_control(t):
#     segment = int(t/get_segment_duration())
#     if(is_valid_segment(segment)):
#         return control_omega_x[segment]
#     else:
#         return 0

# def get_omega_y_control(t):
#     segment = int(t/get_segment_duration())
#     if(is_valid_segment(segment)):
#         return control_omega_y[segment]
#     else:
#         return 0

#update the control list with the latest 
def append_controls(detuning, omega_x, omega_y):
    #TODO [STANDARDIZE] : Ensure all the control list has same lenght
    control_detuning.append(detuning)
    control_omega_x.append(omega_x)
    control_omega_y.append(omega_y)

def clear_controls():
    control_detuning.clear()
    control_omega_x.clear()
    control_omega_y.clear()

#%%
# Hamiltonian 
h =   1   # 2*np.pi*6.6*1e-36  # planks constatn 
w0 = .5   # static magnetic field in T 
w =  .4   # driving frequency

detuning =  w0 - w 
# def detuning(t, args):
#     return h/2*(w0 - get_detuning_control(t))
# def Omega_x(t, args):
#     return get_omega_x_control(t)*h/2     # Rabi rate  along x 

# def Omega_y(t, args):
#     return get_omega_y_control(t)*h/2     # Rabi rate  along y

# def noise(t, args):  
#     #TODO [ALGORITHM]:  make this staochastic using random variable for each segment
#     return np.sin(10*t)*10*h/2

# H_0 = q.sigmaz()
# H_noise = q.sigmam()
# H_x = q.sigmax()
# H_y = q.sigmay() 

# #H  = h/2*(detuning*q.sigmaz() + noise*q.sigmaz() + Omega_x*q.sigmax() + Omega_y*q.sigmay())
# H = [[H_0, detuning], [H_noise, noise], [H_x, Omega_x], [H_y, Omega_y]]

# #%% Simulation 

# psi0 = q.basis(2,0) 
# TIME_UPPER_BOUND  = 2
# segments = TIME_UPPER_BOUND/get_segment_duration()
# t = np.arange(0,.5,.1)
# c_ops = []
# append_controls(.4,5,5)
# output = q.mesolve(H, psi0, t, c_ops)

# # %% Visualize the evolution 
# sphere = q.Bloch()
# for states in output.states:
#     sphere.clear()
#     sphere.add_states(states)
#     sphere.save(dirc = 'temp')

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

#TODO [STANDARDIZE] : Ensure all the values are float for consistancy. numpy mgrid stores the value aS 0. while a float is 0.0
# This may create inconsitancy while converting to string. This is a minor issue as the coordinates will be always 
# used from the coordinate matrix. (Since we are restricting any other control coordinate)
OMEGA_X_UPPER_BOUND = 10.0
OMEGA_X_CHANGE = 1.0

OMEGA_Y_UPPER_BOUND = 10.0
OMEGA_Y_CHANGE = 1.0

DETUNING_CONTROL_UPPER_BOUND = 0.5
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
#%%
from queue import PriorityQueue
from sklearn.metrics.pairwise import euclidean_distances
#TODO [OOP] : Encapsulate the control range into a class (ControlRange) for easy initialization and variations 



#Create an array of coordinates corresponding to all possible combination of 
#control actions 
# TODO: Make this a method of ControlRange

def get_control_cordinates_3d():
    x,y,z = np.mgrid[0. : OMEGA_X_UPPER_BOUND : OMEGA_X_CHANGE,
                    0. : OMEGA_Y_UPPER_BOUND : OMEGA_Y_CHANGE,
                    0. : DETUNING_CONTROL_UPPER_BOUND : DETUNING_CONTROL_CHANGE]

    control_coordinates = np.empty(x.shape + (3,), dtype = float)
    control_coordinates[:,:,:,0] = x                   
    control_coordinates[:,:,:,1] = y
    control_coordinates[:,:,:,2] = z 

    control_coordinates = np.reshape(control_coordinates, 
                                    (x.shape[0]*x.shape[1]*x.shape[2], 3))
    return control_coordinates

## Find a proximity array for each cooridnate in the control space. Use eucledian distance for finding the 
## 'k' close elements. When k is a training hyperparameter to define the search spectrum of controls. 
# ## Here we are using eucledian distance SO : 
# ##                  1. Use a ascending sorted array for fine tuning the controls
#                     2. Use a descending sorted array for broad tuning
#             The above differentiation can be decided ( ie between fine tuning and broad tuning based of the
#             fidelity cost to go. For larger cost to go (or fidelity we expect broad changes in the control for 
#             larger rotations. 
#             But when the fidelity approach 1 we need fine tuning of the controls. 
#             This differentiation can be decided on a finetuning_treshold

#TODO [ALGORITHM] : The search of control space currently is based on the eucledian distance in control space and 
# a treshold in the fidelity base. Need to improve the algorithm to introduce a statistical mixing of these values.
# An approach could be to use the gradient of rotation in the direction of the final state as the correctness measure or reward
#   ie. Since the controls is associated with rotation in x,y,z axis, the set of parameter that creates the maximum rotation in the 
#   direction of the final state should have an advantage. 

#%%

## ordered nearnest neighbours for fine tuning
k_value = 100

def get_ordered_knn_map(k_value):
    
    coordinates = get_control_cordinates_3d()
    ecucledian_distance_matrix = euclidean_distances(coordinates, coordinates)
    ordered_neighbours_map = {}
    for index in range(len(coordinates)):
        proximity = ecucledian_distance_matrix[index].argsort()[:k_value]
        sorted_neighbours = coordinates[proximity]
        current_coordinate  = str(coordinates[index])
        ordered_neighbours_map[current_coordinate] = sorted_neighbours
    return ordered_neighbours_map

## ordered farthest neighbours for broad tuning ## needs to improve the logic. Insted of fartherst 
## a central portion could be an immediate improvement
def get_ordered_kfn_map(k_value):
    
    coordinates = get_control_cordinates_3d()
    ecucledian_distance_matrix = euclidean_distances(coordinates, coordinates)
    ordered_neighbours_map = {}
    for index in range(len(coordinates)):
        proximity = ecucledian_distance_matrix[index].argsort()[::-1][:k_value]
        sorted_neighbours = coordinates[proximity]
        current_coordinate  = str(coordinates[index])
        ordered_neighbours_map[current_coordinate] = sorted_neighbours
    return ordered_neighbours_map

ordered_knn_map = get_ordered_knn_map(k_value)
ordered_kfn_map = get_ordered_knn_map(k_value)

def get_next_actions(current_controls):
    current_controls = str(current_controls)
    current_fidelity = get_fidelity(current_state, target_state)
    if(is_finetuning_required(current_fidelity)):
        possible_controls  = ordered_knn_map[current_controls]
    else:
        possible_controls = ordered_kfn_map[current_controls]
    
    return possible_controls

## TODO [OOP] : Convert the control space functionalities to a Class and define the relevant get and 
# set methods

#%%
from queue import PriorityQueue
## Define the operation to be performed 
target_unitary = q.sigmax()
current_state = q.basis(2,0)
target_state = target_unitary*current_state

## Set the threshold to start fine tuning [with respect to fidelity] 
fine_tuning_treshold = .9999999601441566
# pick the user defined intial control (default 0 0 0). Based on the intial control we find the k controls from knn or kfn for the fist time segment
initial_control = '[0. 0. 0.]'

h =   1   # 2*np.pi*6.6*1e-36  # planks constatn 
w0 = .5   # static magnetic field in T 
w =  .4   # driving frequency

detuning =  w0 - w 

def noise(t, args):  
    #TODO [ALGORITHM]:  make this staochastic using random variable for each segment
    return np.sin(10*t)*10*h/2

H_0 = q.sigmaz()
H_noise = q.sigmam()
H_x = q.sigmax()
H_y = q.sigmay() 

def is_finetuning_required(fidelity):
    return fidelity > fine_tuning_treshold

def get_cost_to_go(fidelity):
    if(fidelity == 0 ):
        cost = float('inf')
    else: 
        cost = 1/fidelity  -1 
    return cost

def get_next_state(current_state):
   current_control = initial_control
   actions = get_next_actions(current_control)
   print("number of actions :", len(actions))
   minimum_cost = float('inf')
  
   next_state = q.basis(2,0)
   preferred_action = current_control
   action_cost = 0 
   
 
   for action in actions : 
       detuning = action[0]
       omx  = action[1]
       omy  = action[2]
       H_control = (h/2)*(detuning*H_0 + omx*H_x + omy*H_y )
       H = [H_control, [H_noise, noise]]
       t = np.arange(0,get_segment_duration(),.1)
       transition_state = q.mesolve(H, current_state, t, []).states[-1]
       #TODO : add the eucledian distance to the action cost. Improve the logic for bias
       action_cost += 1
       fidelity = get_fidelity(transition_state, target_state)
       if(fidelity):
         action_cost += 1 
       else:
         action_cost = 0 
       Q = get_cost_to_go(fidelity)  + action_cost
       if(Q < minimum_cost):
            minimum_cost = Q
            next_state = transition_state
            preferred_action = action

   append_controls(preferred_action[0], preferred_action[1], preferred_action[2])
   return next_state

def get_fidelity(state_i , state_f):
    return (state_i.dag()*state_f).norm()

fidelity = get_fidelity(current_state, target_state)
states = []
while(fidelity < 1):
    states.append(current_state)
    current_state = get_next_state(current_state)
    fidelity = get_fidelity(current_state, target_state)
    print(current_state, fidelity)
#%%
