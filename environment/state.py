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

import qutip as q
import numpy as np 
from queue import PriorityQueue

class State(object):
 
    """
    Divide the total time frame of operation into n segments of equal duration. 
    There are two possibilities of going about this. 
       1. Use a heuristic to find the time taken for the whole operation. Assume that 
          this time is our worst case time. ie our algorithm will defnitely improve the 
           time. So divide the time required by heuristic into n segments

       2. Take the interval_width as a tunable parameter. Start with a random value. And 
           find the number of time steps such that, after n steps the state obtained will have 
           the maximum fidelity

        For computational reasons we go by method 2 [it goes better with RL approach]
    TODO [SCALE]: CUrrently the implemention is limited to single qubits. Make it generic using numpy array for n-qubits

    # Set the threshold to start fine tuning [with respect to fidelity] 
    pick the user defined intial control (default 0 0 0). Based on the intial control we find the k controls from knn or kfn for the fist time segment
    
    Create control list which can be appended after each segment duration. 

    detuning, Omega_x and Omega_y are the control parameters (or the actions) which can be varied
    for each time segment. Based on these controls the system will evolve into a new state. We start
    with controlling only the amplitude  and not the wave type (ie its always square wave)
    TODO [SCALE] : Have a functionality for changing the wave type
    
    TODO [STANDARDIZE] : Ensure all the values are float for consistancy. numpy mgrid stores the value aS 0. while a float is 0.0
    # This may create inconsitancy while converting to string. This is a minor issue as the coordinates will be always 
    # used from the coordinate matrix. (Since we are restricting any other control coordinate
    """
    
    def __init__(self, 
                 interval_width, 
                 target_unitary, 
                 initial_state, 
                 hamiltonian):
     
       self.interval_width = interval_width
       self.target_unitary = target_unitary
       self.initial_state = initial_state
       self.current_state = initial_state
       self.hamiltonian = hamiltonian
    
    def get_target_state(self):
        return self.target_unitary*self.initial_state

    def get_current_state(self):
        return self.current_state

    def get_interval_width(self):
        return self.interval_width

    def get_fidelity(self):
        return (self.get_target_state().dag()* self.get_current_state()).norm()
    
    def get_transition_fidelity(self, transition_state):
        return (self.get_target_state().dag()*transition_state).norm()

    def get_hamiltonian_operator_z(self):
        return self.hamiltonian[0]
    
    def get_hamiltonian_operator_x(self):
        return self.hamiltonian[1]
    
    def get_hamiltonian_operator_y(self):
        return self.hamiltonian[2]

    def get_hamiltonian_operator_noise(self):
        return self.hamiltonian[3]

    def noise(self, t, args):  
        #TODO [ALGORITHM]:  make this staochastic using random variable for each segment
        h = 1 
        return np.sin(10*t)*10*h/2
    
    def update_state(self, next_state):
        self.current_state = next_state

    def get_hamiltonian_control(self, controls):
        h = 1 
        detuning = controls[0]
        omegax = controls[1]
        omegay = controls[2]
        H_z = self.get_hamiltonian_operator_z()
        H_x = self.get_hamiltonian_operator_x()
        H_y = self.get_hamiltonian_operator_y()
        H_control = (h/2)*(detuning*H_z 
                           + omegax*H_x 
                           + omegay*H_y)
        return H_control   

    def get_hamiltonian(self, controls):
        H_control = self.get_hamiltonian_control(controls)        
        H_noise = [self.get_hamiltonian_operator_noise(), self.noise]     
        return [H_control, H_noise]  