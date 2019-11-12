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

from state import State
from control_space import ControlSpace
import numpy as np
import qutip as q 

class Agent:
    def __init__(self,
                 fine_tuning_threshold,
                 state,
                 control_space,
                 initial_control = '[0. 0. 0.]', 
                 current_control = None):
       
       self.fine_tuning_threshold = fine_tuning_threshold
       self.initial_control = '[0. 0. 0.]' 
       self.control_detuning = []
       self.control_omega_x = []
       self.control_omega_y =[]
       self.state = state
       self.current_control = '[0. 0. 0.]' 
       self.control_space = control_space

    def get_current_segment(self):
        return len(control_detuning)

    def get_environment(self):
        return self.state

    def is_valid_segment(self, segment):
        return segment < self.get_current_segment()

    def get_segment(self, t):
        return int(t/self.state.get_interval_width())

    def is_finetuning_required(self):
        return self.state.get_fidelity() > self.get_fine_tuning_threshlod()

    def get_fine_tuning_threshlod(self):
        return self.fine_tuning_threshold

    def get_detuning_control(self, t):
        segment = self.get_segment(t)
        if(self.is_valid_segment(segment)):
            return self.control_detuning[segment]
        else:
            return 0

    def get_omega_x_control(self, t):
        segment = self.get_segment(t)
        if(self.is_valid_segment(segment)):
            return self.control_omega_x[segment]
        else:
            return 0

    def get_omega_y_control(self, t):
        segment = self.get_segment(t)
        if(self.is_valid_segment(segment)):
            return self.control_omega_y[segment]
        else:
            return 0

  
    def append_controls(self, detuning, omega_x, omega_y):
        
        """
        update the control list with the latest 
        TODO [STANDARDIZE] : Ensure all the control list has same lenght
        """
        self.control_detuning.append(detuning)
        self.control_omega_x.append(omega_x)
        self.control_omega_y.append(omega_y)

    def update_current_action(self, action):
        self.current_control = str(action)

    def get_current_control(self):
        return self.current_control

    def clear_controls(self):
        self.control_detuning.clear()
        self.control_omega_x.clear()
        self.control_omega_y.clear()

    def get_cost_to_go(self, fidelity):
        if(fidelity == 0):
            cost = float('inf')
        else: 
            cost = 1/fidelity - 1 
        return cost
    
    def get_next_actions(self, current_control, environment):
        current_controls = str(current_control)
        current_fidelity = environment.get_fidelity()
        if(self.is_finetuning_required()):
            possible_controls  = self.control_space.get_nearest_controls(current_control)
        else:
            possible_controls = self.control_space.get_farthest_controls(current_control)
        
        return possible_controls

    def get_next_state(self, current_state, current_control, environment): 
        actions = self.get_next_actions(current_control, environment)
        print("number of actions :", len(actions))
        minimum_cost = float('inf')
        current_state = environment.get_current_state()
        preferred_action = current_control
        action_cost = 0 
        
        for action in actions : 
            H = environment.get_hamiltonian(action)
            t = np.arange(0,environment.get_interval_width(),.1)
            transition_state = q.mesolve(H, current_state, t, []).states[-1]
            #TODO : add the eucledian distance to the action cost. Improve the logic for bias
            action_cost += 1
            fidelity = environment.get_transition_fidelity(transition_state)
            if(fidelity):
                action_cost += 1 
            else:
                action_cost = 0 
            Q = self.get_cost_to_go(fidelity)  + action_cost
            if(Q < minimum_cost):
                    minimum_cost = Q
                    next_state = transition_state
                    preferred_action = action

        self.append_controls(preferred_action[0], preferred_action[1], preferred_action[2])
        self.update_current_action(preferred_action)
        environment.update_state(next_state)
        return next_state

    def learn_rollout(self):
        environment = self.state      
        fidelity = environment.get_fidelity()
        current_state = environment.get_current_state()
        states = []
        while(fidelity < 1):
            states.append(environment.get_current_state())
            current_state = self.get_next_state(current_state, 
                                            self.current_control, environment)

            fidelity = state.get_fidelity()
            print(current_state, fidelity)