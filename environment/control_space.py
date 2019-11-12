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

import numpy as np 
from sklearn.metrics.pairwise import euclidean_distances

class ControlSpace:
    """
    Class to predefine the action space for the agent. 
    This includes all possible control sequence in the space 
    called control space with the dimensions as :
    detuning
    omega_x
    omega_y

    TODO [IMPLEMENTATION] : The control space is currently defined for single qubits.
    Make a generalized n-d array implementation for n qubits where the controls 
    will be 

    detuning_q1
    omega_x_q1
    omega_y_q1
    detuning_q2
    omega_x_q2
    omega_y_q2
    ...
    ...
    detuning_qn
    omega_x_qn
    omega_y_qn

    exchange_q1q2
    exchangne_q1q3
    ...
    exchange_q1qn
    ...
    echange_qn-1qn
    """

    def __init__(self, 
                 control_bound, 
                 control_change, 
                 k_value = 100):
       self.control_bound = control_bound
       self.control_change = control_change
       self.k_value = k_value
       self.ordered_knn_map = None
       self.ordered_kfn_map = None
       self.set_ordered_nn_maps()

    def get_control_bound(self):
        return self.control_bound

    def get_control_change(self):
        return self.control_change

    def get_control_bound_detuning(self):
        return self.get_control_bound()[0]
    
    def get_control_change_detuning(self):
        return self.get_control_change()[0]

    def get_control_bound_omega_x(self):
        return self.get_control_bound()[1]
    
    def get_control_change_omega_x(self):
        return self.get_control_change()[1]

    def get_control_bound_omega_y(self):
        return self.get_control_bound()[2]
    
    def get_control_change_omega_y(self):
        return self.get_control_change()[2]

    def get_detuning_actions(self):
        return np.arange(0, 
                         self.get_control_bound_detuning(), 
                         self.get_control_change_detuning())
    
    def get_omega_x_actions(self):
        return np.arange(0, 
                         self.get_control_bound_omega_x(), 
                         self.get_control_change_omega_x())
                    
    
    def get_omega_y_actions(self):
        return np.arange(0, 
                         self.get_control_bound_omega_y(), 
                         self.get_control_change_omega_y())    
    
    def get_ordered_knn_map(self):
        return self.ordered_knn_map
    
    def get_ordered_kfn_map(self):
        return self.ordered_kfn_map
    
    def get_nearest_controls(self, control):
        return self.get_ordered_knn_map()[control]

    def get_farthest_controls(self, control):
        return self.get_ordered_kfn_map()[control]

    def get_control_cordinates_3d(self):
        
        """  
        Since the cost is independent of the state we can precalculate them and arrange them in a priority queue 
        with corresponding cost of the transition in a map for easier computation

        #Use a numpy mgrid to store the above actions for compulation of eucledian distance. 
        %%
    
        TODO [OOP] : Encapsulate the control range into a class (ControlRange) for easy initialization and variations 

        Create an array of coordinates corresponding to all possible combination of 
        control actions 
        TODO: Make this a method of ControlRange
        """
        
        x,y,z = np.mgrid[0. : self.get_control_bound_omega_x() : self.get_control_change_omega_x(),
                         0. : self.get_control_bound_omega_y() : self.get_control_change_omega_y(),
                         0. : self.get_control_bound_detuning(): self.get_control_change_detuning()]

        control_coordinates = np.empty(x.shape + (3,), dtype = float)
        control_coordinates[:,:,:,0] = x                   
        control_coordinates[:,:,:,1] = y
        control_coordinates[:,:,:,2] = z 

        control_coordinates = np.reshape(control_coordinates, 
                                        (x.shape[0]*x.shape[1]*x.shape[2], 3))
        return control_coordinates

    
    def set_ordered_knn_map(self):
        """
        Find a proximity array for each cooridnate in the control space. Use eucledian distance for finding the 
        'k' close elements. When k is a training hyperparameter to define the search spectrum of controls. 
        
        Here we are using eucledian distance SO : 
                        1. Use a ascending sorted array for fine tuning the controls
                        2. Use a descending sorted array for broad tuning
                            The above differentiation can be decided ( ie between fine tuning and broad tuning based of the
                            fidelity cost to go. For larger cost to go (or fidelity we expect broad changes in the control for 
                            larger rotations. 
                            But when the fidelity approach 1 we need fine tuning of the controls. 
                            This differentiation can be decided on a finetuning_treshold

        TODO [ALGORITHM] : The search of control space currently is based on the eucledian distance in control space and 
        a treshold in the fidelity base. Need to improve the algorithm to introduce a statistical mixing of these values.
        An approach could be to use the gradient of rotation in the direction of the final state as the correctness measure or reward
        
        ie. Since the controls is associated with rotation in x,y,z axis, the set of parameter that creates the maximum rotation in the 
        direction of the final state should have an advantage. 
        """
        
        coordinates = self.get_control_cordinates_3d()
        ecucledian_distance_matrix = euclidean_distances(coordinates, coordinates)
        ordered_neighbours_map = {}
        for index in range(len(coordinates)):
            proximity = ecucledian_distance_matrix[index].argsort()[:self.k_value]
            sorted_neighbours = coordinates[proximity]
            current_coordinate  = str(coordinates[index])
            ordered_neighbours_map[current_coordinate] = sorted_neighbours
        self.ordered_knn_map = ordered_neighbours_map

     
    def set_ordered_kfn_map(self):
        """
        ordered farthest neighbours for broad tuning ## needs to improve the logic. Insted of fartherst 
        a central portion could be an immediate improvement
        """
        
        coordinates = self.get_control_cordinates_3d()
        ecucledian_distance_matrix = euclidean_distances(coordinates, coordinates)
        ordered_neighbours_map = {}
        for index in range(len(coordinates)):
            proximity = ecucledian_distance_matrix[index].argsort()[::-1][:self.k_value]
            sorted_neighbours = coordinates[proximity]
            current_coordinate  = str(coordinates[index])
            ordered_neighbours_map[current_coordinate] = sorted_neighbours
        self.ordered_kfn_map = ordered_neighbours_map

    def set_ordered_nn_maps(self):
        self.set_ordered_knn_map()
        self.set_ordered_kfn_map()