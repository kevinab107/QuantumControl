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

from agent import *
from state import *
from control_space import *

"""
Sample run with intial state Qobj = [[1]  and target = [[0]
                                    [0]]                [1]]

The measure of accuracy is fidelity. 
looking for a target accuracy of .99965 
The slighest of variations matters here given the growth of error 
with increase in qubits
"""
action_space = ControlSpace([.4,10,10], [.1, 1, 1])
environment = State(0.25, 
                    q.sigmax(), 
                    q.basis(2,0),
                    [q.sigmaz(), q.sigmax(), q.sigmay(), q.sigmam()])

control_agent = Agent(0.9, environment, action_space)
control_agent.learn_rollout()
