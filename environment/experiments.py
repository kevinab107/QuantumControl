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
Under development (code for experimentation.) RL will be experimented here
"""

#%%
import numpy as np
import qutip as q

# Hamiltonian 
h =   1   # 2*np.pi*6.6*1e-36  # planks constatn 
w0 = .5   # static magnetic field in T 
w =  .4   # driving frequency
detuning =  w0 - w 
def Omega_x(t, args):
    return 0*h/2     # Rabi rate  along x 

def Omega_y(t, args):
    return 0*h/2     # Rabi rate  along y

def noise(t, args):  # make this staochastic using random variable
    return np.sin(10*t)*10*h/2

H_0 = h/2*(detuning*q.sigmaz())
H_noise = q.sigmam()
H_x = q.sigmax()
H_y = q.sigmay() 

#H  = h/2*(detuning*q.sigmaz() + noise*q.sigmaz() + Omega_x*q.sigmax() + Omega_y*q.sigmay())
H = [H_0, [H_noise, noise], [H_x, Omega_x], [H_y, Omega_y]]

#%% Simulation 
psi0 = q.basis(2,0) 
t = np.linspace(0 , 2, 100)
c_ops = []

output = q.mesolve(H, psi0, t, c_ops)

# %% Visualize the evolution 

sphere = q.Bloch()
for states in output.states:
    sphere.clear()
    sphere.add_states(states)
    sphere.save(dirc = 'temp')

# %%
