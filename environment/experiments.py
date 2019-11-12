#%%
from state import State
from control_space import ControlSpace
from agent import Agent
import numpy as np
import qutip as q 

action_space = ControlSpace([1,1,1])
environment = State(0.25, 
                    q.sigmax(), 
                    q.basis(2,0),
                    [q.sigmaz(), q.sigmax(), q.sigmay(). q.sigmam()])

control_agent = Agent(0.9, environment, action_space)

control_agent.learn_rollout()

# %%
