# QuantumControl
** A work in progress repository for quantum control using Reinforcement Learning**
Quantum Control for high fidelity and fast quantum gates plays a crucial role in scalable quantum computing as it is an interface between the hardware stack and the cloud/compiler stack. The project is an attempt to make a universal (hardware independent) software solution for quantum control using Reinforcement learning. 

## Resources
1) Reinforcement laerning and optimal quantum control (Prof. Dimitri P. Bertsekas)
2) Universal quantum control through deep reinforcement learning, Murphy Yuezhen Niu, Sergio Boixo, Vadim N. Smelyanskiy & Hartmut Neven 
3) Reinforcement Learning in Different Phases of Quantum Control, Marin Bukov, Alexandre G.R. Day, Dries Sels, Phillip Weinberg, Anatoli      Polkovnikov and Pankaj Mehta1

## Approach 
Much of the work is based on the resources provided above. For implementation QuTip is used for qubit dynamics simulation and visualization(in bloch sphere), partial differential equations. The GRAPE algorithm provided in the QuTip is utilized as a heuristic in the approach.  

Using GRAPE algorithm for quantum control as the heuristic, various RL approaches are tested to implement a Unitary on an approximated hamiltonian.  

The simulation package provided in QuTip is used for creating the agent. The sequence created using GRAPE algorithm is used as the control. 
The reward and cost function is adapted from the above resources. 

Below given is a short description of the planned software development timeline

                                 10/10/19---------------|-----Complete the Simulation and Heuristics system
                                                        |
                                 24/10/19---------------|-----Code clean-up and documentation   / Add to git                                                                                     
                                 24/10/19---------------|----- Convert the system output to dynamic decoupling sequence     
                                                        |
                                 24/10/19---------------|----- Integrate with open quantum control
                                                        |           - https://github.com/qctrl/python-open-controls.git
                                                        |            - Use open quantum control to make use of IBM Q                                                                                                 
                                                        |
                              Deadline to be decided----|----- Create a feedback system from IBM Q using qistkit and improve the                                                   
                                                        |      RL agent based on realtime feedback 
                                                        |
                                                        |------ Apply an improved RL algorithm and compare the 
