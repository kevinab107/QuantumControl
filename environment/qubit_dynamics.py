"""
===============================
environment.qubit_dynamics
===============================
"""
import json
import numpy as np
import qutip as q
from qutip.control import *
from ..exceptions.exceptions import ArgumentsValueError


class SystemVariables(object):   #pylint: disable=too-few-public-methods
    """
    Creates a System environment with an initialized state of the qubit. 
    Parameters
    ----------
    H : class:`qutip.Qobj`
        System Hamiltonian, or a callback function for time-dependent Hamiltonians
        A list of format [H0, [H1, H1_coeff]] =  [Qutip.Qobj , [Qutip.Qobj, python time dependent function]] 
        that represents a time dependent hamiltonian 
        H0 = time independent term , H1 = Time dependent term , H1_coeff = time dependent function      
    resonance_frequency : int, optional
        frequency(Hz) corresponding to the Zeeman Splitting; Defaults to None
    initial_state : qutip.Qobj, optional 
        State corresponding to the initial state of the qubit. 
        Default to ground state |0>
    Unitary : qutip.Qobj, optional
        Unitary operation to be implemented in the system
    final_state : qutip.Qobj, optional
        Alternative to providing the unitary, provide the final state
    time_max : int, optional
        Maximum time in milliseconds that can be untilized for implementing the 
        unitary. Default to None, which implies there is no time 
        constraint
    t_frame : int 
            time frame width of evaluation in milliseconds
    name : string, optional
        An optional string to name the system. Defaults to None.
    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """

    def __init__(self,
                 hamiltonian=None,
                 resonance_frequency=None,
                 initial_state=None,
                 Unitary = None,
                 final_state = None,
                 time_max = None,
                 t_frame = None,
                 name=None):

        self.name = name
        if self.name is not None:
            self.name = str(self.name)

        check_none_values = [(hamiltonian is None), (resonance_frequency is None),
                             (initial_state is None), (Unitary is None),(final_state is None), 
                             (time_max is None), (t_frame is None)]
        all_are_none = all(value is True for value in check_none_values)
        if all_are_none:
            hamiltonian = q.sigmaz()
            resonance_frequency = 0
            initial_state = q.basis(2,0)
            Unitary = q.sigmax()
            time_max = None
        else:
            # some may be None while others are not
            if check_none_values[0]:
                hamiltonian = q.sigmaz()

            if check_none_values[1]:
                resonance_frequency = 0

            if check_none_values[2]:
                initial_state = q.basis(2,0)

            if check_none_values[3] and check_none_values[4]:
                Unitary = q.sigmax()


        self.hamiltonian = hamiltonian
        self.resonance_frequency = resonance_frequency
        self.initial_state = initial_state
        self.Unitary = Unitary
        self.time_max = time_max

        # TODO: check if the hamiltonian is valid
        

        # check if all the resonance frequency is greater than zero
        if (resonance_frequency < 0):
            raise ArgumentsValueError('resonance_frequency must be greater'
                                      + ' than zero.',
                                      {'resonance_frequency': self.resonance_frequency})

        # check if all the initial state is a qutip.qob
        if not isinstance(initial_state, q.Qobj):
            raise ArgumentsValueError('initial_state must be a Qobj',
                                      {'initial_state': initial_state},
                                      extras={
                                          'hamiltonian': hamiltonian,
                                          'resonance_frequency': resonance_frequency,
                                          'initial_state': initial_state})

        if not isinstance(hamiltonian, q.Qobj):
            raise ArgumentsValueError('Unitary must be a Qobj',
                                      {'Unitary': Unitary},
                                      extras={
                                          'hamiltonian': hamiltonian,
                                          'resonance_frequency': resonance_frequency,
                                          'initial_state': initial_state,
                                          'Unitary': Unitary,
                                          'time_max' : time_max})

def get_free_evolution_states(self, t_limit):
        q.mesolve
        """Gets the free evolution state array.

        Parameters
        ----------
        t_limit : int
            maximum time frame for evolution

        Returns
        -------
        np.array
            A numpy array containing Qutip.qobj states
            corresponding to the evolved path

        Raises
        ------
        ArgumentsValueError
            Raised if `t_frame` is greater than 't_limit'.
        """
        H = self.hamiltonian
        initial_state = self.initial_state
    

        if (t_frame > t_limit):
            raise ArgumentsValueError("frame width cannot be greater than maximum time limit")
            
        n_frames = int(t_limit/self.t_frame)
        t_list = np.linspace(0, t_limit, n_frames)
        evolution = q.mesolve(H, 
                             initial_state, 
                             t_list)
        
        return evolution.states

def get_approximate_hamiltonian(self, heuristic, H_ops):
       
        """Gets the approximate hamiltonian for the unitary operation

        Parameters
        ----------
        heuristic : string, optional
            The approximation algorithm to be used
            defautl, usse GRAPE algorithm to definea control sequence 
            and derive a hamiltonian 
        H_ops : list, Qutip.Qobj,
            list of allowed control actions 
            eg : [sigmax()]
        Returns
        -------
        Qutip.Qobj
            A Qutip.qobj states corresponding to the Hamiltonian

        TODO : add funcitonality for multiple heuristic algorithm. Currenlty 
        only use GRAPE algorithm
        """
        #Extract the time independent part of the hamiltonian for approximation
           
        H = self.hamiltonian[0]
        initial_state = self.initial_state
        heuristic = "GRAPE"
        grape_iterations = 150
        target_Unitary = self.Unitary
        T = self.time_max
        t_frame = self.t_frame

        n_frames = int(T/t_frame)
        times = np.linspace(0, T, n_frames)
        u0 = np.array([np.random.rand(len(times)) * 2 * np.pi * 0.005 
                      for _ in range(len(H_ops))])
        u0 = [np.convolve(np.ones(10)/10, u0[idx,:], 
                         mode='same') 
                         for idx in range(len(H_ops))]
        
        result = cy_grape_unitary(target_Unitary, 
                                 H, 
                                 H_ops, 
                                 grape_iterations, 
                                 times, 
                                 u_start=u0, 
                                 eps=2*np.pi/T, 
                                 phase_sensitive=False)
                              
        
        return result.H_t

def get_unitary_evolution_states(self, H_ops):
        q.mesolve
        """Gets the unitary evolution under heuristic hamiltonian state array.

        Parameters
        ----------
        H_ops : list, Qutip.Qobj,
            list of allowed control actions 
            eg : [sigmax()]

        Returns
        -------
        list
            A list array containing Qutip.qobj states
            corresponding to the evolved path

        """
        H = get_approximate_hamiltonian(self, "GRAPE", H_ops)
        initial_state = self.initial_state
        T = self.time_max
        t_frame = self.t_frame
        n_frames = int(T/t_frame)
        times = np.linspace(0, T, n_frames )
        
        evolution = q.mesolve(H, 
                             initial_state, 
                             times)
        
        return evolution.states

def get_fidelity_array(self, states, reference_state):
        q.mesolve
        """Gets the fidelity array corresponding to the final state with respect to
           an array of states.

        Parameters
        ----------
        states : list, Qutip.Qobj,
            list of evolved quatnum states 

        reference_state: Qutip.Qobj
            state to which the fidelity has to be calculated
        
        Returns
        -------
        list
            A list containing fidelity value in float

        Raises
        ------
        ArgumentsValueError
            Raised if `t_frame` is greater than 't_limit'.
        """

        if not all(isinstance(x, q.Qobj) for x in states):
            raise ArgumentsValueError("States are not Qutip.Qobj types")
        if not isinstance(reference_state, q.Qobj):
            raise ArgumentsValueError("State is not Qutip.Qobj types")

        result = [states[i].dag()*reference_state for i in range(len(states))]
        result = [result[i].norm() for i in range(len(result))]
        
        return result

def get_max_fidelity(self, fidelity_array):
    """Gets maximum fidelity and the corresponding time of acheivement 
       from the fidelity array 

        Parameters
        ----------
        fidelity_array : list, int
            list of fidelities
        
        Returns
        -------
        list
            A list containing maximum fidelity value and corresponding time
    """       
    max_fidelity = max(fidelity_array)
    time_achieved = fidelity_array.index(max_fidelity)*self.t_frame
    return max_fidelity, time_achieved


if __name__ == '__main__':
    pass