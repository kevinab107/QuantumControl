# %%
import qutip.control.pulseoptim as cpo
import matplotlib.pyplot as plt
from qutip import *
import numpy as np
from qutip import Qobj, identity, sigmax, sigmaz
from qutip.qip import hadamard_transform
import qutip.logging_utils as logging
logger = logging.get_logger()
# Set this to None or logging.WARN for 'quiet' execution
log_level = logging.INFO
# QuTiP control modules


def get_pulseoptim_results(H_d, H_c, U_targ, alg='CRAB'):

    # Number of time slots
    n_ts = 10
    # Time allowed for the evolution
    evo_time = 1

    fid_err_targ = 1e-10
    # Maximum iterations for the optisation algorithm
    max_iter = 200
    # Maximum (elapsed) time allowed in seconds
    max_wall_time = 120
    # Minimum gradient (sum of gradients squared)
    # as this tends to 0 -> local minima has been found
    min_grad = 1e-20

    p_type = 'SQUARE'

    return cpo.optimize_pulse_unitary(H_d, H_c, identity(2), U_targ, n_ts, evo_time,
                                      fid_err_targ=fid_err_targ, min_grad=min_grad,
                                      max_iter=max_iter, max_wall_time=max_wall_time,
                                      out_file_ext=None, init_pulse_type=p_type, alg=alg,
                                      log_level=log_level, gen_stats=True)


def plot_benchmark_pulse(result):
    amps = result.final_amps.transpose()
    fig1 = plt.figure()
    steps = range(amps.shape[1])
    ax1 = fig1.add_subplot(2, 1, 1)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("X")
    ax1.step(steps,
             amps[0])
    ax2 = fig1.add_subplot(2, 1, 2)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Y")
    ax2.step(steps,
             amps[1])
    plt.tight_layout()
    plt.show()
    print("Final evolution\n{}\n".format(result.evo_full_final))
    print("********* Summary *****************")
    print("Final fidelity error {}".format(result.fid_err))
    print("Final gradient normal {}".format(result.grad_norm_final))
    print("Terminated due to {}".format(result.termination_reason))
    print("Number of iterations {}".format(result.num_iter))
    # %%


H_d = 2*np.pi*10*.1*sigmaz()
H_c = [2*np.pi*10*sigmax(), 2*np.pi*10*sigmay()]
U_targ = sigmax()
alg = 'GRAPE'
res = get_pulseoptim_results(H_d, H_c, U_targ, alg)

# %%
plot_benchmark_pulse(res)

# %%
