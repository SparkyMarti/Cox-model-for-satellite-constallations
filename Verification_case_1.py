# ====================================================================================================================================
# Title       : Verification of Case 1
# Description : Compares Monte Carlo simulation and the analytical formulas derived in our Mat-Tek 6 Project
# Author      : Martin Madsen
# Date        : 2025-05-28
# Version     : 1.0
# ====================================================================================================================================

# ============================================================= Imports ==============================================================
import numpy as np
import matplotlib.pyplot as plt
import Analytical_Results as AN
import Monte_Carlo_Simulation as MC
from tqdm import tqdm

# =========================================================== Plot Style =============================================================

import matplotlib as mpl
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.facecolor': 'white',
    'legend.edgecolor': 'black',
})

# ========================================================= Global Parameters ========================================================

CP_params_250={
    'Case':             1,
    'Monte Carlo Iter': 100000,
    'min range':        -10,
    'max range':        60,
    'MC range':         51,
    'AN range':         501,
    'lam':              5,
    'mu' :              5,
    'tau_dB':           0,
    'r_earth':          6371000, 
    'r_sat':            6371000 + 250000,
    'power':            10,
    'gain':             1000,
    'noise_power':      4e-13, #100Mhz 
    'fading_type':      'Rayleigh',     # Rayleigh, Rician, Nakagami
    'fading_parameters':[      ],       #    []       [K]  [m, Omega]
    
}
CP_params_500 = CP_params_250.copy()
CP_params_500['r_sat'] = 6371000 + 500000

CP_params_750 = CP_params_250.copy()
CP_params_750['r_sat'] = 6371000 + 750000

CP_params_1000 = CP_params_250.copy()
CP_params_1000['r_sat'] = 6371000 + 1000000

CP_params_1250 = CP_params_250.copy()
CP_params_1250['r_sat'] = 6371000 + 1250000

CP_list= [CP_params_250, CP_params_500, CP_params_750, CP_params_1000, CP_params_1250]


EC_params={
    'Case':             1,
    'Monte Carlo Iter': 100000,
    'min range':        6371000 + 250000,
    'max range':        6371000 + 1250000,
    'N range':          51,
    'lam':              30,
    'mu' :              30,
    'r_earth':          6371000,
    'r_sat':            6871000,
    'power':            10,
    'gain':             1000,
    'noise_power':      4e-13, #100Mhz 
    'fading_type':      'Rayleigh',     # Rayleigh, Rician, Nakagami
    'fading_parameters':[      ],       #    []       [K]  [m, Omega]
    
}

# =============================================================== 2D Plots ===========================================================

def test_CP(params):
    Case = params['Case']
    N   =  params['Monte Carlo Iter']
    min =  params['min range']
    max =  params['max range']
    MC_range = params['MC range']
    AN_range = params['AN range']
    if Case==1:
        MC_coverage_probability = MC.case_2_cp
        AN_coverage_probability = AN.case_1_coverage_probability
        AN_cp_subparams = MC_cp_subparams = {key: params[key] for key in ['lam', 'mu', 'r_earth', 'r_sat', 'power', 'gain', 'noise_power']}
    elif Case==2:
        MC_coverage_probability = MC.case_2_cp
        AN_coverage_probability = AN.case_2_coverage_probability
        AN_cp_subparams = MC_cp_subparams = {key: params[key] for key in ['lam', 'mu', 'r_earth', 'r_sat', 'power', 'gain', 'noise_power', 'fading_type', 'fading_parameters']}
    elif Case==3:
        MC_coverage_probability = MC.case_3_cp
        AN_coverage_probability = AN.case_3_coverage_probability
        AN_cp_subparams = {key: params[key] for key in ['lam', 'mu', 'r_earth', 'r_sat', 'power', 'gain', 'noise_power']}
        MC_cp_subparams = {key: params[key] for key in ['lam', 'mu', 'r_earth', 'r_sat', 'power', 'gain', 'noise_power', 'fading_type', 'fading_parameters']}

    AN_tau_dB=np.linspace(min,max,AN_range,endpoint=True)
    AN_tau=10**(AN_tau_dB/10)
    AN_result=np.zeros_like(AN_tau)
    MC_tau_dB=np.linspace(min,max,MC_range,endpoint=True)
    MC_tau=10**(MC_tau_dB/10)
    MC_result=np.zeros_like(MC_tau)

    #Monte Carlo Simulation 
    for i in range(N):
        t=MC_coverage_probability(**MC_cp_subparams)
        for j, y in enumerate(MC_tau):
            if t>=y:
                MC_result[j]+=1
        if i % (N // 10) == 0:
            print(f"Completed {i}/{N} Monte Carlo samples")
    MC_result/=N

    #Analytical results
    for i, t in enumerate(tqdm(AN_tau, desc="Analytical CP")):
        AN_result[i]=AN_coverage_probability(tau=t,**AN_cp_subparams)
    if AN_range == MC_range:
        print(f"Max deaviation: {np.max(np.abs(MC_result-AN_result))}, with r_min: {int((params['r_sat']-params['r_earth'])/1000)}")
    return MC_tau_dB, MC_result, AN_tau_dB, AN_result

def CP_Plot(params_list):
    plt.figure(figsize=(10, 4))
    if type(params_list) != list:
        params_list = [params_list]
    for params in params_list:
        l = int((params['r_sat']-params['r_earth'])/1000)
        MC_x, MC_y, AN_x, AN_y = test_CP(params)
        plt.plot(AN_x, AN_y, label=f'{l} km', linewidth=1)
        plt.scatter(MC_x, MC_y, s=15, alpha=0.9)
    plt.xlabel('SINR Threshold [dB]')
    plt.ylabel('Coverage Probability')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def test_EC(params, variable='lam'):
    Case = params['Case']
    N=params['Monte Carlo Iter']
    min = params['min range']
    max = params['max range']
    N_range = params['N range']
    MC_ec_subparams = {key: value for key, value in params.items()
                    if key in ['lam', 'mu', 'r_earth', 'r_sat', 'power', 'gain', 'noise_power']
                    and key != variable}
    if Case==1:
        MC_ergodic_capacity = MC.case_2_ec
        AN_ergodic_capacity = AN.case_1_ergodic_capacity
        AN_ec_subparams = MC_ec_subparams.copy()
    elif Case==2:
        MC_ergodic_capacity = MC.case_2_ec
        AN_ergodic_capacity = AN.case_2_ergodic_capacity
        MC_ec_subparams['fading_type'] = params['fading_type']
        MC_ec_subparams['fading_parameters'] = params['fading_parameters']
        AN_ec_subparams = MC_ec_subparams.copy()
    elif Case==3:
        MC_ergodic_capacity = MC.case_3_ec
        AN_ergodic_capacity = AN.case_3_ergodic_capacity
        AN_ec_subparams = MC_ec_subparams.copy()
        MC_ec_subparams['fading_type'] = params['fading_type']
        MC_ec_subparams['fading_parameters'] = params['fading_parameters']
        

    var_range = np.linspace(min, max, N_range, endpoint=True) 
    MC_results = np.zeros_like(var_range, dtype=float)
    AN_results = np.zeros_like(var_range, dtype=float)

    for i, val in enumerate(tqdm(var_range, desc="Ergodic Capacity")):
        MC_test_params = MC_ec_subparams.copy()
        MC_test_params[variable] = val
        AN_test_params = AN_ec_subparams.copy()
        AN_test_params[variable] = val
        for _ in range(N):
            MC_results[i] += MC_ergodic_capacity(**MC_test_params)
        AN_results[i] = AN_ergodic_capacity(**AN_test_params)
    MC_results/=N

    return var_range, MC_results, var_range, AN_results


def EC_Plot(params, variable):
    MC_x, MC_y, AN_x, AN_y = test_EC(params=params, variable=variable)
    MC_x = (MC_x-6371000)/1000
    AN_x = (AN_x-6371000)/1000
    plt.figure(figsize=(10, 3))
    plt.plot(AN_x, AN_y, linewidth=1)
    plt.scatter(MC_x, MC_y, s=15, alpha=1)
    plt.xlabel('Altitude [km]')
    plt.ylabel('Ergodic Capacity [bits/s/Hz]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    EC_Plot(EC_params,variable='r_sat')
    CP_Plot(CP_list)