# ====================================================================================================================================
# Title       : Monte_Carlo_Simulations
# Description : Monte Carlo simulation of the formulas derived in our Mat-Tek 6 Project
# Author      : Martin Madsen
# Date        : 2025-05-28
# Version     : 1.0
# ====================================================================================================================================

# ============================================================= Imports ==============================================================
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# ============================================================ Functions =============================================================

def cox_process(lam, mu, r_sat=7000000):
    n_orbits = np.random.poisson(lam)
    if n_orbits == 0:
        return np.array([[0], [0], [np.inf]])

    theta = np.random.uniform(0, np.pi, n_orbits)
    phi = np.arccos(1 - 2 * np.random.uniform(0, 1, n_orbits))
    n_sats = np.random.poisson(mu, n_orbits)

    total_sats = np.sum(n_sats)
    if total_sats == 0:
        return np.array([[0], [0], [np.inf]])

    valid = n_sats > 0
    theta = np.repeat(theta[valid], n_sats[valid])
    phi = np.repeat(phi[valid], n_sats[valid])

    omega = np.random.uniform(0, 2 * np.pi, total_sats)

    return orbital_to_cartesian(r_sat, theta, phi, omega)


@njit
def orbital_to_cartesian(r, theta, phi, omega):
    x = r * (np.cos(theta) * np.cos(omega) - np.sin(theta) * np.sin(omega) * np.cos(phi))
    y = r * (np.sin(theta) * np.cos(omega) + np.cos(theta) * np.sin(omega) * np.cos(phi))
    z = r * (np.sin(omega) * np.sin(phi))
    return x, y, z

@njit
def calculate_r_i(sats,user):
    lenghts=np.zeros_like(sats[0])
    for i in range(len(sats[0])):
        lenghts[i]=np.sqrt((sats[0][i]-user[0])**2+(sats[1][i]-user[1])**2+(sats[2][i]-user[2])**2)
    return np.min(lenghts), np.delete(lenghts, np.argmin(lenghts))


@njit
def SNR(power,gain,h,r0,alpha,noise_power,r_max=10):
    if r0<=r_max:
        return (power * gain * h) / (r0**alpha * noise_power)
    return 0

# ======================================================== Isotropy Test =============================================================

@njit
def count_satellites_in_segments(var, segments=20):
    count=np.zeros(segments)
    for i in range(segments):
        for j, val in enumerate(var):
            if -1 + 2 * i / segments <= val <= -1 + 2 * (i + 1) / segments:
                count[i] +=1
    return count

def Isotropy(lam, mu, segments=200, n=100000):
    count=np.zeros([3,segments])
    for _ in range(n):
        x,y,z  = cox_process(lam,mu,r_sat=1)
        count[0] += count_satellites_in_segments(x,segments=segments)
        count[1] += count_satellites_in_segments(y,segments=segments)
        count[2] += count_satellites_in_segments(z,segments=segments)
    plt.plot(np.transpose(count/n))
    plt.ylim(0,1.5*np.max(count/n))
    plt.title("Average Satellite Distribution Histogram (X, Y, Z)")
    plt.xlabel("Segment Index")
    plt.ylabel("Satellites/Segment")
    plt.legend(["x", "y", "z"])
    plt.show()


# ============================================================ Case 1&2 ==============================================================


def fading(fading_type, fading_parameters):
    if fading_type == 'Rayleigh':
        return np.random.exponential(1)
    elif fading_type == 'Rician':
        nc = fading_parameters[0]
        return np.random.noncentral_chisquare(2, nc)/(2+nc)
    elif fading_type == 'Nakagami':
        m, Omega = fading_parameters
        return np.random.gamma(shape=m, scale=Omega/m)
    else:
        return 1

def case_2_cp(lam, mu, r_earth, r_sat, power, gain, noise_power, fading_type=None, fading_parameters=None):
    satelites=cox_process(lam,mu,r_sat=r_sat)
    r0=calculate_r_i(satelites,np.array([0,0,r_earth]))[0]
    r_max=np.sqrt(r_sat**2-r_earth**2)
    h = fading(fading_type, fading_parameters)
    return SNR(power,gain,h,r0,2,noise_power,r_max=r_max)

def case_2_ec(lam, mu, r_earth, r_sat, power, gain, noise_power, fading_type=None, fading_parameters=None):
    satelites=cox_process(lam,mu,r_sat=r_sat)
    r0=calculate_r_i(satelites,np.array([0,0,r_earth]))[0]
    r_max=np.sqrt(r_sat**2-r_earth**2)
    h = fading(fading_type, fading_parameters)
    val = SNR(power,gain,h,r0,2,noise_power,r_max=r_max)
    return np.log2(1+val)

# ============================================================== Case 3 ===============================================================

def generate_fading_array(size, fading_type, fading_parameters):
    if fading_type == 'Rayleigh':
        return np.random.exponential(1.0, size)
    elif fading_type == 'Rician':
        nc = fading_parameters[0]
        return np.random.noncentral_chisquare(2, nc, size)/(2+nc)
    elif fading_type == 'Nakagami':
        m, Omega = fading_parameters
        return np.random.gamma(m, Omega/m, size)
    else:
        return np.ones(size)

@njit
def SINR(r_0, r_i, h_0, h_i, power, gain, noise_power):

    signal = power * gain * h_0 / (r_0 ** 2)
    interference = noise_power
    for j in range(len(r_i)):
        interference += power * h_i[j] / (r_i[j] ** 2)

    return signal / interference if interference > 0 else np.inf


def case_3_cp(lam, mu, r_earth, r_sat, power, gain, noise_power, fading_type=None, fading_parameters=None):
    satelites=cox_process(lam,mu,r_sat=r_sat)
    r_0, sat_dist = calculate_r_i(satelites,np.array([0,0,r_earth]))
    r_max=np.sqrt(r_sat**2-r_earth**2)
    if r_0 > r_max:
        return 0
    r_i = sat_dist[sat_dist <= r_max]
    if len(r_i) == 0:
        r_i=np.array([np.inf])
    h_0 = fading(fading_type, fading_parameters)
    h_i = generate_fading_array(len(r_i), fading_type, fading_parameters)
    return SINR(r_0, r_i, h_0, h_i, power, gain, noise_power)


def case_3_ec(lam, mu, r_earth, r_sat, power, gain, noise_power, fading_type=None, fading_parameters=None):
    satelites=cox_process(lam,mu,r_sat=r_sat)
    r_0, sat_dist=calculate_r_i(satelites,np.array([0,0,r_earth]))
    r_max=np.sqrt(r_sat**2-r_earth**2)
    if r_0 > r_max:
        return 0
    r_i = sat_dist[sat_dist <= r_max]
    if len(r_i) == 0:
        r_i=np.array([np.inf])
    h_0 = fading(fading_type, fading_parameters)
    h_i = generate_fading_array(len(r_i), fading_type, fading_parameters)
    val = SINR(r_0, r_i, h_0, h_i, power, gain, noise_power)
    return np.log2(1+val)

if __name__=='__main__':

    Isotropy(10,10)

# ============================================================= End of File ==========================================================