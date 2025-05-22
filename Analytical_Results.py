# ====================================================================================================================================
# Title       : Analytical_Results
# Description : Computes the formulas derived in our Mat-Tek 6 Project
# Author      : Martin Madsen
# Date        : 2025-05-XX
# Version     : 1.0
# ====================================================================================================================================

# ============================================================= Imports ==============================================================
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.stats import ncx2, expon, gamma
from numba import njit

# ============================================================ Functions =============================================================

def average_visible_satellites(lam, mu, r_earth, r_sat):
    integrand = lambda phi: np.cos(phi) * np.arcsin(np.sqrt(1 - (r_earth / r_sat)**2 * (1 / np.cos(phi)**2)))
    result, _ = quad(integrand, 0, np.arccos(r_earth / r_sat))
    return lam * mu / np.pi * result


def void_probability(lam, mu, r_earth, r_sat):
    integrand = lambda phi: np.cos(phi)*(1-np.exp(-mu/np.pi*np.arcsin(np.sqrt(1-(r_earth/r_sat/np.cos(phi))**2))))
    result, _ = quad(integrand, 0, np.arccos(r_earth / r_sat))
    return np.exp(-lam * result)

@njit
def xi_fun(r_0,r_sat,r_earth):
    return np.arccos((r_sat**2+r_earth**2-r_0**2)/(2*r_earth*r_sat))

def satellite_distance_ccdf(dist, lam, mu, r_earth, r_sat):

    if dist < r_sat-r_earth:
        return 1
    
    if dist > np.sqrt((r_sat)**2-r_earth**2):
        return void_probability(lam,mu,r_earth,r_sat)

    xi=xi_fun(dist,r_sat,r_earth)

    def integrand(phi):
        inner_sqrt = 1 - (np.cos(xi) / np.cos(phi))**2
        arcsin_term = np.arcsin(np.sqrt(inner_sqrt))
        return np.cos(phi) * (1 - np.exp(-mu / np.pi * arcsin_term))
    
    result, _ = quad(integrand, 0, xi)
    return np.exp(-lam*result)


@njit
def constant_pdf(r_0,mu,lam,r_sat,r_earth):
    return r_0 * mu * lam  / (np.pi * r_earth * r_sat )

@njit
def integrand_first_phi(phi,xi,mu):
    inner_sqrt = 1 - (np.cos(xi) / np.cos(phi))**2
    arcsin_term = np.arcsin(np.sqrt(inner_sqrt))
    return np.cos(phi) * (1 - np.exp(-mu / np.pi * arcsin_term))

@njit
def integrand_second_phi(varphi,xi,mu):
    term = np.sqrt(1 - np.cos(xi)**2 * (1 / np.cos(varphi))**2)
    return np.exp(-mu / np.pi * np.arcsin(term))/(term + 1e-10)

def satellite_distance_pdf(r_0, lam, mu, r_earth, r_sat):

    xi = xi_fun(r_0,r_sat,r_earth)
    C = constant_pdf(r_0,mu,lam,r_sat,r_earth)

    integral_first_phi, _ = quad(integrand_first_phi, 0, xi, args=(xi,mu))
    integral_second_phi, _ = quad(integrand_second_phi, 0, xi, args=(xi,mu), limit=200)
    result = C * np.exp(-lam*integral_first_phi) * integral_second_phi
    
    return result


# ============================================================== Case 1 ===============================================================

def case_1_coverage_probability(tau, lam, mu, r_earth, r_sat, power, gain, noise_power):

    threshold = np.sqrt(power * gain / (tau * noise_power))

    return 1 - satellite_distance_ccdf(threshold, lam, mu, r_earth, r_sat)


def case_1_ergodic_capacity(lam, mu, r_earth, r_sat, power, gain, noise_power):
    
    def threshold(r):
        return np.sqrt(power * gain/ ((2**r - 1) * noise_power))

    def integrand(r):
        return (1 - satellite_distance_ccdf(threshold(r), lam, mu, r_earth, r_sat))

    result, _ = quad(integrand, 0, 20)
    return result


# ============================================================== Case 2 ===============================================================

def fading_cdf(fading_type, fading_parameters):
    if fading_type == 'Rayleigh':
        dist = expon
    elif fading_type == 'Rician':
        nc = fading_parameters[0]
        dist = ncx2(2, nc, scale=(1/(2+nc)))
    elif fading_type == 'Nakagami':
        m, omega = fading_parameters
        dist = gamma(a=m,scale=omega/m)
    else:
        raise ValueError("Unsupported fading type")
    
    return dist.cdf  # Return the .cdf method as a function

@njit
def threshold_cp(r_0,tau,power,gain,noise_power):
    return (tau*noise_power*r_0**2)/(power*gain)

def case_2_coverage_probability(lam, mu, r_earth, r_sat, tau, power, gain, noise_power, fading_type, fading_parameters=[]):
    r_min=r_sat-r_earth
    r_max= np.sqrt(r_sat**2-r_earth**2)

    fading = fading_cdf(fading_type, fading_parameters)

    def integrand(r_0):
        return (1-fading(threshold_cp(r_0,tau,power,gain,noise_power)))*satellite_distance_pdf(r_0,lam,mu,r_earth,r_sat)

    result , _ = quad(integrand,r_min,r_max)
    return result

@njit
def threshold_ec(t,r_0,power,gain,noise_power):
        return ((noise_power*r_0**2)/(power * gain)*(2**t - 1))

def case_2_ergodic_capacity(lam, mu, r_earth, r_sat, power, gain, noise_power, fading_type, fading_parameters=[]):
    r_min=r_sat-r_earth
    r_max= np.sqrt(r_sat**2-r_earth**2)

    fading = fading_cdf(fading_type, fading_parameters)

    def integrand(t,r_0):
        return (1-fading(threshold_ec(t,r_0,power,gain,noise_power)))*satellite_distance_pdf(r_0,lam,mu,r_earth,r_sat)

    result, _ = dblquad(integrand, r_min, r_max, 0 , 100)

    return result

# ============================================================== Case 3 ===============================================================

@njit
def xi(r_0,r_sat,r_earth):
    return np.arccos((r_sat ** 2 + r_earth ** 2 - r_0 ** 2) / (2 * r_earth * r_sat))

@njit
def omega_1(phi, r_0 ,r_sat, r_earth):
    return np.arcsin(np.sqrt(1 - (np.cos(xi(r_0,r_sat,r_earth)) / np.cos(phi)) ** 2))

@njit
def omega_2(phi, r_sat, r_earth):
    return np.arcsin(np.sqrt(1 - ((r_earth / r_sat) / np.cos(phi)) ** 2))

@njit
def laplace_integrand(omega, r_0, tau, nu, gain, r_sat, r_earth):
    k_bar = np.sqrt(r_sat**2 + r_earth**2 - 2 * r_earth * r_sat * np.cos(omega) * np.cos(nu))
    return 1 - 1/(1+(tau * r_0 ** 2 / (gain * k_bar ** 2)))

def laplace_integral(lower, upper, r_0, tau, nu, gain, r_sat, r_earth):
    return quad(laplace_integrand, lower, upper, args=(r_0, tau, nu, gain, r_sat, r_earth))[0]


def first_integrand(nu, mu, r_0, r_sat, r_earth, tau, gain):
    laplace = laplace_integral(0.0, omega_2(nu, r_sat, r_earth), r_0, tau, nu, gain, r_sat, r_earth)
    return (1 - np.exp(-mu / np.pi * laplace)) * np.cos(nu)

def first_integral(mu, r_0, r_sat, r_earth, tau, gain):
    lower = xi(r_0, r_sat, r_earth)
    upper = np.arccos(r_earth / r_sat)
    return quad(first_integrand, lower, upper, args=(mu, r_0, r_sat, r_earth, tau, gain))[0]


def second_integrand(nu, mu, r_0, r_sat, r_earth, tau, gain):
    arcsin = omega_1(nu, r_0, r_sat, r_earth)
    laplace = laplace_integral(arcsin, omega_2(nu, r_sat, r_earth), r_0, tau, nu, gain, r_sat, r_earth)
    return (1 - np.exp(-mu / np.pi * (arcsin + laplace))) * np.cos(nu)

def second_integral(mu, r_0, r_sat, r_earth, tau, gain):
    return quad(second_integrand, 0, xi(r_0, r_sat, r_earth), args=(mu, r_0, r_sat, r_earth, tau, gain))[0]


def third_integrand(phi, mu, r_0, r_sat, r_earth, tau, gain):
    arcsin = omega_1(phi, r_0, r_sat, r_earth)
    laplace = laplace_integral(arcsin, omega_2(phi, r_sat, r_earth), r_0, tau, phi, gain, r_sat, r_earth)
    xi_val = xi(r_0, r_sat, r_earth)
    return (np.exp(-mu / np.pi * (arcsin + laplace))) / (np.sqrt(1 - (np.cos(xi_val) / np.cos(phi)) ** 2))

def third_integral(mu, r_0, r_sat, r_earth, tau, gain):
    xi_val = xi(r_0, r_sat, r_earth)
    return quad(third_integrand, 0, xi_val, args=(mu, r_0, r_sat, r_earth, tau, gain))[0]


def case_3_coverage_probability(mu, lam, tau, gain, r_sat, r_earth, power, noise_power):
    integrand = lambda r_0: lam * mu * r_0 / (np.pi * r_earth * r_sat) * np.exp(-noise_power*tau*r_0**2/(power*gain)) * np.exp(
        -lam * first_integral(mu, r_0, r_sat, r_earth, tau, gain)
    ) * np.exp(
        -lam * second_integral(mu, r_0, r_sat, r_earth, tau, gain)
    ) * third_integral(mu, r_0, r_sat, r_earth, tau, gain)
    return quad(integrand, r_sat-r_earth, np.sqrt(r_sat ** 2 - r_earth ** 2))[0]


def case_3_ergodic_capacity(lam, mu, r_earth, r_sat, power, gain, noise_power):

    def integrand(r):
        return (case_3_coverage_probability(mu, lam, (2**r - 1), gain, r_sat, r_earth, power, noise_power))

    result, _ = quad(integrand, 0, 20)
    return result

# ============================================================= End of File ==========================================================

