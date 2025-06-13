import numpy as np

def calculate_analytical_crb(config):
    """
    Calculates the analytical Cramer-Rao Bound (CRB) for target estimation.

    Args:
        config: A configuration object containing system and simulation parameters.

    Returns:
        None (for now).
    """

    # Create a grid of Delay-Doppler indices (l, k) centered at zero
    l_indices = np.arange(config.M)
    k_indices = np.arange(config.N)
    l_grid, k_grid = np.meshgrid(l_indices, k_indices)

    # Calculate the 2D Gaussian pulse x_dd
    # Ensure to use (l - M/2) and (k - N/2) for centering if M and N are even.
    # If M or N is odd, this will be slightly off-center, but consistent with typical DFT indexing.
    # For simplicity, we'll assume M and N are used directly as grid sizes
    # and the pulse is centered within this grid.
    # The prompt uses (l - M/2) and (k - N/2) which implies a shift.
    # Let's use np.fft.fftshift to handle centering correctly for FFT-based grids.

    # Centered indices for Gaussian calculation
    l_centered = l_grid - config.M / 2
    k_centered = k_grid - config.N / 2

    x_dd = np.exp(-((l_centered**2 / config.sigma_tau_pulse**2) +
                    (k_centered**2 / config.sigma_nu_pulse**2)))

    # Normalize x_dd so that its total energy E_tx = sum(|x_dd[l,k]|^2) is 1
    E_tx = np.sum(np.abs(x_dd)**2)
    x_dd = x_dd / np.sqrt(E_tx)

    # Calculate numerical gradients
    # np.gradient returns gradients along each axis.
    # The first element is gradient along axis 0 (rows, corresponding to k or Doppler in our meshgrid)
    # The second element is gradient along axis 1 (columns, corresponding to l or Delay in our meshgrid)
    grad_k_x_dd, grad_l_x_dd = np.gradient(x_dd)

    # Adjust gradients based on delta_tau and delta_nu
    # deriv_tau_prime = d(x_dd)/dl * dl/d(tau_prime) = grad_l_x_dd / config.delta_tau
    # deriv_nu_prime  = d(x_dd)/dk * dk/d(nu_prime)  = grad_k_x_dd / config.delta_nu
    deriv_tau_prime = grad_l_x_dd / config.delta_tau
    deriv_nu_prime = grad_k_x_dd / config.delta_nu

    # Calculate energies of the derivatives
    # The sum is over discrete samples, so we multiply by the area element delta_tau * delta_nu
    energy_deriv_tau = np.sum(np.abs(deriv_tau_prime)**2) * config.delta_tau * config.delta_nu
    energy_deriv_nu = np.sum(np.abs(deriv_nu_prime)**2) * config.delta_tau * config.delta_nu

    crb_tau_theory_list = []
    crb_nu_theory_list = []

    for snr_db in config.SNR_dB_vec:
        # Calculate alpha_tau_T
        alpha_tau_T = config.K_gain / (config.tau_T_true**2)

        # Calculate alpha_prime_tau_T
        alpha_prime_tau_T = -2 * alpha_tau_T / config.tau_T_true

        # Calculate echo signal power E_eco (E_tx = 1 due to normalization)
        E_eco = (alpha_tau_T * config.beta_T)**2

        # Calculate noise variance sigma_echo_2
        sigma_echo_2 = E_eco / (10**(snr_db / 10))

        # Calculate I_tautau (FIM term for tau)
        # E_tx is 1.0
        term1_tt = (2 * (config.beta_T**2) * 1.0 / sigma_echo_2) * (alpha_prime_tau_T**2)
        term2_tt = (2 * (alpha_tau_T**2) * (config.beta_T**2) / sigma_echo_2) * energy_deriv_tau

        # Third term for I_tautau
        # nu_grid: Doppler frequency for each bin k, centered around 0
        nu_grid = (np.arange(config.N) - config.N / 2) * config.delta_nu
        # nu_grid_shifted_by_true_doppler will be (1, N) for broadcasting
        nu_grid_shifted_by_true_doppler = nu_grid[np.newaxis, :] + config.nu_T_true

        # mu_lk_squared_approx = |alpha(tau_T) * beta_T * x_dd[l,k]|^2
        mu_lk_squared_approx = (alpha_tau_T * config.beta_T)**2 * np.abs(x_dd)**2

        # Sum over l and k. nu_grid_shifted_by_true_doppler**2 is (1,N), mu_lk_squared_approx is (M,N)
        # We need to ensure correct broadcasting.
        # (nu_grid_shifted_by_true_doppler**2) will broadcast its rows M times.
        sum_term_tt = np.sum((nu_grid_shifted_by_true_doppler**2) * mu_lk_squared_approx) * config.delta_tau * config.delta_nu
        term3_tt = (2 * (2 * np.pi)**2 / sigma_echo_2) * sum_term_tt

        I_tautau = term1_tt + term2_tt + term3_tt

        # Calculate I_nunu (FIM term for nu)
        term1_nn = (2 * (alpha_tau_T**2) * (config.beta_T**2) / sigma_echo_2) * energy_deriv_nu
        # E_eco already includes (alpha_tau_T * config.beta_T)**2 and E_tx=1
        term2_nn = (2 * (2 * np.pi * config.tau_T_true)**2 / sigma_echo_2) * E_eco
        I_nunu = term1_nn + term2_nn

        # Calculate CRBs
        crb_tau = 1 / I_tautau
        crb_nu = 1 / I_nunu

        crb_tau_theory_list.append(crb_tau)
        crb_nu_theory_list.append(crb_nu)

    return (crb_tau_theory_list, crb_nu_theory_list)
