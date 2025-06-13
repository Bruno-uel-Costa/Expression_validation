import config
import analytical_crb
import numpy as np
import matplotlib.pyplot as plt

# --- Pre-calculate x_dd (normalized) ---
# These coordinates are for the generation of x_dd, centered around (0,0)
_L_coords = np.arange(config.M) - config.M/2
_K_coords = np.arange(config.N) - config.N/2
# Meshgrid for x_dd generation. 'ij' indexing means L changes along rows, K along columns.
_ll_grid, _kk_grid = np.meshgrid(_L_coords, _K_coords, indexing='ij')
_x_dd_unnormalized = np.exp(-((_ll_grid**2 / config.sigma_tau_pulse**2) + \
                             (_kk_grid**2 / config.sigma_nu_pulse**2)))
_E_tx_unnormalized = np.sum(np.abs(_x_dd_unnormalized)**2)
X_DD_NORMALIZED = _x_dd_unnormalized / np.sqrt(_E_tx_unnormalized)


def generate_signal(cfg, tau_target, nu_target, x_dd_input):
    """
    Generates the received signal mu[l,k] in the Delay-Doppler domain.

    Args:
        cfg: Configuration object.
        tau_target: True target delay (s).
        nu_target: True target Doppler (Hz).
        x_dd_input: The normalized transmitted waveform in DD domain (M, N).
                    Assumed to be centered at (0,0) delay/Doppler shift effectively.

    Returns:
        mu_lk: The received signal in DD domain (M, N).
    """
    # Calculate discrete shifts l_T and k_T (bin indices for the target)
    # These represent the closest discrete bin to the true tau_target and nu_target
    l_T = np.round(tau_target / cfg.delta_tau).astype(int)
    k_T = np.round(nu_target / cfg.delta_nu).astype(int)

    # Apply circular shift to x_dd_input to place the peak at (l_T, k_T)
    # x_dd_input is an (M,N) array where the pulse is effectively centered.
    # np.roll shifts elements. If l_T, k_T are positive, it shifts "down" and "right".
    # The indexing of x_dd_input is 0..M-1, 0..N-1.
    # The center of the pulse in x_dd_input is at (M/2, N/2) if M,N are even.
    # A shift of l_T means the element originally at (M/2, N/2) moves to (M/2+l_T, N/2+k_T) mod (M,N).
    x_dd_shifted = np.roll(x_dd_input, (l_T, k_T), axis=(0, 1))

    # Calculate the delay grid tau_l for phase calculation
    # This grid represents the physical delay values for each 'l' bin.
    # If x_dd_input is centered (e.g. generated with M/2 shifts),
    # then the 'l' index 0 corresponds to -M/2*delta_tau.
    # The phase term involves (tau[l] - tau_target).
    # tau_l_physical = (np.arange(cfg.M) - cfg.M/2) * cfg.delta_tau
    # This is consistent with _L_coords used for generating x_dd_input.
    tau_l_centered = (np.arange(cfg.M) - cfg.M/2) * cfg.delta_tau

    # Calculate the phase term e^(j * 2 * pi * nu_target * (tau_l - tau_target))
    # This phase is applied across the delay dimension (l).
    phase_vals_at_l = np.exp(1j * 2 * np.pi * nu_target * (tau_l_centered - tau_target))
    phase_l_expanded = phase_vals_at_l[:, np.newaxis]  # Shape (M, 1) to broadcast over N columns

    # Calculate alpha(tau_T)
    epsilon = 1e-16 # To prevent division by zero if tau_target is ever zero
    alpha_val = cfg.K_gain / (tau_target**2 + epsilon)

    # Calculate mu[l,k] = alpha(tau_T) * beta_T * x_dd_shifted[l,k] * phase_term[l]
    mu_lk = alpha_val * cfg.beta_T * x_dd_shifted * phase_l_expanded
    return mu_lk


def ml_estimator(cfg, y_received, x_dd_input, search_grid_tau, search_grid_nu):
    """
    Maximum Likelihood Estimator for delay and Doppler.

    Args:
        cfg: Configuration object.
        y_received: The received signal in DD domain (M, N), including noise.
        x_dd_input: The normalized transmitted waveform in DD domain (M, N).
        search_grid_tau: 1D array of delay values to search over.
        search_grid_nu: 1D array of Doppler values to search over.

    Returns:
        tau_hat: Estimated delay.
        nu_hat: Estimated Doppler.
    """
    max_corr_squared = -1  # Initialize with a value that will be overcome
    tau_hat = 0
    nu_hat = 0

    for tau_hip in search_grid_tau:
        for nu_hip in search_grid_nu:
            # Generate template signal for the current hypothesis
            # This template is noise-free and based on the hypothesized parameters
            mu_hip = generate_signal(cfg, tau_hip, nu_hip, x_dd_input)

            # Calculate correlation: |sum(conj(mu_hip) * y_received)|
            # The problem statement implies maximizing the squared magnitude of the correlation.
            correlation = np.abs(np.sum(np.conj(mu_hip) * y_received))
            correlation_squared = correlation**2 # Square the magnitude

            if correlation_squared > max_corr_squared:
                max_corr_squared = correlation_squared
                tau_hat = tau_hip
                nu_hat = nu_hip

    return tau_hat, nu_hat

# Calculate theoretical CRBs
crb_tau_theory, crb_nu_theory = analytical_crb.calculate_analytical_crb(config)

# Initialize lists for Mean Squared Errors from simulation
MSE_tau = []
MSE_nu = []

UPSAMPLING_FACTOR = 10

# --- Define Search Grid for ML Estimator ---
# Search grid covers the full range of discrete delays/Dopplers, upsampled for finer search.
# search_tau_indices = np.arange(config.M) # Old version
# search_nu_indices = np.arange(config.N)  # Old version

# Convert indices to physical values, centered around 0.
# The new grid is M*UPSAMPLING_FACTOR points long, effectively dividing delta_tau by UPSAMPLING_FACTOR.
# The range of search remains similar: from approx -M/2*delta_tau to +M/2*delta_tau.
search_grid_tau = (np.arange(config.M * UPSAMPLING_FACTOR) / UPSAMPLING_FACTOR - config.M/2) * config.delta_tau
search_grid_nu  = (np.arange(config.N * UPSAMPLING_FACTOR) / UPSAMPLING_FACTOR - config.N/2) * config.delta_nu


# --- Monte Carlo Simulation Loop ---
for snr_db in config.SNR_dB_vec:
    print(f"Simulating SNR: {snr_db} dB") # Progress indicator

    # Calculate E_eco (average energy of the echo signal without noise)
    # This should be based on the true target parameters.
    # alpha_val_true is K_gain / tau_T_true^2
    alpha_val_true = config.K_gain / (config.tau_T_true**2 + 1e-16) # Epsilon for stability
    # E_eco = |alpha_val_true * beta_T|^2 * E_tx. Since X_DD_NORMALIZED has E_tx = 1.
    E_eco = (alpha_val_true * config.beta_T)**2

    # Calculate noise variance sigma_echo_2 for the complex noise y_received = mu_true + noise
    # SNR = E_eco / sigma_echo_2  => sigma_echo_2 = E_eco / SNR_linear
    sigma_echo_2 = E_eco / (10**(snr_db / 10))

    errors_tau_sq_current_snr = []
    errors_nu_sq_current_snr = []

    for i in range(config.N_mc):
        if (i + 1) % 200 == 0: # Print progress within the MC loop
            print(f"  MC run {i + 1}/{config.N_mc} for SNR {snr_db} dB")

        # Generate true signal mu_true using the globally defined X_DD_NORMALIZED
        mu_true = generate_signal(config, config.tau_T_true, config.nu_T_true, X_DD_NORMALIZED)

        # Generate complex Gaussian noise
        # Variance of complex noise is sigma_echo_2.
        # Real and imaginary parts each have variance sigma_echo_2 / 2.
        noise_real = np.random.normal(0, np.sqrt(sigma_echo_2 / 2), (config.M, config.N))
        noise_imag = np.random.normal(0, np.sqrt(sigma_echo_2 / 2), (config.M, config.N))
        noise = noise_real + 1j * noise_imag

        # Received signal with noise
        y_received = mu_true + noise

        # Estimate parameters using ML estimator
        # The estimator uses X_DD_NORMALIZED as its template basis.
        (tau_hat, nu_hat) = ml_estimator(config, y_received, X_DD_NORMALIZED,
                                         search_grid_tau, search_grid_nu)

        # Store squared error for this Monte Carlo run
        errors_tau_sq_current_snr.append((tau_hat - config.tau_T_true)**2)
        errors_nu_sq_current_snr.append((nu_hat - config.nu_T_true)**2)

    # Calculate MSE for the current SNR and append to global lists
    MSE_tau.append(np.mean(errors_tau_sq_current_snr))
    MSE_nu.append(np.mean(errors_nu_sq_current_snr))

if __name__ == "__main__":
    print("Simulation loop finished.")
    print("Theoretical CRB for tau:", crb_tau_theory)
    print("Theoretical CRB for nu:", crb_nu_theory)
    print("Simulated MSE_tau:", MSE_tau)
    print("Simulated MSE_nu:", MSE_nu)

    # --- Plotting Results ---
    plt.figure(figsize=(10, 7))

    # Plot for Delay Estimation
    plt.subplot(2, 1, 1)
    plt.semilogy(config.SNR_dB_vec, crb_tau_theory, 'b-', label='CRB_tau (Theory)')
    plt.semilogy(config.SNR_dB_vec, MSE_tau, 'ro--', label='MSE_tau (Simulated)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('MSE / CRB (log scale)')
    plt.title('Delay Estimation Performance vs. SNR')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Plot for Doppler Estimation
    plt.subplot(2, 1, 2)
    plt.semilogy(config.SNR_dB_vec, crb_nu_theory, 'g-', label='CRB_nu (Theory)')
    plt.semilogy(config.SNR_dB_vec, MSE_nu, 'ms--', label='MSE_nu (Simulated)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('MSE / CRB (log scale)')
    plt.title('Doppler Estimation Performance vs. SNR')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.savefig('crb_vs_mse_simulation.png') # Save the plot to a file
    print("Plot saved as crb_vs_mse_simulation.png")
    # plt.show() # Optionally show plot if running in an interactive environment
