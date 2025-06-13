import numpy as np
import config_sinr # For default values if not passed, but config is an argument

def generate_channel_dd(M, N, P, user_idx_for_seed=0):
    """
    Generates an M x N channel matrix in the Delay-Doppler domain.

    Args:
        M (int): Number of delay bins.
        N (int): Number of Doppler bins.
        P (int): Number of paths.
        user_idx_for_seed (int): Seed for reproducibility (e.g., user index).

    Returns:
        np.ndarray: M x N complex channel matrix (h_dd).
    """
    np.random.seed(user_idx_for_seed) # For reproducibility
    h_dd = np.zeros((M, N), dtype=complex)

    for _ in range(P):
        # Generate random integer delays and Dopplers
        l_p = np.random.randint(0, M)
        k_p = np.random.randint(0, N)

        # Generate random complex gains (standard complex normal)
        alpha_p = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)

        h_dd[l_p, k_p] += alpha_p

    return h_dd

def calculate_theoretical_sinr(config):
    """
    Calculates the theoretical SINR for the private stream of a user in an RSMA system.

    Args:
        config: A configuration object with parameters like M, N, K, P,
                SNR_dB_vec, POWER_COMMON_STREAM, POWER_PRIVATE_STREAM.

    Returns:
        list: A list of theoretical SINR values (linear scale) for each SNR_dB point.
    """
    sinr_theory_vec = []
    epsilon = 1e-18 # Small value to prevent division by zero

    # Generate one representative channel realization for a user (e.g., user 0 for seed)
    # This channel is specific to one user, let's call it user k.
    h_k_dd = generate_channel_dd(config.M, config.N, config.P, user_idx_for_seed=0)

    # Transform to DFT domain (per-subcarrier channel gain)
    # lambda_Hk[m,n] is the channel gain for user k on subcarrier (m,n)
    lambda_Hk_user_k = np.fft.fft2(h_k_dd) # This is an M x N matrix

    # Power density of the private stream of user k, per DFT bin (assuming symbols s_pk have unit power)
    # config.POWER_PRIVATE_STREAM is total power for that stream.
    power_density_private_stream_k = config.POWER_PRIVATE_STREAM / (config.M * config.N)

    for snr_db in config.SNR_dB_vec:
        # Calculate noise variance (sigma_n2) per DFT bin
        # The SNR is defined with respect to the power density of the desired private stream *before* channel gain.
        # SNR_per_bin_pre_channel = power_density_private_stream_k / sigma_n2_per_dft_bin
        # So, sigma_n2_per_dft_bin = power_density_private_stream_k / (10**(snr_db / 10))
        sigma_n2_per_dft_bin = power_density_private_stream_k / (10**(snr_db / 10))

        # lambda_R_Ys_k: Cross-correlation of received signal (Y') and desired private symbols (s_k) for user k, per DFT bin.
        # This is lambda_Hk_user_k * (precoder_gain_for_s_k).
        # Precoder for s_k is sqrt(power_density_private_stream_k) if s_k symbols have unit power.
        lambda_R_Ys_k = lambda_Hk_user_k * np.sqrt(power_density_private_stream_k)

        # lambda_R_YY_prime: Auto-correlation of Y' (signal after SIC of common stream), per DFT bin.
        # This includes interference from other users' private streams and thermal noise.
        # Assumes other K-1 users have statistically similar channels (using |lambda_Hk_user_k|^2 as representative for others).
        # Assumes POWER_PRIVATE_STREAM is the same for all users.
        # Assumes perfect SIC of the common stream.

        # Interference power density from K-1 other users' private streams
        # Each interfering user contributes |lambda_H_interferer|^2 * power_density_private_stream_interferer
        # For simplicity, assuming interferer channels and powers are similar to user k's
        interference_power_density_others = (config.K - 1) * np.abs(lambda_Hk_user_k)**2 * power_density_private_stream_k

        lambda_R_YY_prime = interference_power_density_others + sigma_n2_per_dft_bin

        # Add epsilon to the denominator term to prevent division by zero
        lambda_R_YY_prime_reg = lambda_R_YY_prime + epsilon

        # Calculate SINR components per subcarrier (per DFT bin)
        # term_mn = | E[Y'_mn * s_k_mn*] |^2 / E[|Y'_mn - E[Y'_mn * s_k_mn*]s_k_mn|^2]
        # This simplifies to |lambda_R_Ys_k|^2 / lambda_R_YY_prime under Gaussian signaling assumptions for MMSE.
        # The formula provided is J_k = mean(1 - |lambda_R_Ys_k|^2 / lambda_R_YY_prime)
        # So, term_per_subcarrier is |lambda_R_Ys_k|^2 / lambda_R_YY_prime
        term_per_subcarrier = np.abs(lambda_R_Ys_k)**2 / lambda_R_YY_prime_reg

        # J_k = mean(1 - term_per_subcarrier) over all subcarriers (M*N DFT bins)
        # Note: if term_per_subcarrier can be > 1 (e.g. due to very low noise/interference for a bin),
        # then 1 - term_per_subcarrier can be negative. J_k is an average.
        J_k = np.mean(1.0 - term_per_subcarrier)

        # SINR_k_final = (1/J_k) - 1
        # Ensure J_k is not zero (or too close to zero leading to huge SINR)
        # Also, if J_k is >= 1 (meaning average term_per_subcarrier <=0), SINR would be negative or undefined.
        # This implies that the term_per_subcarrier should be <= 1 for J_k to be positive.
        # If J_k is very small (term_per_subcarrier close to 1), SINR is high.
        # If J_k approaches 1 (term_per_subcarrier close to 0), SINR approaches 0.
        if J_k <= epsilon or J_k >= 1.0 - epsilon: # handles J_k near 0 or J_k near 1 (problematic cases)
            # If J_k is near 0, SINR is very high.
            # If J_k is near 1 (or >1), it means avg signal component is weak or negative, so SINR is effectively 0 or very low.
            # A robust way is to cap J_k or handle resulting SINR.
            # Let's assume term_per_subcarrier is generally <=1.
            # If J_k is very small (e.g. 1e-10), 1/J_k is large, SINR is large. This is fine.
            # If J_k becomes negative (avg term > 1), or J_k is 1 (avg term = 0), then SINR is ill-defined or 0.
            # Let's cap J_k slightly away from 0 if it's negative or zero for stability,
            # and if J_k is >=1, the SINR is effectively 0.
            if J_k >= 1.0 - epsilon :
                 sinr_k_final = 0.0
            elif J_k <= epsilon: # J_k is very small positive or zero/negative
                 sinr_k_final = 1.0 / (epsilon) -1 # very large SINR
            else:
                 sinr_k_final = (1.0 / J_k) - 1.0

        else:
            sinr_k_final = (1.0 / J_k) - 1.0

        # Ensure SINR is not negative (can happen if J_k > 1)
        if sinr_k_final < 0:
            sinr_k_final = 0.0

        sinr_theory_vec.append(sinr_k_final)

    return sinr_theory_vec

```
