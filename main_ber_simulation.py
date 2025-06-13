import config_sinr
import analytical_sinr
import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt # Will be used later for plotting

# --- Configuration ---
# Using config_sinr module directly for its attributes.
# Functions from analytical_sinr will take config_sinr as an argument.

# --- Theoretical SINR Calculation ---
# calculate_theoretical_sinr is expected to return SINR per symbol.
sinr_theory_vec_symbol_linear = analytical_sinr.calculate_theoretical_sinr(config_sinr)
sinr_theory_vec_symbol_linear = np.array(sinr_theory_vec_symbol_linear)

# --- Theoretical BER Calculation for QPSK ---
# BER for QPSK = Q(sqrt(2 * SINR_b)) where SINR_b is SINR per bit.
# For QPSK, SINR_symbol = 2 * SINR_b, so SINR_b = SINR_symbol / 2.
# BER = Q(sqrt(SINR_symbol)) = 0.5 * erfc(sqrt(SINR_symbol) / sqrt(2)) = 0.5 * erfc(sqrt(SINR_symbol / 2.0))
# This matches the formula given in the problem description,
# implying sinr_theory_vec_symbol_linear is indeed SINR_symbol.
# Original interpretation: ber_theory = 0.5 * erfc(np.sqrt(sinr_theory_vec_symbol_linear / 2.0))
# This formula is equivalent to 0.5 * erfc(sqrt(SINR_bit_actual)), which is the standard QPSK BER.

# New interpretation from user request:
# Step 1: Calculate SINR per bit
# Assuming sinr_theory_vec_symbol_linear from analytical_sinr.py is SINR_symbol
# For QPSK, there are 2 bits per symbol.
sinr_theory_vec_bit_actual = sinr_theory_vec_symbol_linear / 2.0

# Step 2: Apply the formula 0.5 * erfc(sqrt(X/2)), where X is now SINR_bit_actual.
# This means the formula structure from the problem "0.5 * erfc(sqrt(argument / 2))"
# is now taking SINR_bit_actual as the 'argument'.
ber_theory = 0.5 * erfc(np.sqrt(sinr_theory_vec_bit_actual / 2.0))
# This effectively becomes 0.5 * erfc(np.sqrt( (SINR_symbol / 2) / 2.0 )) = 0.5 * erfc(np.sqrt(SINR_symbol / 4.0))

# Add a small epsilon to prevent log(0) if any BER is zero (e.g., for plotting later)
ber_theory = np.maximum(ber_theory, 1e-10)

# --- Initialize list for simulated BER ---
ber_simulated = [] # This will store BER values from the simulation loop


# --- QPSK Modulation and Demodulation Functions ---
def qpsk_mapper(bits):
    """
    Maps a sequence of bits to QPSK symbols using Gray coding.
    Output symbols have average power 1.
    Mapping:
        00 -> (1+1j)/sqrt(2)
        01 -> (1-1j)/sqrt(2)
        10 -> (-1+1j)/sqrt(2)
        11 -> (-1-1j)/sqrt(2)
    """
    if bits.size % 2 != 0:
        raise ValueError("Number of bits must be a multiple of 2 for QPSK.")

    symbols = np.zeros(bits.size // 2, dtype=complex)
    sqrt2_inv = 1.0 / np.sqrt(2.0)

    for i in range(0, bits.size, 2):
        b1 = bits[i]    # First bit (MSB for this mapping logic)
        b0 = bits[i+1]  # Second bit (LSB for this mapping logic)

        val_I = 0
        val_Q = 0

        # b1 determines the sign of the Real part (0 -> Positive Real)
        if b1 == 0:
            val_I = 1
        else: # b1 == 1
            val_I = -1

        # b0 determines the sign of the Imaginary part (0 -> Positive Imag)
        if b0 == 0:
            val_Q = 1
        else: # b0 == 1
            val_Q = -1

        symbols[i//2] = (val_I + 1j*val_Q)

    return symbols * sqrt2_inv

def qpsk_demapper(symbols):
    """
    Demaps QPSK symbols (assumed to be Gray coded and normalized) back to bits.
    Assumes symbols are centered around (+/-1 +/-1j)/sqrt(2).
    Decision boundaries are Re(symbol)=0 and Im(symbol)=0.
    """
    bits = np.zeros(symbols.size * 2, dtype=int)

    for i in range(symbols.size):
        sym_real = np.real(symbols[i])
        sym_imag = np.imag(symbols[i])

        idx = i * 2 # Index for the first bit of the symbol

        # Demapping based on the Gray mapping in qpsk_mapper:
        # Real part > 0 implies b1 was 0
        # Imag part > 0 implies b0 was 0

        if sym_real > 0:
            bits[idx] = 0 # b1 = 0
        else:
            bits[idx] = 1 # b1 = 1

        if sym_imag > 0:
            bits[idx+1] = 0 # b0 = 0
        else:
            bits[idx+1] = 1 # b0 = 1

    return bits

# --- Print theoretical values for verification ---

# --- Main Simulation Loop ---
# Target user for BER calculation will be user 0
TARGET_USER_IDX = 0

for snr_db in config_sinr.SNR_dB_vec:
    print(f"Simulating SNR: {snr_db} dB...")

    # Calculate sigma_n2 (noise variance per DFT bin) consistent with analytical_sinr.py
    # This definition relates SNR to the power density of one private stream.
    power_density_private_stream = config_sinr.POWER_PRIVATE_STREAM / (config_sinr.M * config_sinr.N)
    sigma_n2 = power_density_private_stream / (10**(snr_db / 10.0))

    # N_bits is total bits to transmit for the target user's private stream to get stable BER.
    # Number of QPSK symbols needed for the target stream: N_bits / 2
    num_qpsk_symbols_per_stream = config_sinr.N_bits // 2

    # Ensure N_bits is even for QPSK mapping
    if config_sinr.N_bits % 2 != 0:
        # This check should ideally be at the beginning or tied to config validation
        raise ValueError("config_sinr.N_bits must be even for QPSK.")

    # Each M*N grid carries M*N QPSK symbols for each stream if fully loaded.
    symbols_per_grid = config_sinr.M * config_sinr.N

    # Number of grids needed to transmit N_bits for the target user's private stream.
    # Each grid contributes 'symbols_per_grid' symbols to the target user's private stream.
    num_grids_to_simulate = (num_qpsk_symbols_per_stream + symbols_per_grid - 1) // symbols_per_grid

    print(f"  Total QPSK symbols for target user's private stream: {num_qpsk_symbols_per_stream}")
    print(f"  Symbols per M*N grid (per stream): {symbols_per_grid}")
    print(f"  Number of M*N grids to simulate for N_bits: {num_grids_to_simulate}")

    # Store all transmitted bits for the target user's private stream for final BER calculation
    # Total bits for target user = num_grids_to_simulate * symbols_per_grid * 2
    # This might be slightly more than N_bits if N_bits is not a multiple of (symbols_per_grid*2)
    total_bits_for_target_user = num_grids_to_simulate * symbols_per_grid * 2
    all_tx_bits_target_user_private = np.zeros(total_bits_for_target_user, dtype=int)
    all_rx_bits_target_user_private = np.zeros(total_bits_for_target_user, dtype=int)

    # Generate channel realizations for all K users
    # These channels are assumed static for the duration of N_bits transmission for this SNR point.
    H_dd_users = [analytical_sinr.generate_channel_dd(config_sinr.M, config_sinr.N, config_sinr.P, user_idx_for_seed=k) for k in range(config_sinr.K)]
    Lambda_H_users_dft = [np.fft.fft2(H_dd_users[k]) for k in range(config_sinr.K)]

    for grid_idx in range(num_grids_to_simulate):
        if (grid_idx+1) % 10 == 0 or grid_idx == num_grids_to_simulate -1 :
             print(f"    Processing grid {grid_idx+1}/{num_grids_to_simulate} for SNR {snr_db} dB")

        # --- TRANSMITTER ---
        # 1. Generate random bits for this grid for all streams
        # Bits for common stream (symbols_per_grid * 2 bits)
        bits_common_grid = np.random.randint(0, 2, symbols_per_grid * 2)
        # Bits for private streams of all K users (each user has symbols_per_grid * 2 bits)
        bits_private_users_grid = [np.random.randint(0, 2, symbols_per_grid * 2) for _ in range(config_sinr.K)]

        # Store transmitted bits for the target user (user TARGET_USER_IDX)
        start_idx_bits = grid_idx * symbols_per_grid * 2
        end_idx_bits = start_idx_bits + symbols_per_grid * 2
        all_tx_bits_target_user_private[start_idx_bits:end_idx_bits] = bits_private_users_grid[TARGET_USER_IDX]

        # 2. Map bits to QPSK symbols (these are 1D arrays of symbols, length symbols_per_grid)
        s_common_1d = qpsk_mapper(bits_common_grid)
        s_private_users_1d = [qpsk_mapper(bits_private_users_grid[k]) for k in range(config_sinr.K)]

        # Reshape symbols to M x N grids (symbols fill the grid in C-style reshape order)
        S_common_dft_grid = s_common_1d.reshape((config_sinr.M, config_sinr.N))
        S_private_users_dft_grid = [s_private_users_1d[k].reshape((config_sinr.M, config_sinr.N)) for k in range(config_sinr.K)]

        # 3. Apply precoders (amplitude scaling for normalized symbols to achieve desired power density)
        # Power is total power for the stream, so density is POWER / (M*N)
        # Amplitude is sqrt(density) = sqrt(POWER / (M*N))
        amp_common = np.sqrt(config_sinr.POWER_COMMON_STREAM / (config_sinr.M * config_sinr.N))
        amp_private = np.sqrt(config_sinr.POWER_PRIVATE_STREAM / (config_sinr.M * config_sinr.N))

        # Precode (apply amplitude)
        Pc_S_common_dft_grid = amp_common * S_common_dft_grid # Precoding for common stream
        Pp_S_private_users_dft_grid = [amp_private * S_private_users_dft_grid[k] for k in range(config_sinr.K)] # Precoding for private streams

        # 4. Form the aggregate transmitted signal X_dft_grid in DFT domain (sum of all streams)
        X_dft_grid = Pc_S_common_dft_grid # Start with common stream
        for k in range(config_sinr.K):
            X_dft_grid += Pp_S_private_users_dft_grid[k] # Add private streams

        # --- CHANNEL + NOISE ---
        # Received signal for the target user (user TARGET_USER_IDX) in DFT domain
        # Y_dft_user_k = Lambda_H_k * X_dft_grid + Noise
        # sigma_n2 is variance per complex DFT bin (total variance for real + imag parts)
        noise_real_part = np.random.normal(0, np.sqrt(sigma_n2 / 2.0), (config_sinr.M, config_sinr.N))
        noise_imag_part = np.random.normal(0, np.sqrt(sigma_n2 / 2.0), (config_sinr.M, config_sinr.N))
        Noise_dft_grid = noise_real_part + 1j * noise_imag_part

        # Signal received by TARGET_USER_IDX
        Y_dft_user0_grid = Lambda_H_users_dft[TARGET_USER_IDX] * X_dft_grid + Noise_dft_grid

        # --- RECEIVER (for TARGET_USER_IDX's private stream) ---

        # 1. Genie-Aided Successive Interference Cancellation (SIC) for the common stream
        # User knows its own channel Lambda_H_users_dft[TARGET_USER_IDX].
        # User decodes common stream S_common_dft_grid (genie-aided: assumes perfect knowledge of S_common_dft_grid).
        # amp_common is also known.
        interference_common_stream_dft = Lambda_H_users_dft[TARGET_USER_IDX] * amp_common * S_common_dft_grid
        Y_prime_dft_user0_grid = Y_dft_user0_grid - interference_common_stream_dft

        # 2. BLMMSE Filter for the private stream of TARGET_USER_IDX

        # lambda_R_Ys_pk: Cross-correlation of Y' and s_pk (private symbols of target user)
        # This is E[Y'_dft_user0_grid * conj(s_p_target_dft_grid)] where s_p_target_dft_grid are the symbols to be estimated.
        # Y'_dft_user0_grid = Lambda_H_target * (amp_private * S_private_target + sum_{j!=target} amp_private * S_private_j) + Noise
        # The desired part is Lambda_H_target * amp_private * S_private_target.
        # So, lambda_R_Ys_p_target = Lambda_H_users_dft[TARGET_USER_IDX] * amp_private
        # (assuming S_private_target has unit power symbols, which it does due to qpsk_mapper)
        lambda_R_Ys_p_target = Lambda_H_users_dft[TARGET_USER_IDX] * amp_private

        # lambda_R_YprimeYprime_pk: Auto-correlation of Y' after SIC of common stream
        # This includes interference from other users' private streams + noise.
        # Y'_dft_user0_grid = sum_{j=0..K-1} Lambda_H_target * amp_private * S_private_j_grid + Noise_dft_grid
        # (if we re-index private streams for Y', or rather, Y' = H_target * (amp_p*S_p_target + sum_{j!=target} amp_p*S_p_j) + Noise)
        # The power of (Lambda_H_target * amp_private * S_private_j_grid) is |Lambda_H_target|^2 * amp_private^2

        interference_power_density_from_others = np.zeros((config_sinr.M, config_sinr.N), dtype=float)
        for j_user_idx in range(config_sinr.K):
            if j_user_idx == TARGET_USER_IDX:
                continue # Skip the target user itself
            interference_power_density_from_others += np.abs(Lambda_H_users_dft[TARGET_USER_IDX])**2 * (amp_private**2)

        lambda_R_YprimeYprime_p_target = interference_power_density_from_others + sigma_n2

        epsilon = 1e-18 # For numerical stability

        # BLMMSE filter weights (per DFT bin)
        W_p_target_blmmse_dft = np.conj(lambda_R_Ys_p_target) / (lambda_R_YprimeYprime_p_target + epsilon)

        # 3. Apply filter to estimate private stream symbols for TARGET_USER_IDX
        S_p_target_hat_dft_grid = W_p_target_blmmse_dft * Y_prime_dft_user0_grid

        # 4. Demodulate symbols to bits
        # Reshape the M x N grid of estimated symbols to a 1D array
        s_p_target_hat_1d = S_p_target_hat_dft_grid.reshape(-1) # Flatten to 1D array

        # Demap to bits
        rx_bits_private_target_user_grid = qpsk_demapper(s_p_target_hat_1d)

        # Store received bits for the target user
        all_rx_bits_target_user_private[start_idx_bits:end_idx_bits] = rx_bits_private_target_user_grid

    # --- Error Counting (after all grids for this SNR) ---
    # total_bits_for_target_user was calculated before the grid loop
    error_count_target_user = np.sum(all_tx_bits_target_user_private != all_rx_bits_target_user_private)

    current_ber = error_count_target_user / total_bits_for_target_user
    ber_simulated.append(current_ber)
    print(f"  SNR {snr_db} dB - Simulated BER for User {TARGET_USER_IDX}: {current_ber:.2e} (Errors: {error_count_target_user}/{total_bits_for_target_user})")

# --- Plotting Results ---
plt.figure(figsize=(10, 7))
plt.semilogy(config_sinr.SNR_dB_vec, ber_theory, 'b-', label='Theoretical BER (QPSK from Analytical SINR)')
plt.semilogy(config_sinr.SNR_dB_vec, ber_simulated, 'ro--', label=f'Simulated BER (User {TARGET_USER_IDX} Private Stream)')

plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('RSMA BER Performance: Theoretical vs. Simulated (BLMMSE Receiver)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.ylim(bottom=1e-6, top=1.0) # Set y-axis limits for typical BER plot
plt.savefig('rsma_ber_simulation.png')
print("Plot saved as rsma_ber_simulation.png")
# plt.show() # Optional: uncomment to display plot if running interactively

if __name__ == "__main__":
    print(f"Configured SNR (dB) vector: {list(config_sinr.SNR_dB_vec)}")
    print(f"Theoretical Symbol SINR (linear): {sinr_theory_vec_symbol_linear}")
    print(f"Theoretical Bit SINR (actual, SINR_symbol/2): {sinr_theory_vec_bit_actual}")
    print(f"Theoretical BER for QPSK (calculated as 0.5*erfc(sqrt(SINR_bit_actual/2))): {ber_theory}")

# Simulation loop and plotting will be added in subsequent steps.
