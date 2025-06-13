# --- System & OTFS Parameters ---
M = 32                  # Delay bins
N = 32                  # Doppler bins
T_symbol = 1e-3         # OTFS symbol duration (s)
delta_f = 15e3          # Subcarrier spacing (Hz)

# --- Simulation Parameters ---
N_mc = 2000             # Number of Monte Carlo runs per SNR point
SNR_dB_vec = range(0, 31, 5) # SNR vector in dB (0, 5, 10, ..., 30)

# --- Ground Truth Target Parameters ---
delta_tau = 1 / (M * delta_f)
delta_nu = 1 / T_symbol
tau_T_true = 8.3 * delta_tau  # True target delay (s)
nu_T_true  = 5.7 * delta_nu   # True target Doppler (Hz)

# --- Channel & Waveform Parameters ---
beta_T    = 1.0           # Target reflectivity
K_gain = 1e-11          # Path gain constant
sigma_tau_pulse = 2     # Transmitted pulse width in delay bins
sigma_nu_pulse  = 2     # Transmitted pulse width in Doppler bins
