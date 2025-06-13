# --- System & OTFS Parameters ---
M = 16
N = 16
K = 2  # Number of users
P = 4  # Number of paths per user channel

# --- Simulation Parameters ---
SNR_dB_vec = range(0, 21, 2)  # SNR vector in dB (0, 2, 4, ..., 20)
N_bits = 100000 # Number of bits to transmit for each SNR point to get stable BER

# --- RSMA & Modulation Parameters ---
MODULATION = 'QPSK' # Assume QPSK for simplicity
# Potências dos pré-codificadores (fixas para este teste)
# A potência total do sinal será a soma de todas as potências de fluxo.
POWER_COMMON_STREAM = 0.5
POWER_PRIVATE_STREAM = 0.5 # Potência por usuário
