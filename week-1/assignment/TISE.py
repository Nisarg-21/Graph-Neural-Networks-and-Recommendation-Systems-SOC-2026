import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 500)

def simulate_wavefunction(x, E, V):
    if E > V:
        k = np.sqrt(2 * (E - V))
        return np.cos(k * x)
    elif E < V:
        kappa = np.sqrt(2 * (V - E))
        return np.exp(-kappa * x)
    else:
        return np.ones_like(x)

psi_A = simulate_wavefunction(x, E=12, V=2)
psi_B = simulate_wavefunction(x, E=2,  V=7)
psi_C = simulate_wavefunction(x, E=5,  V=5)

plt.figure(figsize=(10, 6))
plt.plot(x, psi_A, color='blue',  linestyle='-',  label='Case A — Oscillatory (E=12, V=2)')
plt.plot(x, psi_B, color='red',   linestyle='--', label='Case B — Decaying (E=2, V=7)')
plt.plot(x, psi_C, color='green', linestyle=':',  label='Case C — Threshold (E=5, V=5)')
plt.title("1D Time-Independent Schrodinger Equation — Wavefunction Simulation")
plt.xlabel("x")
plt.ylabel("psi(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()