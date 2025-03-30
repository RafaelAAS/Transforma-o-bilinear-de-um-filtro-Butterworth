import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# 1. Definição do filtro contínuo Hc(s)
num_s = [0.20238]
den_s = np.polymul([1, 0.3996, 0.5871], np.polymul([1, 1.0836, 0.5871], [1, 1.4802, 0.5871]))
Hc_s = signal.TransferFunction(num_s, den_s)

# 2. Definição do filtro discreto H(z)
num_z = [0.0007378, 0.0044268, 0.011712, 0.015616, 0.011712, 0.0044268, 0.0007378]
den_z = np.polymul(
    [1, -1.2686, 0.7051], np.polymul([1, -1.0106, 0.3583], [1, -0.9044, 0.2155])
)
Hz_z = signal.dlti(num_z, den_z, dt=1)  # Sistema discreto com Td = 1

# 3. Gerar os diagramas de Bode

# Filtro contínuo
w, mag_s, phase_s = signal.bode(Hc_s)

# Filtro discreto (convertendo para frequências normalizadas)
w_z, mag_z, phase_z = signal.dbode(Hz_z)

# Criando figuras separadas para deixar claro qual é qual
fig, axs = plt.subplots(2, 2, figsize=(12, 6))

# Magnitude - Filtro Contínuo H(s)
axs[0, 0].semilogx(w, mag_s, label="Hc(s) - Contínuo", color="b")
axs[0, 0].set_ylabel("Magnitude (dB)")
axs[0, 0].set_title("Diagrama de Bode - Magnitude (Hc(s))")
axs[0, 0].legend()
axs[0, 0].grid()

# Fase - Filtro Contínuo H(s)
axs[1, 0].semilogx(w, phase_s, label="Hc(s) - Contínuo", color="b")
axs[1, 0].set_ylabel("Fase (graus)")
axs[1, 0].set_xlabel("Frequência (rad/s)")
axs[1, 0].set_title("Diagrama de Bode - Fase (Hc(s))")
axs[1, 0].legend()
axs[1, 0].grid()

# Magnitude - Filtro Discreto H(z)
axs[0, 1].semilogx(w_z, mag_z, label="H(z) - Discreto", color="r", linestyle="dashed")
axs[0, 1].set_ylabel("Magnitude (dB)")
axs[0, 1].set_title("Diagrama de Bode - Magnitude (H(z))")
axs[0, 1].legend()
axs[0, 1].grid()

# Fase - Filtro Discreto H(z)
axs[1, 1].semilogx(w_z, phase_z, label="H(z) - Discreto", color="r", linestyle="dashed")
axs[1, 1].set_ylabel("Fase (graus)")
axs[1, 1].set_xlabel("Frequência (rad/s)")
axs[1, 1].set_title("Diagrama de Bode - Fase (H(z))")
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()
plt.show()
