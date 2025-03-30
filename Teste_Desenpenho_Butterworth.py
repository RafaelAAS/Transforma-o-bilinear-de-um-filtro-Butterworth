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

# 3. Resposta ao Impulso e ao Degrau

# Tempo para simulações
t_s = np.linspace(0, 10, 1000)  # Tempo contínuo
t_z = np.arange(0, 50, 1)       # Tempo discreto

# Resposta ao Impulso
t_imp_s, y_imp_s = signal.impulse(Hc_s, T=t_s)
t_imp_z, y_imp_z = signal.dimpulse(Hz_z, n=50)

# Resposta ao Degrau
t_step_s, y_step_s = signal.step(Hc_s, T=t_s)
t_step_z, y_step_z = signal.dstep(Hz_z, n=50)

# Plotando as respostas
fig, axs = plt.subplots(2, 2, figsize=(12, 6))

# Resposta ao Impulso
axs[0, 0].plot(t_imp_s, y_imp_s, label="Hc(s) - Contínuo", color="b")
axs[0, 0].stem(t_imp_z, np.squeeze(y_imp_z), label="H(z) - Discreto", linefmt="r-", markerfmt="ro", basefmt="r-")
axs[0, 0].set_title("Resposta ao Impulso")
axs[0, 0].legend()
axs[0, 0].grid()

# Resposta ao Degrau
axs[1, 0].plot(t_step_s, y_step_s, label="Hc(s) - Contínuo", color="b")
axs[1, 0].stem(t_step_z, np.squeeze(y_step_z), label="H(z) - Discreto", linefmt="r-", markerfmt="ro", basefmt="r-")
axs[1, 0].set_title("Resposta ao Degrau")
axs[1, 0].legend()
axs[1, 0].grid()

# 4. Resposta a um Sinal Senoidal

# Sinal senoidal dentro da faixa de passagem (ω = 0.1π para H(z))
fs = 1  # Taxa de amostragem do filtro discreto
f_pass = 0.1 * np.pi / (2 * np.pi) * fs  # Convertendo para Hz
t = np.linspace(0, 10, 1000)
t_discrete = np.arange(0, 50, 1)

x_pass_s = np.sin(2 * np.pi * f_pass * t)  # Contínuo
x_pass_z = np.sin(2 * np.pi * f_pass * t_discrete)  # Discreto

y_pass_s = signal.lsim(Hc_s, x_pass_s, t)[1]
y_pass_z = signal.dlsim(Hz_z, x_pass_z)[1]

# Sinal senoidal dentro da faixa de rejeição (ω = 0.4π para H(z))
f_stop = 0.4 * np.pi / (2 * np.pi) * fs  # Convertendo para Hz
x_stop_s = np.sin(2 * np.pi * f_stop * t)
x_stop_z = np.sin(2 * np.pi * f_stop * t_discrete)

y_stop_s = signal.lsim(Hc_s, x_stop_s, t)[1]
y_stop_z = signal.dlsim(Hz_z, x_stop_z)[1]

# Plotando Respostas para Senóides
axs[0, 1].plot(t, y_pass_s, label="Hc(s) - Contínuo", color="b")
axs[0, 1].stem(t_discrete, np.squeeze(y_pass_z), label="H(z) - Discreto", linefmt="r-", markerfmt="ro", basefmt="r-")
axs[0, 1].set_title("Resposta - Senoide em Faixa de Passagem")
axs[0, 1].legend()
axs[0, 1].grid()

axs[1, 1].plot(t, y_stop_s, label="Hc(s) - Contínuo", color="b")
axs[1, 1].stem(t_discrete, np.squeeze(y_stop_z), label="H(z) - Discreto", linefmt="r-", markerfmt="ro", basefmt="r-")
axs[1, 1].set_title("Resposta - Senoide em Faixa de Rejeição")
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()
plt.show()
