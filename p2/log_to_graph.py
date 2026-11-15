import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import os
import argparse

p = argparse.ArgumentParser()

p.add_argument("--file-name", type=str)
p.add_argument("--hide", action="store_true")

args = p.parse_args()
# Configurar el estilo visual de seaborn
sns.set_theme(style="whitegrid")

# 1. Cargar el archivo CSV
df = pd.read_csv(args.file_name)

# Asegurar nombres de columnas limpios por si hay espacios
df.columns = df.columns.str.strip()
# Las columnas esperadas son: 'Wall time', 'Step', 'Value'

# 2. Aplicar suavizado (media móvil)
window_size = 10  # Ajusta el tamaño de la ventana según el ruido
df["fit_media_smoothed"] = df["fit_media"].rolling(window=window_size).mean()
df["fit_std_smoothed"] = df["fit_std"].rolling(window=window_size).mean()
# 3. Crear la figura con Seaborn
plt.figure(figsize=(10, 6))

# Mejor individuo
sns.lineplot(
    data=df,
    x="generacion",
    y="fit_mejor",
    color="green",
    alpha=1,
    linewidth=1,
    label="Fitness mejor individuo",
)

# Fitness media
sns.lineplot(
    data=df,
    x="generacion",
    y="fit_media",
    color="blue",
    linewidth=0.3,
    label="Fitness media",
)

M_new_vec = np.array(df["fit_media_smoothed"])
Sigma_new_vec = np.array(df["fit_std_smoothed"])

lower_bound = M_new_vec - Sigma_new_vec
upper_bound = M_new_vec + Sigma_new_vec
plt.fill_between(df["generacion"], lower_bound, upper_bound, alpha=.3)

# Fitness media suavizada (trazo principal más marcado)
sns.lineplot(
    data=df,
    x="generacion",
    y="fit_media_smoothed",
    color="blue",
    linewidth=2,
    label=f"Media móvil (ventana = {window_size})",
)

# Fitness peor individuo
sns.lineplot(
    data=df,
    x="generacion",
    y="fit_min",
    color="red",
    linewidth=2,
    label="Fitness peor individuo",
)

# 4. Etiquetas y formato exigidos por el profesor
plt.title(
    "Curva de Evolución",
    fontsize=14,
    weight="bold",
)
plt.xlabel(f"Generaciones.\n Poblacion = 100, Episodios por Individuo = 5, Duración máxima = 150 pasos, Pasos = {sum(df["pasos"])}", fontsize=12)
plt.ylabel("Fitness [-6.5, 1.8]", fontsize=12)

# Leyenda y diseño final
plt.legend(loc="upper left")
plt.tight_layout()


name = args.file_name.split("/")[-1].split(".")[0]
# Guardar en alta resolución para la memoria
os.makedirs("plots", exist_ok=True)
plt.savefig(f"plots/{name}.png", dpi=300)
if not args.hide:
    plt.show()