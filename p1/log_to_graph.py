import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configurar el estilo visual de seaborn
sns.set_theme(style="whitegrid")

# 1. Cargar el archivo CSV (reemplaza 'tu_archivo.csv' por el nombre real de tu archivo)
df = pd.read_csv("seguir_logs_SAC_CONST_1_EpLenMean.csv")

# Asegurar nombres de columnas limpios por si hay espacios
df.columns = df.columns.str.strip()
# Las columnas esperadas son: 'Wall time', 'Step', 'Value'

# 2. Aplicar suavizado (media móvil)
window_size = 10  # Ajusta el tamaño de la ventana según el ruido
df["Value_Smoothed"] = df["Value"].rolling(window=window_size).mean()

# 3. Crear la figura con Seaborn
plt.figure(figsize=(10, 6))

# Serie original sin suavizar (trazo tenue)
sns.lineplot(
    data=df,
    x="Step",
    y="Value",
    color="blue",
    alpha=0.3,
    linewidth=1,
    label="Sin suavizar",
)

# Serie suavizada (trazo principal más marcado)
sns.lineplot(
    data=df,
    x="Step",
    y="Value_Smoothed",
    color="blue",
    linewidth=2,
    label=f"Media móvil (ventana = {window_size})",
)

# 4. Etiquetas y formato exigidos por el profesor
plt.title(
    "Curva de Aprendizaje - Duración del Episodio (SAC)",
    fontsize=14,
    weight="bold",
)
plt.xlabel("Pasos del entorno (step())", fontsize=12)
plt.ylabel("Duración del episodio (pasos)", fontsize=12)

# Leyenda y diseño final
plt.legend(loc="upper left")
plt.tight_layout()

# Guardar en alta resolución para la memoria
plt.savefig("curva_ep_len_mean.png", dpi=300)
plt.show()