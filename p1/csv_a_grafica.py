import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# NOMBRE DEL CSV PARA GRAFICAR
CSV_NAME = "logs csv/seguir_logs_SAC_CONST_3_EpRewMean.csv"

# CAMBIAR DEPENDIENDO DE SI la curva del csv es de EpisodeLengthMean o EpisodeRewardMean
CSV_TYPE = "EpisodeRewMean"

# estilo visual de seaborn
sns.set_theme(style="whitegrid")

# introducir nombre del cs
df = pd.read_csv(CSV_NAME)

df.columns = df.columns.str.strip()
# las columnas son 'Wall time', 'Step', 'Value'

# aplicamos suavizado (media móvil)
window_size = 10
df["Value_Smoothed"] = df["Value"].rolling(window=window_size).mean()

plt.figure(figsize=(10, 6))

# serie sin suavizar
sns.lineplot(
    data=df,
    x="Step",
    y="Value",
    color="blue",
    alpha=0.3,
    linewidth=1,
    label="Sin suavizar",
)

# serie suavizada
sns.lineplot(
    data=df,
    x="Step",
    y="Value_Smoothed",
    color="blue",
    linewidth=2,
    label=f"Media móvil (ventana = {window_size})",
)

# etiquetas y leyenda
if CSV_TYPE == "EpisodeLengthMean":
    plt.title(
        "Curva de Aprendizaje - Duración del Episodio (SAC)",
        fontsize=14,
        weight="bold",
    )
else:
    plt.title(
        "Curva de Aprendizaje - Recompensa del Episodio (SAC)",
        fontsize=14,
        weight="bold",
    )

plt.xlabel("Pasos del entorno (step())", fontsize=12)

if CSV_TYPE == "EpisodeLengthMean":
    plt.ylabel("Duración del episodio (pasos)", fontsize=12)
else:
    plt.ylabel("Recompensa media por episodio (adimensional)", fontsize=12)

plt.legend(loc="upper left")
plt.tight_layout()

plt.show()