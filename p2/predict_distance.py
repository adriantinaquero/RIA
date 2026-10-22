"""
Relaciona |col2 - col1| con la tercera columna (tamaño del blob)
y construye una función que predice la distancia a partir del tamaño.

Uso:  python predictor.py            (lee datos.txt)
      from predictor import predecir_distancia
"""
import numpy as np

d = np.loadtxt("datos.txt")
col1, col2, tam = d.T
dist = np.abs(col2 - col1)

def r2(y, yh):
    return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)

# Modelo elegido: dist = a * ln(tamaño) + b  (el mejor de los probados, R² ≈ 0.95)
a, b = np.polyfit(np.log(tam), dist, 1)

def predecir_distancia(tamano):
    """Distancia estimada |col2 - col1| a partir del tamaño del blob."""
    tamano = np.clip(np.asarray(tamano, dtype=float), 1, None)
    return a * np.log(tamano) + b

if __name__ == "__main__":
    pred = predecir_distancia(tam)
    rmse = np.sqrt(np.mean((dist - pred) ** 2))
    print(f"dist = {a:.2f} * ln(tamaño) + {b:.2f}")
    print(f"R² = {r2(dist, pred):.4f}   RMSE = {rmse:.1f}")
    for t in (7, 14, 22, 43, 57, 72, 90, 109, 133):
        print(f"tamaño {t:3d} -> distancia ≈ {float(predecir_distancia(t)):.0f}")