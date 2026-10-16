# Entrenamiento velocidad constante

## Recompensa:
```py
def _recompensa(self, blob_x, dist, visible, accion, accion_prev,
                    perdido, choque):
        if choque:
            return -3.0
        if perdido:
            return -2.0
        if not visible:
            return -0.25 * self.pasos_perdido
        # ---- 1) término principal: distancia + centrado ----
        e_x = abs(blob_x)
    
        # Error de distancia respecto al objetivo
        e_dist = abs(dist)

        # Recompensa continua en el rango (0, 1] cuando está bien posicionado
        r_posicion = (1.0 - e_x) * 0.4 + (1.0 - e_dist) * 0.6

        # Bonus adicional si se encuentra dentro de la tolerancia deseada
        if e_x <= X_TOLERANCIA and BANDA_OBJETIVO_M[0] <= dist <= BANDA_OBJETIVO_M[1]:
            r_posicion += 0.5
        
        
        r_posicion -= abs(accion_prev[1]-accion[1])/10
        r_posicion -= abs(accion_prev[0]-accion[0])/5
        return float(r_posicion)
```

## Espacio observación:

```py
self.observation_space = spaces.Box(
        # [blob_x, dist, visible, ultimo_blob_x, ultima_accion_v, ultima_accion_w]
        low=np.array([-1.0, -1.0, 0.0, -1, -1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0, 1, 1.0, 1.0], dtype=np.float32),
        dtype=np.float32)
```

## Variables

```py
# Color por el que se filtra dentro de BlobArray.msg.blobs.
COLOR_BLOB = 'green'

# posx del blob en [0, 100], donde 50 es el centro de la imagen.
BLOB_X_CENTRO = 50.0

DIST_CHOQUE_M = 0.12
DIST_PERDIDO_M = 0.9
DIST_TARGET_M = 0.3
BANDA_OBJETIVO_M = [0.2, 0.45]


# Desviación horizontal del blob que se tolera 
# En las mismas unidades que blob_x normalizado ([-1, 1]).
X_TOLERANCIA = 0.15

# Pasos consecutivos sin ver el blob antes de dar el episodio por perdido.
PASOS_PERDIDO_MAX = 4

# Tiempo máximo desde la última detección del blob para seguir
# considerándolo "visible ahora mismo"
BLOB_VIGENCIA_S = 0.4

V_MIN = 0.02
V_MAX = 0.15
W_MAX = 0.35

PASO_S = 0.2

ESPERA_MAX_S = 1.0
```


# Entrenamiento contraste

## Recompensa:
```py
```

## Espacio observación:
```py
```

## Variables

```py
```

# Entrenamiento rápido constante

## Recompensa:
```py
```

## Espacio observación:
```py
```

## Variables

```py
```

# Entrenamiento velocidad fluctuante

## Recompensa:
```py
```

## Espacio observación:
```py
```

## Variables

```py
```