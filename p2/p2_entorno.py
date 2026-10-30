import numpy as np
import neat
from p1_environment import RoboboSeguimientoEnv # el entorno de la Práctica 1

env = RoboboSeguimientoEnv() # un único entorno y un único nodo
PASOS_EVAL = 150 # 30 s con pasos de 0,2 s
CONTADOR = {"pasos": 0, "evaluaciones": 0}

def evaluar_genoma(genoma, config):
    red = neat.nn.FeedForwardNetwork.create(genoma, config)
    obs, info = env.reset()
    registro = [] # lo que necesite la calidad
    for _ in range(PASOS_EVAL):
        accion = np.array(red.activate(obs), dtype=np.float32)
        obs, _, terminado, truncado, info = env.step(accion)
        registro.append(info)
        if terminado or truncado:
            break
    CONTADOR["pasos"] += len(registro)
    CONTADOR["evaluaciones"] += 1
    return calidad(registro) # la define cada grupo (apartado 4)

def eval_genomes(genomas, config):
    for _, genoma in genomas:
        genoma.fitness = evaluar_genoma(genoma, config)


# info = {
#     'blob_x': blob_x,
#     'dist': dist,
#     'visible': visible,
#     'choque': choque,
#     'perdido': perdido,
#     'en_banda_consecutivos': self.en_banda,
#     'pose_sim': self._pose,
#     'distancia_real_mm': d_m,
# }

SIGMA = 0.18
MU = 0.0
BLOB_X_CENTRO = 50.0

DIST_CHOQUE_M = 0.12
DIST_PERDIDO_M = 0.9
DIST_TARGET_M = 0.3
BANDA_OBJETIVO_M = np.array([0.2, 0.45])
NORM_BANDA_OBJETIVO = (BANDA_OBJETIVO_M - DIST_TARGET_M)/DIST_TARGET_M  

X_TOLERANCIA = 0.15

def calidad(registro:list[dict]) -> float:
    calidad_total = 0
    for info in registro:
        if info['choque']:
            calidad_total += -3.0
            continue
        if info['perdido']:
            calidad_total += -2.0
            continue
        if not info['visible']:
            calidad_total += -0.25
            continue
        # ---- 1) término principal: distancia + centrado ----
        e_x = abs(info['blob_x'])

        # Error de distancia respecto al objetivo
        reward_e_dist = np.exp(-((info['dist'] - MU) ** 2) / (2 * SIGMA ** 2)) / (SIGMA * np.sqrt(2 * np.pi))
        reward_e_dist = (reward_e_dist - 0.5)*2
        # Recompensa continua en el rango (0, 1] cuando está bien posicionado
        r_posicion = (1.0 - e_x) * 0.4 + reward_e_dist * 0.6

        # Bonus adicional si se encuentra dentro de la tolerancia deseada
        if e_x <= X_TOLERANCIA and NORM_BANDA_OBJETIVO[0] <= info['dist'] <= NORM_BANDA_OBJETIVO[1]:
            r_posicion += 0.75

        calidad_total += r_posicion
            
    return calidad_total/PASOS_EVAL