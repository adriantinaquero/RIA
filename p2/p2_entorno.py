import numpy as np
import neat
from p1_environment import RoboboSeguimientoEnv # el entorno de la Práctica 1

env = RoboboSeguimientoEnv() # un único entorno y un único nodo
PASOS_EVAL = 150 # 30 s con pasos de 0,2 s
CONTADOR = {"pasos": 0, "evaluaciones": 0}
N_ITERACIONES = 3

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
    pos = 1
    for _, genoma in genomas:
        print(f"Individuo {pos}:")
        scores = []
        for _ in range(N_ITERACIONES):
            scores.append(evaluar_genoma(genoma, config))
        scores = np.array(scores)
        genoma.fitness = float(scores.mean() - 0.5 * scores.std())
        print(f"Fitness = {genoma.fitness}")
        pos += 1


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

SIGMA_CERCA = 0.20
SIGMA_LEJOS = 0.40
SIGMA_X     = 0.35
BLOB_X_CENTRO = 50.0

DIST_CHOQUE_M = 0.12
DIST_PERDIDO_M = 0.9
DIST_TARGET_M = 0.3
BANDA_OBJETIVO_M = np.array([0.2, 0.45])
NORM_BANDA_OBJETIVO = (BANDA_OBJETIVO_M - DIST_TARGET_M)/DIST_TARGET_M  

X_TOLERANCIA = 0.15

def _r_dist(info):
    # unclipped normalised error from ground truth, for the fitness only
    d_real = info['distancia_real_mm'] / 1000.0
    d = (d_real - DIST_TARGET_M) / DIST_TARGET_M      # can exceed +1
    s = SIGMA_CERCA if d < 0 else SIGMA_LEJOS
    return np.exp(-0.5 * (d / s) ** 2)

def calidad(registro: list[dict]) -> float:
    total = 0.0
    n = len(registro)

    for info in registro:
        if info['choque'] or info['perdido']:
            # terminal failure: charge for every step that was not played
            restantes = PASOS_EVAL - n
            total -= 1.0 * (restantes + 1)
            if info['choque']:
                total -= 2.0                       # extra for crashing vs. losing
            break

        if not info['visible']:
            # use ground truth (not available to the net) to grade failures
            d_real = info['distancia_real_mm'] / 1000.0
            err = min(1.0, abs(d_real - DIST_TARGET_M) / DIST_TARGET_M)
            total += -0.2 - 0.3 * err              # in [-0.5, -0.2]
            continue

        e_x  = abs(info['blob_x'])
        r_d  = _r_dist(info)
        r_x  = np.exp(-0.5 * (e_x / SIGMA_X) ** 2)

        # distance gates centering; both in (0, 1]
        r = r_d * (0.4 + 0.6 * r_x)

        # smooth "in the sweet spot" bonus, grows with sustained tracking
        if e_x <= X_TOLERANCIA and NORM_BANDA_OBJETIVO[0] <= info['dist'] <= NORM_BANDA_OBJETIVO[1]:
            racha = min(info['en_banda_consecutivos'], 50) / 50
            r += 0.3 * (0.5 + 0.5 * racha)

        total += r

    return total / PASOS_EVAL