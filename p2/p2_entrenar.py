import os, pickle
from registro import RegistroGeneracion # apartado 6
from p2_entorno import eval_genomes  
import neat
import numpy as np
from stable_baselines3 import SAC


os.makedirs("chk", exist_ok=True) # el Checkpointer no crea carpetas
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    "config-p2.ini")

poblacion = neat.Population(config)
poblacion.add_reporter(neat.StdOutReporter(True))
estadisticas = neat.StatisticsReporter()
poblacion.add_reporter(estadisticas)
poblacion.add_reporter(neat.Checkpointer(generation_interval=1,
                                        filename_prefix="chk/p2-"))
poblacion.add_reporter(RegistroGeneracion("evolucion.csv", "genomas"))

# --------- Entrenar ---------
mejor = poblacion.run(eval_genomes, 30) # 30 generaciones
with open("mejor_genoma.pkl", "wb") as f:
    pickle.dump(mejor, f)

# --------- Reanudar entrenamiento ---------
poblacion = neat.Checkpointer.restore_checkpoint("chk/p2-12")
# los registros no se recuperan: se vuelven a añadir, igual que al empezar
poblacion.add_reporter(neat.StdOutReporter(True))
poblacion.add_reporter(neat.Checkpointer(generation_interval=1,
                                        filename_prefix="chk/p2-"))
poblacion.add_reporter(RegistroGeneracion("evolucion.csv", "genomas"))
mejor = poblacion.run(eval_genomes, 18) # las generaciones que faltan

# --------- Evaluar red final ---------
def controlador_neat(ruta_genoma, ruta_config):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        ruta_config)
    with open(ruta_genoma, "rb") as f:
        genoma = pickle.load(f)
    red = neat.nn.FeedForwardNetwork.create(genoma, config)
    return lambda obs: np.array(red.activate(obs), dtype=np.float32)

def controlador_sb3(ruta_modelo):
    modelo = SAC.load(ruta_modelo) # o PPO.load
    return lambda obs: modelo.predict(obs, deterministic=True)[0]

# red = controlador_neat("mejor_genoma.pkl", "config-p2.ini")
# evaluar_anexo_b(env, red, configuracion="C2") # el evaluador de la P1