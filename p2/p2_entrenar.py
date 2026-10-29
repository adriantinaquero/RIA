import os, pickle
from registro import RegistroGeneracion # apartado 6
from p2_entorno import eval_genomes  
import neat
import numpy as np
from stable_baselines3 import SAC
import argparse


def modo_entrenar(args):
    os.makedirs("chk", exist_ok=True) # el Checkpointer no crea carpetas
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        args.ruta_config)

    poblacion = neat.Population(config)
    poblacion.add_reporter(neat.StdOutReporter(True))
    estadisticas = neat.StatisticsReporter()
    poblacion.add_reporter(estadisticas)
    poblacion.add_reporter(neat.Checkpointer(generation_interval=1,
                                            filename_prefix=f"chk/{args.prefijo}-"))
    poblacion.add_reporter(RegistroGeneracion(args.archivo_final, "genomas"))

    mejor = poblacion.run(eval_genomes, 30) # 30 generaciones
    with open(args.ruta_modelo, "wb") as f:
        pickle.dump(mejor, f)

# --------- Reanudar entrenamiento ---------
def reanudar_entrenamiento(args):
    poblacion = neat.Checkpointer.restore_checkpoint(f"chk/{args.prefijo}-{args.checkpoint}")
    # los registros no se recuperan: se vuelven a añadir, igual que al empezar
    poblacion.add_reporter(neat.StdOutReporter(True))
    poblacion.add_reporter(neat.Checkpointer(generation_interval=1,
                                            filename_prefix=f"chk/{args.prefijo}-"))
    poblacion.add_reporter(RegistroGeneracion(args.archivo_final, "genomas"))
    mejor = poblacion.run(eval_genomes, args.pasos_faltantes) # las generaciones que faltan
    with open(args.ruta_modelo, "wb") as f:
        pickle.dump(mejor, f)

# --------- Evaluar red final ---------
def modo_evaluar(args):
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

    red = controlador_neat(args.ruta_genoma, args.ruta_config)
    # evaluar_anexo_b(env, red, configuracion="C2") # el evaluador de la P1

def main():
    p = argparse.ArgumentParser(
        description='Entrenamiento con aprendizaje por refuerzo'
                    'del seguimiento de un robot objetivo con '
                    'blob verde, manteniendo siempre la misma distancia.')

    modo = p.add_mutually_exclusive_group(required=True)
    modo.add_argument('--calibrar', action='store_true',
                      help='mide el tamaño del blob a distintas distancias')
    modo.add_argument('--comprobar', action='store_true',
                      help='verifica la interfaz del entorno')
    modo.add_argument('--aleatorio', action='store_true',
                      help='ejecuta episodios con acciones aleatorias')
    modo.add_argument('--entrenar', action='store_true')
    modo.add_argument('--evaluar', action='store_true')

    p.add_argument('--pasos', type=int, default=10_000,
                   help='pasos totales de entrenamiento')
    p.add_argument('--episodios', type=int, default=5)
    p.add_argument('--pasos-max', type=int, default=200,
                   help='longitud máxima de un episodio')
    p.add_argument('--sin-simulador', action='store_true',
                   help='no usar el modulo sim; los robots se reponen a mano')
    p.add_argument('--sin-mover', action='store_true',
                   help='con --calibrar, no mueve al seguidor')
    p.add_argument('--detalle', action='store_true',
                   help='imprime una línea por paso')
    p.add_argument('--ruta-modelo', type=str, default='mejor_genoma.plk',
                    help='Archivo .pkl donde se guarda el mejore genoma')
    p.add_argument('--ruta-config', type=str, default='config.txt',
                    help='Archivo donde esta guardada la configuración del sistema')
    p.add_argument('--prefijo', type=str, default='p2',
                    help='prefijo del nombre del checkpoint')
    p.add_argument('--archivo-final', type=str, default='evolucion.csv',
                    help='Archivo .csv donde se guarda el historico de la evoludcion')
    p.add_argument('--checkpoint', type=int,
                    help='Numero de pasos que se guardaron en el checkpoint')
    p.add_argument('--pasos-faltantes', type=int,
                    help='Numero de pasos adicionales que tiene que hacer el sistema')

            
    

    args = p.parse_args()

    if args.calibrar:
        calibrar(avanzar=not args.sin_mover)
    elif args.comprobar:
        modo_comprobar(args)
    elif args.aleatorio:
        modo_aleatorio(args)
    elif args.entrenar:
        modo_entrenar(args)
    elif args.evaluar:
        modo_evaluar(args)


if __name__ == '__main__':
    main()
