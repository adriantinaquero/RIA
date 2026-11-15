import os, pickle, time, json
from registro import RegistroGeneracion # apartado 6
from p2_entorno import eval_genomes  
import neat
import numpy as np
from stable_baselines3 import SAC
import argparse
from p1_environment import RoboboSeguimientoEnv
import p2_metricas as met

def crear_env(args):
    # un único sitio donde se construye el entorno para no repetir los
    # mismos argumentos en los cinco modos
    return RoboboSeguimientoEnv(
        pasos_max=args.pasos_max,
        usar_simulador=not args.sin_simulador,
        verbose=args.detalle)

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

    mejor = poblacion.run(eval_genomes, args.generaciones) # 30 generaciones
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

    modelos = rutas_modelos(args)               # {"C1": ruta, "C2": ruta, ...}: un modelo por configuración
    faltan = [f"{c}: {r}" for c, r in modelos.items() if not os.path.isfile(r)]
    if faltan:                                  # mejor fallar ahora que a mitad de la evaluación
        raise SystemExit("No se encuentran estos modelos:\n  " + "\n  ".join(faltan))

    env = crear_env(args)
    episodios = []
    for c, ruta in modelos.items():
        red = controlador_neat(ruta, args.ruta_config)          # el modelo de ESTA configuración
        # el guía lo configura quien lance su script: se pausa para poder cambiarlo entre configuraciones
        if not args.sin_pausa:
            input(f"\nConfigura al guía en {c} (modelo {ruta}) y pulsa Enter para continuar... ")
        episodios += resumen(env, red, args.episodios, c, modelo=ruta)

    informe(episodios, args)


def rutas_modelos(args):
    """Modelo (.pkl) que se evalúa en cada configuración del guía: {config: ruta}, en orden.

      --modelos C1=a.pkl,C2=b.pkl,...   uno por configuración (lo normal)
      --ruta-modelo m_{config}.pkl      plantilla: {config} se sustituye por C1, C2... (en --configs)
      --ruta-modelo m.pkl               sin {config}: el MISMO modelo en todas (aviso)
    """
    if args.modelos:
        modelos = {}
        for par in args.modelos:
            if '=' not in par:
                raise SystemExit(f"--modelos: '{par}' no tiene la forma CONFIG=ruta.pkl")
            c, ruta = par.split('=', 1)
            modelos[c.strip().upper()] = ruta.strip()
        if args.configs:                        # --configs filtra y ordena
            pedidas = [c.strip().upper() for c in args.configs.split(',') if c.strip()]
            sin_modelo = [c for c in pedidas if c not in modelos]
            if sin_modelo:
                raise SystemExit(f"--configs pide {sin_modelo} pero --modelos no tiene modelo para ellas")
            modelos = {c: modelos[c] for c in pedidas}
        return modelos
    configs = [c.strip().upper() for c in (args.configs or 'C1,C2,C3,C4').split(',') if c.strip()]
    if '{config}' in args.ruta_modelo:
        return {c: args.ruta_modelo.format(config=c) for c in configs}
    print(f"AVISO: se evalúa el mismo modelo ({args.ruta_modelo}) en todas las configuraciones; "
          "usa --modelos C1=...,C2=... para un modelo por configuración.")
    return {c: args.ruta_modelo for c in configs}


def _pose_guia(env, info):
    """Pose (x, z, yaw) del guía: de info si el entorno la incluye; si no, del propio entorno."""
    pose = info.get('pose_objetivo')
    if pose is None:
        pose = getattr(env, '_pose_robot_objetivo', None)
    return pose


def resumen(env, politica, episodios, titulo, modelo=''):
    """Ejecuta `episodios` episodios con la configuración del guía `titulo` (p. ej. "C2") y devuelve la lista
    de met.Episodio, con TODOS los pasos registrados: instante, posición del seguidor y del guía (la pelota va
    pegada a él), separación, desviación y acción aplicada. Las métricas se calculan en informe()."""
    registros = []

    cabecera = titulo + (f'   (modelo: {modelo})' if modelo else '')
    print('\n' + cabecera)
    print('-' * len(cabecera))

    for ep in range(episodios):
        obs, _ = env.reset()
        episodio = met.Episodio(config=titulo, indice=ep + 1, modelo=modelo)
        t0 = time.monotonic()
        info = {}
        terminado = truncado = error = False

        while not (terminado or truncado):
            if politica is None:
                accion = env.action_space.sample()
            else:
                accion = politica(obs)
            resultado = env.step(accion)
            if resultado is None:
                # p1_environment.step() hace "return" a secas si no llega la lectura de la cámara
                error = True
                break
            obs, r, terminado, truncado, info = resultado
            instante = info['t'] if 't' in info else time.monotonic() - t0   # s desde el inicio del episodio
            episodio.pasos.append(met.crear_paso(instante, info, accion, _pose_guia(env, info)))

        episodio.desenlace = met.clasificar_desenlace(info, error)
        episodio.duracion_s = time.monotonic() - t0
        registros.append(episodio)

        m = met.metricas([episodio])
        print('episodio {:2d}   pasos {:3d}   {:5.1f} s   separación media {}   en banda {}   '
              '|desv| {}   {}'.format(
                  ep + 1, len(episodio.pasos), episodio.duracion_s,
                  met._f(m['sep_media_m'], '{:.3f} m'), met._f(m['pct_banda'], '{:5.1f}%'),
                  met._f(m['desv_media_abs_deg'], '{:5.1f}°'), episodio.desenlace))
    return registros


def informe(episodios, args=None):
    """Imprime las métricas en conjunto y por configuración, y guarda el registro por pasos y las métricas."""
    res = met.metricas_por_configuracion(episodios)
    print('\n' + met.formatear_tabla(res))

    det = res['TOTAL']['paradas_detalle']
    if det:
        print('\nParadas del guía (C4):')
        for p in det:
            veredicto = 'SUPERADA' if p['superada'] else 'no superada'
            motivo = []
            if p['sin_reanudar']:
                motivo.append('el guía no reanuda antes de acabar el episodio')
            if not p['seguidor_se_detiene']:
                motivo.append('el seguidor no se detiene')
            if p['choque']:
                motivo.append('choque')
            if p['fallo_tras_reanudar']:
                motivo.append('fallo en los 10 s tras reanudar')
            if p['ventana_truncada'] and p['superada']:
                motivo.append('ventana de 10 s truncada por el fin del episodio')
            print('  ep {:2d}  parada {:6.1f}-{:6.1f} s  {}{}'.format(
                p['episodio'], p['inicio_s'], p['fin_s'], veredicto, '  (' + '; '.join(motivo) + ')' if motivo else ''))

    desv = res['TOTAL']['desv_media_abs_deg']
    if desv == desv and desv > 60.0:
        print('\nAVISO: |desviación| media > 60°. Revisa met.YAW_EN_GRADOS / la convención del yaw '
              '(el frente debería ser (sin yaw, cos yaw)).')

    if args is not None and getattr(args, 'csv_pasos', None):
        met.escribir_csv(episodios, args.csv_pasos)
        print(f'\nRegistro por pasos guardado en {args.csv_pasos}')
    if args is not None and getattr(args, 'json_metricas', None):
        with open(args.json_metricas, 'w') as f:
            json.dump(res, f, indent=2, default=float)
        print(f'Métricas guardadas en {args.json_metricas}')
    return res


def main():
    p = argparse.ArgumentParser(
        description='Entrenamiento con aprendizaje por refuerzo'
                    'del seguimiento de un robot objetivo con '
                    'blob verde, manteniendo siempre la misma distancia.')

    modo = p.add_mutually_exclusive_group(required=True)
    modo.add_argument('--entrenar', action='store_true')
    modo.add_argument('--evaluar', action='store_true')
    modo.add_argument('--reanudar-entrenamiento', action='store_true')

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
    p.add_argument('--ruta-modelo', type=str, default='mejor_genoma.pkl',
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
    p.add_argument('--generaciones', type=int, default=30,
                    help='Numero de generaciones que se van a entrenar')
    p.add_argument('--modelos', type=str, default=None, nargs="+",
                    help='con --evaluar: un modelo por configuración del guía, p. ej. '
                         'C1=genoma_c1.pkl,C2=genoma_c2.pkl,C3=genoma_c3.pkl,C4=genoma_c4.pkl')
    p.add_argument('--configs', type=str, default=None,
                    help='con --evaluar: qué configuraciones evaluar y en qué orden (por defecto, las de --modelos '
                         'o C1,C2,C3,C4). Con --ruta-modelo "genoma_{config}.pkl" se sustituye {config}')
    p.add_argument('--sin-pausa', action='store_true',
                    help='con --evaluar: no pausar antes de cada configuración para configurar al guía')
    p.add_argument('--csv-pasos', type=str, default=None,
                    help='con --evaluar: archivo .csv donde se guarda el registro de cada paso')
    p.add_argument('--json-metricas', type=str, default=None,
                    help='con --evaluar: archivo .json donde se guardan las métricas')

            
    

    args = p.parse_args()

    if args.reanudar_entrenamiento:
        reanudar_entrenamiento(args)
    elif args.entrenar:
        modo_entrenar(args)
    elif args.evaluar:
        modo_evaluar(args)


if __name__ == '__main__':
    main()
