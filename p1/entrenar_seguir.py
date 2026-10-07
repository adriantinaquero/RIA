#!/usr/bin/env python3
"""
Entrenamiento y evaluación de la política de seguimiento (robot objetivo con
blob verde).

Utiliza Stable-Baselines3 sobre el entorno definido en
robobo_seguir_env.py.

--------------------------------------------------------------------
USO
--------------------------------------------------------------------

Orden recomendado de ejecución:

  1. Calibrar el blob (obligatorio antes de entrenar). Con el robot objetivo
     colocado y quieto delante del seguidor, éste avanza solo hacia él y
     se imprime una fila por pulso con el tamaño del blob:

         python3 entrenar_seguir.py --calibrar

     Con --sin-mover se limita a imprimir lecturas sin mover al
     seguidor (útil si quien se mueve es el robot objetivo).

  2. Comprobar que el entorno cumple la interfaz de Gymnasium:

         python3 entrenar_seguir.py --comprobar

  3. Medir el comportamiento de una política aleatoria, como referencia:

         python3 entrenar_seguir.py --aleatorio --episodios 5

  4. Entrenar.

         python3 entrenar_seguir.py --entrenar --pasos 10000

  5. Evaluar la política aprendida:

         python3 entrenar_seguir.py --evaluar --episodios 10

"""

import argparse
import os
import time

import numpy as np

from enviroment import RoboboSeguimientoEnv, calibrar


RUTA_MODELO = 'seguir_modelo'
RUTA_LOGS = 'seguir_logs'


# =====================================================================
# Hiperparámetros
# =====================================================================

HIPER_SAC = dict(
    learning_rate=3e-4,
    buffer_size=50_000,
    # pasos aleatorios al principio para llenar un poco el buffer antes
    # de que empiece a entrenar de verdad
    learning_starts=500,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    # Una actualización de la red por cada paso del entorno. Es asumible
    # porque el paso del entorno es mucho más lento que la actualización.
    train_freq=1,
    gradient_steps=1,
)


def construir_algoritmo(env):
    from stable_baselines3 import SAC

    return SAC('MlpPolicy', env, verbose=1,
               tensorboard_log=RUTA_LOGS, **HIPER_SAC)


def cargar_algoritmo(ruta, env):
    from stable_baselines3 import SAC

    return SAC.load(ruta, env=env)


# =====================================================================
# Modos de ejecución
# =====================================================================

def modo_comprobar(args):
    """Verifica que el entorno cumple el contrato de Gymnasium.

    Conviene tener al robot objetivo colocado y a la vista antes de
    lanzar esto, porque check_env llama a step() varias veces y si el
    blob nunca aparece no se llega a probar la rama "visible" de la
    observación.
    """
    from stable_baselines3.common.env_checker import check_env

    env = crear_env(args)
    try:
        check_env(env, warn=True)
        print('\nEl entorno cumple la interfaz de Gymnasium.')
    finally:
        env.close()


def modo_aleatorio(args):
    """Ejecuta episodios con acciones aleatorias, como línea base."""
    env = crear_env(args)
    try:
        resumen(env, politica=None, episodios=args.episodios,
                titulo='POLÍTICA ALEATORIA')
    finally:
        env.close()


def modo_entrenar(args):
    env = crear_env(args)
    try:
        from stable_baselines3.common.monitor import Monitor
        # sin este wrapper las columnas ep_rew_mean / ep_len_mean del
        # log de SB3 salen vacías, me pasó la primera vez que lo probé
        env = Monitor(env)

        modelo = construir_algoritmo(env)

        print('\nEntrenando SAC durante {} pasos.'.format(args.pasos))
        print('El robot objetivo tiene que estar moviéndose por su cuenta.')
        print('Se puede interrumpir con Ctrl-C: el modelo se guarda igual.\n')

        t0 = time.time()
        try:
            modelo.learn(total_timesteps=args.pasos, progress_bar=False)
        except KeyboardInterrupt:
            print('\nEntrenamiento interrumpido por el usuario.')

        ruta = RUTA_MODELO
        modelo.save(ruta)
        print('\nModelo guardado en {}.zip'.format(ruta))
        print('Tiempo empleado: {:.1f} min'.format((time.time() - t0) / 60.0))
    finally:
        env.close()


def modo_evaluar(args):
    env = crear_env(args)
    try:
        ruta = RUTA_MODELO
        if not os.path.exists(ruta + '.zip'):
            raise SystemExit(
                'No existe {}.zip. Hay que entrenar primero.'.format(ruta))

        modelo = cargar_algoritmo(ruta, env)

        # determinístico=True toma la acción de mayor probabilidad en
        # lugar de muestrear de la distribución. Durante el
        # entrenamiento se muestrea (hace falta explorar); al evaluar,
        # no.
        resumen(env,
                politica=lambda obs: modelo.predict(obs, deterministic=True)[0],
                episodios=args.episodios,
                titulo='POLÍTICA APRENDIDA (SAC)')
    finally:
        env.close()


# =====================================================================
# Ejecución de episodios y métricas
# =====================================================================

def resumen(env, politica, episodios, titulo):
    recompensas = []
    longitudes = []
    choques = 0
    perdidos = 0
    tiempo_en_banda_total = 0

    print('\n' + titulo)
    print('-' * len(titulo))

    for ep in range(episodios):
        obs, _ = env.reset()
        total = 0.0
        pasos = 0
        pasos_en_banda_ep = 0
        info = {}
        terminado = False
        truncado = False

        while not (terminado or truncado):
            if politica is None:
                accion = env.action_space.sample()
            else:
                accion = politica(obs)
            obs, r, terminado, truncado, info = env.step(accion)
            total += r
            pasos += 1
            # en_banda_consecutivos es un contador de pasos seguidos,
            # pero aquí sólo interesa si en este paso estaba en banda o
            # no, por eso basta con mirar si es >= 1
            if info.get('en_banda_consecutivos', 0) >= 1:
                pasos_en_banda_ep += 1

        recompensas.append(total)
        longitudes.append(pasos)
        tiempo_en_banda_total += pasos_en_banda_ep / pasos if pasos else 0.0

        # para clasificar el episodio en una sola palabra, por orden de
        # prioridad: si hubo choque es lo primero que se mira
        if info.get('choque'):
            choques += 1
            desenlace = 'choque'
        elif info.get('perdido'):
            perdidos += 1
            desenlace = 'perdido'
        else:
            desenlace = 'tiempo agotado'

        print('episodio {:2d}   pasos {:3d}   recompensa {:8.2f}   '
              'tam final {:.3f}   % en banda {:5.1f}   {}'
              .format(ep + 1, pasos, total, info.get('blob_tam', 0.0),
                      100.0 * pasos_en_banda_ep / pasos if pasos else 0.0,
                      desenlace))

    print('\nrecompensa media  {:.2f}  (desviación {:.2f})'
          .format(float(np.mean(recompensas)), float(np.std(recompensas))))
    print('longitud media    {:.1f} pasos'.format(float(np.mean(longitudes))))
    print('% en banda medio  {:.1f}'
          .format(100.0 * tiempo_en_banda_total / episodios))
    print('choques           {}/{}'.format(choques, episodios))
    print('perdidos          {}/{}'.format(perdidos, episodios))


# =====================================================================
# Construcción del entorno y línea de órdenes
# =====================================================================

def crear_env(args):
    # un único sitio donde se construye el entorno para no repetir los
    # mismos argumentos en los cinco modos
    return RoboboSeguimientoEnv(
        pasos_max=args.pasos_max,
        usar_simulador=not args.sin_simulador,
        verbose=args.detalle)


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
