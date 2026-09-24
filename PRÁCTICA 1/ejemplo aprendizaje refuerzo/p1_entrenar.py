#!/usr/bin/env python3
"""
Práctica 1 — Entrenamiento y evaluación de la política de aproximación.

Utiliza Stable-Baselines3 sobre el entorno definido en p1_robobo_env.py.

--------------------------------------------------------------------
USO
--------------------------------------------------------------------

Orden recomendado de ejecución:

  1. Calibrar los sensores (obligatorio antes de entrenar):

         python3 p1_entrenar.py --calibrar

  2. Comprobar que el entorno cumple la interfaz de Gymnasium:

         python3 p1_entrenar.py --comprobar

  3. Medir el comportamiento de una política aleatoria. Es la referencia
     contra la que hay que comparar: un entrenamiento que no supere este
     valor no ha aprendido nada.

         python3 p1_entrenar.py --aleatorio --episodios 5

  4. Entrenar:

         python3 p1_entrenar.py --entrenar --algoritmo sac --pasos 10000

  5. Evaluar la política aprendida:

         python3 p1_entrenar.py --evaluar --episodios 10

--------------------------------------------------------------------
SOBRE LA ELECCIÓN DEL ALGORITMO
--------------------------------------------------------------------

Los dos algoritmos admiten espacios de acción continuos, pero no se
comportan igual con el presupuesto de pasos del que se dispone aquí.

  SAC   Fuera de política (off-policy). Reutiliza cada transición
        muchas veces desde un búfer de repetición, de modo que
        aprovecha mucho mejor un número reducido de interacciones. Es
        la opción adecuada cuando cada paso cuesta tiempo real, como
        ocurre aquí.

  PPO   Dentro de política (on-policy). Descarta las transiciones tras
        cada actualización, así que necesita bastantes más pasos para
        alcanzar el mismo resultado. Es más estable y más fácil de
        ajustar, pero en esta práctica resulta más lento.

Con un paso de entorno cada 0.3 s aproximadamente, 10 000 pasos suponen
del orden de 50 minutos de simulación. Conviene lanzarlo con tiempo.
"""

import argparse
import os
import time

import numpy as np

from p1_robobo_env import RoboboParedEnv, calibrar


RUTA_MODELO = 'p1_modelo'
RUTA_LOGS = 'p1_logs'


# =====================================================================
# Hiperparámetros
# =====================================================================
#
# Los valores por defecto de Stable-Baselines3 están pensados para
# entornos simulados que se ejecutan a miles de pasos por segundo. Aquí
# cada paso cuesta tiempo real, de modo que hay que ajustar sobre todo
# el tamaño de los lotes de recogida de datos.

HIPER_SAC = dict(
    learning_rate=3e-4,
    buffer_size=50_000,
    # Pasos que se recogen con acciones aleatorias antes de empezar a
    # entrenar. Llenan el búfer de repetición con algo de variedad.
    learning_starts=500,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    # Una actualización de la red por cada paso del entorno. Es asumible
    # porque el paso del entorno es mucho más lento que la actualización.
    train_freq=1,
    gradient_steps=1,
)

HIPER_PPO = dict(
    learning_rate=3e-4,
    # PPO recoge n_steps transiciones antes de cada actualización. El
    # valor por defecto (2048) supondría diez minutos de simulación entre
    # actualización y actualización.
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
)


def construir_algoritmo(nombre, env):
    from stable_baselines3 import PPO, SAC

    if nombre == 'sac':
        return SAC('MlpPolicy', env, verbose=1,
                   tensorboard_log=RUTA_LOGS, **HIPER_SAC)
    if nombre == 'ppo':
        return PPO('MlpPolicy', env, verbose=1,
                   tensorboard_log=RUTA_LOGS, **HIPER_PPO)
    raise ValueError('Algoritmo desconocido: ' + nombre)


def cargar_algoritmo(nombre, ruta, env):
    from stable_baselines3 import PPO, SAC

    clase = SAC if nombre == 'sac' else PPO
    return clase.load(ruta, env=env)


# =====================================================================
# Modos de ejecución
# =====================================================================

def modo_comprobar(args):
    """Verifica que el entorno cumple el contrato de Gymnasium.

    check_env recorre los espacios declarados, llama a reset() y a
    step() y avisa de las incoherencias habituales: tipos que no
    coinciden con el espacio, observaciones fuera de los límites
    declarados, valores devueltos en el orden equivocado. Conviene
    ejecutarlo siempre después de tocar el entorno: un fallo aquí se
    manifestaría durante el entrenamiento como una falta de convergencia
    difícil de diagnosticar.
    """
    from stable_baselines3.common.env_checker import check_env

    env = crear_env(args)
    try:
        check_env(env, warn=True)
        print('\nEl entorno cumple la interfaz de Gymnasium.')
    finally:
        env.close()


def modo_aleatorio(args):
    """Ejecuta episodios con acciones aleatorias.

    Establece la línea base. También sirve para comprobar que el
    simulador responde y que el reposicionamiento entre episodios
    funciona.
    """
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
        # Monitor registra la recompensa y la longitud de cada episodio.
        # Sin él, las columnas ep_rew_mean y ep_len_mean del registro que
        # imprime Stable-Baselines3 aparecen vacías.
        env = Monitor(env)

        modelo = construir_algoritmo(args.algoritmo, env)

        print('\nEntrenando {} durante {} pasos.'
              .format(args.algoritmo.upper(), args.pasos))
        print('Se puede interrumpir con Ctrl-C: el modelo se guarda igual.\n')

        t0 = time.time()
        try:
            modelo.learn(total_timesteps=args.pasos, progress_bar=False)
        except KeyboardInterrupt:
            print('\nEntrenamiento interrumpido por el usuario.')

        ruta = RUTA_MODELO + '_' + args.algoritmo
        modelo.save(ruta)
        print('\nModelo guardado en {}.zip'.format(ruta))
        print('Tiempo empleado: {:.1f} min'.format((time.time() - t0) / 60.0))
    finally:
        env.close()


def modo_evaluar(args):
    env = crear_env(args)
    try:
        ruta = RUTA_MODELO + '_' + args.algoritmo
        if not os.path.exists(ruta + '.zip'):
            raise SystemExit(
                'No existe {}.zip. Hay que entrenar primero.'.format(ruta))

        modelo = cargar_algoritmo(args.algoritmo, ruta, env)

        # determinístico=True toma la acción de mayor probabilidad en
        # lugar de muestrear de la distribución. Durante el
        # entrenamiento se muestrea (hace falta explorar); al evaluar,
        # no.
        resumen(env,
                politica=lambda obs: modelo.predict(obs, deterministic=True)[0],
                episodios=args.episodios,
                titulo='POLÍTICA APRENDIDA ({})'.format(args.algoritmo.upper()))
    finally:
        env.close()


# =====================================================================
# Ejecución de episodios y métricas
# =====================================================================

def resumen(env, politica, episodios, titulo):
    recompensas = []
    longitudes = []
    choques = 0
    resueltos = 0

    print('\n' + titulo)
    print('-' * len(titulo))

    for ep in range(episodios):
        obs, _ = env.reset()
        total = 0.0
        pasos = 0
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

        recompensas.append(total)
        longitudes.append(pasos)
        if info.get('choque'):
            choques += 1
            desenlace = 'choque'
        elif info.get('resuelto'):
            resueltos += 1
            desenlace = 'resuelto'
        else:
            desenlace = 'tiempo agotado'

        print('episodio {:2d}   pasos {:3d}   recompensa {:8.2f}   '
              'proximidad final {:.3f}   {}'
              .format(ep + 1, pasos, total, info.get('proximidad', 0.0),
                      desenlace))

    print('\nrecompensa media  {:.2f}  (desviación {:.2f})'
          .format(float(np.mean(recompensas)), float(np.std(recompensas))))
    print('longitud media    {:.1f} pasos'.format(float(np.mean(longitudes))))
    print('resueltos         {}/{}'.format(resueltos, episodios))
    print('choques           {}/{}'.format(choques, episodios))


# =====================================================================
# Construcción del entorno y línea de órdenes
# =====================================================================

def crear_env(args):
    return RoboboParedEnv(
        pasos_max=args.pasos_max,
        usar_simulador=not args.sin_simulador,
        verbose=args.detalle)


def main():
    p = argparse.ArgumentParser(
        description='Práctica 1 de RIA: aproximación a una pared con '
                    'aprendizaje por refuerzo en espacios continuos.')

    modo = p.add_mutually_exclusive_group(required=True)
    modo.add_argument('--calibrar', action='store_true',
                      help='imprime las lecturas de los infrarrojos')
    modo.add_argument('--comprobar', action='store_true',
                      help='verifica la interfaz del entorno')
    modo.add_argument('--aleatorio', action='store_true',
                      help='ejecuta episodios con acciones aleatorias')
    modo.add_argument('--entrenar', action='store_true')
    modo.add_argument('--evaluar', action='store_true')

    p.add_argument('--algoritmo', choices=['sac', 'ppo'], default='sac')
    p.add_argument('--pasos', type=int, default=10_000,
                   help='pasos totales de entrenamiento')
    p.add_argument('--episodios', type=int, default=5)
    p.add_argument('--pasos-max', type=int, default=200,
                   help='longitud máxima de un episodio')
    p.add_argument('--sin-simulador', action='store_true',
                   help='no usar el modulo sim; el robot se repone a mano')
    p.add_argument('--detalle', action='store_true',
                   help='imprime una línea por paso')

    args = p.parse_args()

    if args.calibrar:
        calibrar()
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
