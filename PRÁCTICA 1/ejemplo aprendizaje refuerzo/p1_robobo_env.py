#!/usr/bin/env python3
"""
Práctica 1 — Entorno Gymnasium para el Robobo en RoboboSim.

Tarea: aproximarse a una pared y detenerse cerca de ella sin llegar a
chocar. El escenario es una superficie plana delimitada por cuatro
paredes, sin obstáculos intermedios.

Este archivo contiene únicamente el entorno. El entrenamiento está en
p1_entrenar.py.

--------------------------------------------------------------------
ESTRUCTURA
--------------------------------------------------------------------

El entorno cumple la interfaz de Gymnasium (reset / step / close) y
combina dos canales de comunicación distintos:

  base/ y smartphone/   percepción y actuación; es lo único que existe
                        también en el robot real.
  sim/                  reinicio de la escena al comienzo de cada
                        episodio y posición real del robot. Es
                        información y control privilegiados del
                        simulador, que el robot real no tiene.

Los dos canales llegan por ROS 2, del mismo nodo puente: el módulo sim
de robobo_ros2 publica la posición del robot y ofrece los servicios de
colocación y de reinicio de la escena, así que ya no hace falta instalar
la biblioteca robobosim.

Regla que conviene respetar: la información privilegiada del simulador
puede emplearse para reiniciar episodios y para registrar métricas, pero
NO debe formar parte del espacio de observación. Si la política aprende a
partir de datos que sólo existen en simulación, no podrá transferirse al
robot real.

--------------------------------------------------------------------
REQUISITOS
--------------------------------------------------------------------

1. RoboboSim en ejecución en el equipo anfitrión, con un escenario plano.

2. El nodo puente en ejecución en otra terminal del contenedor, con el
   módulo del simulador cargado:

       ros2 run robobo_ros2 robobo_container --ros-args \
            -p ip:=host.docker.internal -p modules:="['sim']"

   Conviene cargar el mínimo de módulos posible: cada módulo añade nodos
   y temporizadores que compiten por el mismo enlace con el simulador.
   El módulo sim es el que da acceso a la escena; sin él, el entorno no
   puede reiniciar los episodios.

--------------------------------------------------------------------
PARÁMETROS QUE HAY QUE CALIBRAR
--------------------------------------------------------------------

Los valores de IR_SATURACION, P_OBJETIVO y P_CHOQUE dependen del
escenario y de la versión del simulador. Antes de entrenar es obligatorio
ejecutar

    python3 p1_entrenar.py --calibrar

y ajustarlos con las lecturas obtenidas.
"""

import time

import numpy as np

import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node

from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import Twist

# Interfaces propias del puente. El módulo sim las usa para hablar con
# el simulador; no existen cuando se trabaja con el robot real.
from robobo_ros2_interfaces.msg import RobotLocation
from robobo_ros2_interfaces.srv import ResetSimulation


# =====================================================================
# Constantes del problema
# =====================================================================

# Espacio de nombres del nodo puente. Con los valores por defecto de
# robobo_container, robot_name vale "0".
NS_BASE = '/robobo/robot_0/base'

# Espacio de nombres del módulo sim, dentro del mismo nodo puente. Es el
# canal privilegiado: existe en el simulador y no en el robot real.
NS_SIM = '/robobo/robot_0/sim'

# Orden de los ocho valores del array publicado en <NS_BASE>/ir:
#   0 FrontLL  1 FrontL  2 FrontC  3 FrontR  4 FrontRR
#   5 BackL    6 BackC   7 BackR
IR_FRONTALES = [0, 1, 2, 3, 4]

# Los infrarrojos del Robobo son sensores de proximidad, no de distancia:
# el valor CRECE al acercarse el obstáculo. El valor bruto es un entero
# sin unidad física. Se normaliza dividiendo por esta constante y
# recortando a [0, 1].
IR_SATURACION = 1000.0

# Proximidad frontal que se considera "cerca de la pared" (objetivo) y
# proximidad a partir de la cual se da el episodio por fracasado.
P_OBJETIVO = 0.55
P_TOLERANCIA = 0.08
P_CHOQUE = 0.85

# Número de pasos consecutivos dentro de la banda objetivo que se exigen
# para dar la tarea por resuelta.
PASOS_EN_BANDA = 10

# Límites de la orden de velocidad enviada en cada paso.
V_MAX = 0.15      # m/s hacia delante
V_MIN = -0.08     # m/s hacia atrás
W_MAX = 1.0       # rad/s

# Duración de la aplicación de cada acción. Debe ser menor que el
# parámetro cmd_vel_timeout del puente (0.5 s por defecto); en caso
# contrario el perro guardián detiene las ruedas entre paso y paso y el
# movimiento resulta entrecortado.
PASO_S = 0.2

# Espera máxima por una lectura nueva de los infrarrojos antes de dar el
# episodio por perdido. Los infrarrojos se publican a 10 Hz.
ESPERA_MAX_S = 3.0


# =====================================================================
# El entorno
# =====================================================================

class RoboboParedEnv(gym.Env):
    """Aproximación a una pared con el Robobo en RoboboSim.

    Observación (Box, float32, dimensión 7)
        [0..4]  proximidad normalizada de los cinco infrarrojos frontales
                (FrontLL, FrontL, FrontC, FrontR, FrontRR), en [0, 1]
        [5]     velocidad lineal ordenada en el paso anterior, en [-1, 1]
        [6]     velocidad angular ordenada en el paso anterior, en [-1, 1]

        Las dos últimas componentes no son estrictamente necesarias, pero
        hacen el estado observable: sin ellas, dos situaciones con las
        mismas lecturas pero distinta inercia son indistinguibles.

    Acción (Box, float32, dimensión 2)
        [0]     velocidad lineal, en [-1, 1], reescalada a [V_MIN, V_MAX]
        [1]     velocidad angular, en [-1, 1], reescalada a [-W_MAX, W_MAX]

        El espacio es continuo, que es lo que exigen PPO y SAC en su
        formulación para control continuo.

    Recompensa
        Definida en _recompensa(). Es la parte que se pide modificar en
        el enunciado de la práctica.
    """

    metadata = {'render_modes': []}

    def __init__(self,
                 pasos_max=200,
                 usar_simulador=True,
                 verbose=False):
        super().__init__()

        self.pasos_max = pasos_max
        self.verbose = verbose

        # -------------------------------------------------- ROS 2
        # rclpy.init() sólo puede llamarse una vez por proceso. La guarda
        # permite crear el entorno varias veces (por ejemplo al evaluar
        # después de entrenar) sin reiniciar el intérprete.
        if not rclpy.ok():
            rclpy.init()

        self.nodo = Node('p1_entorno_rl')

        # El publicador y la suscripción se crean UNA sola vez, aquí. Un
        # error habitual es crearlos dentro de step(): el descubrimiento
        # de ROS 2 tarda cientos de milisegundos y las primeras órdenes
        # se perderían.
        self.pub_vel = self.nodo.create_publisher(Twist, NS_BASE + '/cmd_vel', 10)

        self._ir = None          # última lectura recibida
        self._n_ir = 0           # número de lecturas recibidas
        self.nodo.create_subscription(
            Int32MultiArray, NS_BASE + '/ir', self._cb_ir, 1)

        # -------------------------------------------------- simulador
        # Canal privilegiado, también por ROS 2: un servicio para
        # reiniciar la escena y un tópico con la posición real del robot.
        # Nada de esto existe en el robot real, y por eso no entra en la
        # observación.
        self.cli_reset = None
        self._pose = None
        if usar_simulador:
            self.cli_reset = self.nodo.create_client(
                ResetSimulation, NS_SIM + '/reset_simulation')
            if not self.cli_reset.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(
                    'No responde el servicio {}/reset_simulation.\n'
                    'Comprobar que robobo_container se ha lanzado con el '
                    'módulo sim, o crear el entorno con '
                    'usar_simulador=False y reponer el robot a mano.'
                    .format(NS_SIM))
            self.nodo.create_subscription(
                RobotLocation, NS_SIM + '/robot_location', self._cb_pose, 1)

        # -------------------------------------------------- espacios
        self.observation_space = spaces.Box(
            low=np.array([0.0] * 5 + [-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0] * 5 + [1.0, 1.0], dtype=np.float32),
            dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # -------------------------------------------------- estado
        self.pasos = 0
        self.en_banda = 0
        self.ultima_accion = np.zeros(2, dtype=np.float32)

        # Esperar a que llegue la primera lectura: confirma que el puente
        # está en marcha antes de que el algoritmo empiece a entrenar.
        self._esperar_lectura_nueva(inicial=True)

    # -----------------------------------------------------------------
    # Comunicación con ROS 2
    # -----------------------------------------------------------------

    def _cb_ir(self, msg):
        """Función de retorno de la suscripción a los infrarrojos.

        Se ejecuta cuando el ejecutor procesa un mensaje pendiente, es
        decir, dentro de alguna llamada a rclpy.spin_once(). Nunca se
        ejecuta "por su cuenta": si no se hace spin, los mensajes se
        acumulan en la cola y la observación se queda congelada.
        """
        self._ir = list(msg.data)
        self._n_ir += 1

    def _vaciar_cola(self):
        """Descarta los mensajes pendientes.

        Se llama justo antes de pedir la observación del paso. Todo lo
        que hay en la cola en ese momento se publicó ANTES o DURANTE la
        aplicación de la acción, y por tanto no refleja necesariamente su
        efecto completo.
        """
        for _ in range(100):
            antes = self._n_ir
            rclpy.spin_once(self.nodo, timeout_sec=0.0)
            if self._n_ir == antes:
                return

    def _esperar_lectura_nueva(self, inicial=False):
        """Bloquea hasta recibir una lectura POSTERIOR a este instante.

        Éste es el punto delicado de acoplar Gymnasium con ROS 2.
        Gymnasium exige que step() devuelva la observación resultante de
        la acción; ROS 2 entrega los mensajes cuando quiere. Si se
        devuelve sin más la última lectura guardada, se devuelve una
        observación ANTERIOR a la acción aplicada.

        El fallo no produce ningún error: el entrenamiento se ejecuta
        con normalidad y simplemente no converge, porque la pareja
        (observación, acción) que recibe el algoritmo está desplazada un
        paso en el tiempo.
        """
        objetivo = self._n_ir + 1
        t0 = time.time()
        while self._n_ir < objetivo:
            rclpy.spin_once(self.nodo, timeout_sec=0.05)
            if time.time() - t0 > ESPERA_MAX_S:
                raise RuntimeError(
                    'No llegan lecturas de {}/ir.\n'
                    'Comprobar que robobo_container está en ejecución y '
                    'conectado al simulador.'.format(NS_BASE)
                    if inicial else
                    'Se ha interrumpido el flujo de lecturas de los '
                    'infrarrojos durante el episodio.')

    def _publicar_velocidad(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub_vel.publish(msg)

    def _detener(self):
        """Detiene las ruedas y espera a que la orden surta efecto."""
        for _ in range(3):
            self._publicar_velocidad(0.0, 0.0)
            time.sleep(0.05)

    # -----------------------------------------------------------------
    # Canal privilegiado del simulador
    # -----------------------------------------------------------------

    def _cb_pose(self, msg):
        """Posición real del robot, en milímetros y grados.

        Verdad del terreno: sirve para registrar métricas y para calcular
        recompensas, nunca para construir la observación.
        """
        self._pose = (msg.position.x, msg.position.z, msg.rotation.y)

    def _reiniciar_escena(self):
        """Devuelve la escena a su estado inicial.

        Es una llamada a servicio, no un mensaje: hay que esperar la
        respuesta haciendo spin, porque quien procesa la respuesta es el
        mismo hilo que ejecuta el bucle de entrenamiento.
        """
        futuro = self.cli_reset.call_async(ResetSimulation.Request())
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=5.0)
        if futuro.result() is None or not futuro.result().success:
            raise RuntimeError('El reinicio de la escena ha fallado.')

    # -----------------------------------------------------------------
    # Observación
    # -----------------------------------------------------------------

    def _observacion(self):
        crudos = [self._ir[i] for i in IR_FRONTALES]
        norm = [min(max(v, 0) / IR_SATURACION, 1.0) for v in crudos]
        return np.array(norm + list(self.ultima_accion), dtype=np.float32)

    def _proximidad_frontal(self):
        """Proximidad del obstáculo más cercano por delante, en [0, 1].

        Se toma el máximo de los cinco frontales: basta con que un sensor
        vea la pared para que el robot esté cerca de ella.
        """
        return float(max(min(max(self._ir[i], 0) / IR_SATURACION, 1.0)
                         for i in IR_FRONTALES))

    # -----------------------------------------------------------------
    # Recompensa
    # -----------------------------------------------------------------

    def _recompensa(self, p, v, w, choque, resuelto):
        """Recompensa del paso.

        ------------------------------------------------------------
        ESTA ES LA FUNCIÓN QUE SE PIDE ESTUDIAR Y MODIFICAR.
        ------------------------------------------------------------

        La versión que sigue es deliberadamente sencilla y tiene defectos
        reconocibles. Consta de cuatro términos:

          1. Conformación (shaping): penaliza la distancia entre la
             proximidad actual y la deseada. Es lo que da señal en cada
             paso; sin ella el problema sería de recompensa dispersa y
             ninguno de los dos algoritmos lo resolvería en un número
             razonable de pasos.

          2. Regularización del giro: penaliza el giro, que en esta
             tarea no aporta nada. Sin este término la política aprende
             a girar sobre sí misma, porque así encuentra paredes sin
             arriesgarse a chocar.

          3. Penalización terminal por choque.

          4. Recompensa terminal por permanecer en la banda objetivo.

        El argumento v no se utiliza en esta versión. Se pasa porque es
        el primer sitio donde se echa en falta al modificar la función:
        penalizar la velocidad cuando la proximidad es alta es una de
        las formas más directas de conseguir que el robot frene en lugar
        de detenerse de golpe.
        """
        r = -abs(p - P_OBJETIVO)
        r -= 0.05 * abs(w) / W_MAX

        if choque:
            r -= 10.0
        if resuelto:
            r += 10.0

        return float(r)

    # -----------------------------------------------------------------
    # Interfaz de Gymnasium
    # -----------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._detener()

        if self.cli_reset is not None:
            # La escena vuelve a su estado inicial, de modo que todos los
            # episodios empiezan en la misma pose. Es lo más sencillo,
            # pero también la razón de que la política pueda aprenderse
            # una secuencia fija de acciones en lugar de reaccionar a lo
            # que ve. Aleatorizar la pose de partida, con el servicio
            # sim/set_robot_location, es la primera mejora que conviene
            # probar; para eso hay que medir antes los límites del
            # escenario, que están en milímetros.
            self._reiniciar_escena()
            time.sleep(0.5)

        self.pasos = 0
        self.en_banda = 0
        self.ultima_accion = np.zeros(2, dtype=np.float32)

        # Descartar lo acumulado durante el reposicionamiento y esperar
        # una lectura tomada ya en la posición nueva.
        self._vaciar_cola()
        self._esperar_lectura_nueva()

        return self._observacion(), {}

    def step(self, accion):
        accion = np.clip(np.asarray(accion, dtype=np.float32), -1.0, 1.0)
        self.ultima_accion = accion

        # 1. Reescalar la acción normalizada a unidades del SI y enviarla.
        #    El puente traduce el Twist a velocidades de rueda.
        v = V_MIN + (accion[0] + 1.0) * 0.5 * (V_MAX - V_MIN)
        w = float(accion[1]) * W_MAX
        self._publicar_velocidad(v, w)

        # 2. Dejar que la acción surta efecto. Durante esta espera no se
        #    hace spin: los mensajes que llegan se acumulan y se
        #    descartarán en el paso siguiente.
        time.sleep(PASO_S)

        # 3. Vaciar la cola y esperar una lectura posterior a la acción.
        self._vaciar_cola()
        self._esperar_lectura_nueva()

        # 4. Evaluar el resultado.
        self.pasos += 1
        p = self._proximidad_frontal()

        choque = p >= P_CHOQUE

        if abs(p - P_OBJETIVO) <= P_TOLERANCIA:
            self.en_banda += 1
        else:
            self.en_banda = 0
        resuelto = self.en_banda >= PASOS_EN_BANDA

        recompensa = self._recompensa(p, v, w, choque, resuelto)

        terminated = bool(choque or resuelto)
        truncated = bool(self.pasos >= self.pasos_max)

        if terminated or truncated:
            self._detener()

        info = {
            'proximidad': p,
            'ir_crudos': [self._ir[i] for i in IR_FRONTALES],
            'choque': choque,
            'resuelto': resuelto,
            # Verdad del terreno, sólo para registro: (x, z) en mm y
            # rotación en grados. Vale None si el módulo sim no está.
            'pose_sim': self._pose,
        }

        if self.verbose:
            print('paso {:3d}  p={:.3f}  v={:+.3f}  w={:+.3f}  r={:+.3f}{}'
                  .format(self.pasos, p, v, w, recompensa,
                          '  CHOQUE' if choque else
                          ('  RESUELTO' if resuelto else '')))

        return self._observacion(), recompensa, terminated, truncated, info

    def close(self):
        try:
            self._detener()
        except Exception:
            pass
        try:
            self.nodo.destroy_node()
        except Exception:
            pass
        # No se llama a rclpy.shutdown() aquí: otros entornos creados en
        # el mismo proceso seguirían necesitando el contexto.


# =====================================================================
# Utilidad de calibración
# =====================================================================

def calibrar():
    """Imprime las lecturas de los infrarrojos frontales de forma continua.

    Sirve para fijar IR_SATURACION, P_OBJETIVO y P_CHOQUE. Procedimiento:

      1. Ejecutarla con el robot lejos de cualquier pared y anotar el
         valor de reposo (debería ser próximo a cero).
      2. Acercar el robot a una pared con las flechas del simulador hasta
         la distancia que se considere "cerca sin chocar" y anotar el
         valor. Ése es P_OBJETIVO multiplicado por IR_SATURACION.
      3. Llevar el robot hasta tocar la pared y anotar el valor máximo.
         Ése es el valor que conviene usar como IR_SATURACION.

    Se interrumpe con Ctrl-C.
    """
    if not rclpy.ok():
        rclpy.init()
    nodo = Node('p1_calibracion')

    estado = {'ir': None}
    nodo.create_subscription(
        Int32MultiArray, NS_BASE + '/ir',
        lambda m: estado.__setitem__('ir', list(m.data)), 1)

    nombres = ['FrontLL', 'FrontL', 'FrontC', 'FrontR', 'FrontRR']
    print('  '.join('{:>8s}'.format(n) for n in nombres) + '   proximidad')
    try:
        while rclpy.ok():
            rclpy.spin_once(nodo, timeout_sec=0.5)
            if estado['ir'] is None:
                continue
            crudos = [estado['ir'][i] for i in IR_FRONTALES]
            p = max(min(max(v, 0) / IR_SATURACION, 1.0) for v in crudos)
            print('  '.join('{:8d}'.format(v) for v in crudos)
                  + '   {:.3f}'.format(p))
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
