#!/usr/bin/env python3
"""
Práctica 1 — Entorno Gymnasium para el Robobo en RoboboSim.

Tarea: nuestro robot (robot_0) debe perseguir a otro robot (robot_1) que lleva pegado un blob de color verde fácilmente
distinguible por la cámara, manteniéndose siempre a una distancia d de él.


--------------------------------------------------------------------
PARÁMETROS QUE HAY QUE CALIBRAR
--------------------------------------------------------------------

Es necesario calibrar BLOB_TAM_SATURACION, TAM_OBJETIVO y TAM_CHOQUE a partir de

    python3 entrenar_seguir.py --calibrar

"""

import time

import numpy as np

import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node

# Interfaces propias del puente para poder reiniciar la escena
try:
    from robobo_ros2_interfaces.msg import RobotLocation
    from robobo_ros2_interfaces.srv import ResetSimulation
    HAY_INTERFACES_SIM = True
except ImportError:
    RobotLocation = ResetSimulation = None
    HAY_INTERFACES_SIM = False


try:
    from robobo_ros2_interfaces.srv import StopWheels, MoveWheels
    from robobo_ros2_interfaces.msg import BlobArray
    HAY_INTERFACES_ROBOT = True
except ImportError:
    StopWheels = MoveWheels = BlobArray = None
    HAY_INTERFACES_ROBOT = False

AVISO_SIN_SIM = (
    'No se encuentran las interfaces del módulo sim '
    '(robobo_ros2_interfaces.msg.RobotLocation).\n'
    'Suele ser una de estas dos cosas:\n'
    '  1. La terminal no tiene cargado el espacio de trabajo del puente:\n'
    '         source /opt/ros/robobo/setup.bash\n'
    '  2. La imagen del contenedor es anterior al módulo sim y hay que\n'
    '     reconstruirla.\n'
    'Mientras tanto, el entorno funciona con usar_simulador=False '
    '(--sin-simulador),\ncon los robots repuestos a mano entre episodios.')

AVISO_SIN_ROBOT = (
    'No se encuentran robobo_ros2_interfaces.srv.MoveWheels/StopWheels o '
    'robobo_ros2_interfaces.msg.BlobArray.\n'
    'La terminal no tiene cargado el espacio de trabajo del puente, o el '
    'paquete de interfaces instalado es anterior a estos mensajes:\n'
    '         source /opt/ros/robobo/setup.bash\n')


# =====================================================================
# Constantes del problema
# =====================================================================

# Espacio de nombres de nuestro robot
NS_BASE = '/robobo/robot_0/base'
NS_SMARTPHONE = '/robobo/robot_0/smartphone'
NS_SIM = '/robobo/robot_0/sim'

# Espacio de nombres del robot objetivo. Solo se usa para leer su posición.
NS_SIM_ROBOT_OBJETIVO = '/robobo/robot_1/sim'

# BLOB VERDE

# Color por el que se filtra dentro de BlobArray.msg.blobs.
COLOR_BLOB = 'green'

# posx del blob en [0, 100], donde 50 es el centro de la imagen.
BLOB_X_CENTRO = 50.0

# El tamaño (área en píxeles) crece al acercarse. Hay que normalizarla
# dividiendo por la constante calibrada y recortando a [0, 1]
BLOB_TAM_SATURACION = 8000.0

# Tamaño de blob que corresponde a la distancia que se quiere mantener,
# y tolerancia alrededor de ese valor. La hay que calibrar
TAM_OBJETIVO = 0.30
TAM_TOLERANCIA = 0.06

# Tamaño de blob a partir del cual se considera que ha colisionado
# con el otro robot. La hay que calibrar
TAM_CHOQUE = 0.75

# Desviación horizontal del blob que se tolera 
# En las mismas unidades que blob_x normalizado ([-1, 1]).
X_TOLERANCIA = 0.15

# Pasos consecutivos sin ver el blob antes de dar el episodio por perdido.
PASOS_PERDIDO_MAX = 15

# Tiempo máximo desde la última detección del blob para seguir
# considerándolo "visible ahora mismo"
BLOB_VIGENCIA_S = 0.4


# Límites de velocidad de cada rueda. La acción ya no es (v, w): el
# robot sólo tiene dos motores, uno por rueda, así que la política
# decide directamente la velocidad de cada una.
V_RUEDA_MAX = 0.20
V_RUEDA_MIN = -0.15


PASO_S = 0.2

# =====================================================================
# El entorno
# =====================================================================

class RoboboSeguimientoEnv(gym.Env):
    """Seguimiento de un robot con blob verde, con el Robobo en RoboboSim.

    Observación (Box, float32, dimensión )
        [0]  posición horizontal del blob, normalizada a [-1, 1]
             (0 = centrado, -1 = borde izquierdo, +1 = borde derecho)
        [1]  tamaño del blob normalizado, en [0, 1] (mayor = más cerca)
        [2]  blob visible ahora mismo: 1.0 si sí, 0.0 si no
        [3]  velocidad de la rueda izquierda en el paso anterior, en [-1, 1]
        [4]  velocidad de la rueda derecha en el paso anterior, en [-1, 1]

        Cuando el blob no es visible, las componentes [0] y [1] se
        ponen a 0, y la componente [2] ya indica que no es visible

    Acción (Box, float32, dimensión 2)
        [0]  velocidad de la rueda izquierda, en [-1, 1], reescalada a
             [V_RUEDA_MIN, V_RUEDA_MAX]
        [1]  velocidad de la rueda derecha, en [-1, 1], reescalada igual

        Sin componente de velocidad angular: control puramente
        diferencial, como el robot real. Se convierte a (v, w) sólo
        para publicar en cmd_vel (ver step()); la política nunca ve v
        ni w.

    Recompensa
        Definida en _recompensa(). Mantiene penalización entre "mantener la
        distancia" y "mantener el centrado", y la penalización por perder
        de vista al otro robot.
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
        if not rclpy.ok():
            rclpy.init()

        self.nodo = Node('seguir_entorno_rl')

        if not HAY_INTERFACES_ROBOT:
            raise RuntimeError(AVISO_SIN_ROBOT)

        self._blob_posx = None
        self._blob_tam = None
        self._blob_t_ultimo = -1.0   # time.time() de la última detección

        self.nodo.create_subscription(
            BlobArray, NS_SMARTPHONE + '/color_blobs', self._cb_blob, 10)

        self.cli_move_wheels = self.nodo.create_client(
            MoveWheels, NS_BASE + '/move_wheels')
        self.cli_stop_wheels = self.nodo.create_client(
            StopWheels, NS_BASE + '/stop_wheels')
        # Usamos la disponibilidad de move_wheels como comprobación de
        # que robobo_container está en marcha (ya no hay IR para eso).
        if not self.cli_move_wheels.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(
                'No responde el servicio {}/move_wheels.\n'
                'Comprobar que robobo_container está en marcha.'
                .format(NS_BASE))

        # -------------------------------------------------- simulador
        self.cli_reset = None
        self._pose = None
        self._pose_robot_objetivo = None
        if usar_simulador and not HAY_INTERFACES_SIM:
            raise RuntimeError(AVISO_SIN_SIM)
        if usar_simulador:
            self.cli_reset = self.nodo.create_client(
                ResetSimulation, NS_SIM + '/reset_simulation')
            if not self.cli_reset.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(
                    'No responde el servicio {}/reset_simulation.\n'
                    'Comprobar que robobo_container se ha lanzado con el '
                    'módulo sim, o crear el entorno con '
                    'usar_simulador=False y reponer los robots a mano.'
                    .format(NS_SIM))
            self.nodo.create_subscription(
                RobotLocation, NS_SIM + '/robot_location', self._cb_pose, 1)
            # Posición del otro robot, si su nombre es
            # distinto (otro robot_name), ajustar NS_SIM_ROBOT_OBJETIVO arriba.
            self.nodo.create_subscription(
                RobotLocation, NS_SIM_ROBOT_OBJETIVO + '/robot_location',
                self._cb_pose_robot_objetivo, 1)

        # -------------------------------------------------- espacios
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # -------------------------------------------------- estado
        self.pasos = 0
        self.en_banda = 0
        self.pasos_perdido = 0
        self.ultima_accion = np.zeros(2, dtype=np.float32)

    # -----------------------------------------------------------------
    # Comunicación con ROS 2
    # -----------------------------------------------------------------

    def _cb_blob(self, msg):
        """Se ejecuta cada vez que llega una detección del blob verde.

        Este topic no es periódico: sólo publica
        cuando hay blob detectado. Por eso no se cuenta "lecturas nuevas"
        para esperar aquí; se guarda el momento de la detección y
        _blob_visible() decide, por antigüedad, si todavía es de fiar.
        """
        for blob in msg.blobs:
            if blob.color == COLOR_BLOB:
                self._blob_x = float(blob.x)
                self._blob_tam = float(blob.size)
                self._blob_t_ultimo = time.time()
                break

    def _cb_pose(self, msg):
        self._pose = (msg.position.x, msg.position.z, msg.rotation.y)

    def _cb_pose_robot_objetivo(self, msg):
        self._pose_robot_objetivo = (msg.position.x, msg.position.z, msg.rotation.y)

    def _mover_ruedas(self, v_izq, v_der):
        peticion = MoveWheels.Request()
        peticion.right_speed = float(v_der)
        peticion.left_speed = float(v_izq)
        futuro = self.cli_move_wheels.call_async(peticion)
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=1.0)

    def _detener(self):
        futuro = self.cli_stop_wheels.call_async(StopWheels.Request())
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=1.0)

    def _reiniciar_escena(self):
        futuro = self.cli_reset.call_async(ResetSimulation.Request())
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=5.0)
        if futuro.result() is None or not futuro.result().success:
            raise RuntimeError('El reinicio de la escena ha fallado.')

    # -----------------------------------------------------------------
    # Observación
    # -----------------------------------------------------------------

    def _blob_visible(self):
        if self._blob_t_ultimo < 0:
            return False
        return (time.time() - self._blob_t_ultimo) <= BLOB_VIGENCIA_S

    def _leer_blob(self):
        """Devuelve (visible, blob_x, blob_tam) ya normalizados."""
        visible = self._blob_visible()
        if visible:
            blob_x = float(np.clip(
                (self._blob_x - BLOB_X_CENTRO) / BLOB_X_CENTRO, -1.0, 1.0))
            blob_tam = float(np.clip(
                self._blob_tam / BLOB_TAM_SATURACION, 0.0, 1.0))
        else:
            blob_x, blob_tam = 0.0, 0.0
        return visible, blob_x, blob_tam


    def _observacion(self):
        visible, blob_x, blob_tam = self._leer_blob()
        return np.array(
            [blob_x, blob_tam, 1.0 if visible else 0.0] +
            list(self.ultima_accion),
            dtype=np.float32)

    # -----------------------------------------------------------------
    # Recompensa
    # -----------------------------------------------------------------

    def _recompensa(self, blob_x, tam, visible, accion, perdido, choque, en_banda):
        """Recompensa del paso

        Cuatro términos:

          1. Conformación: si el blob es visible, penaliza alejarse del
             tamaño objetivo (mantener la distancia d) y alejarse del
             centro (mantenerse detrás del robot, no a un lado). Si no es
             visible, no hay señal de conformación posible y se aplica una
             penalización fija: evita que la política aprenda a "mirar
             para otro lado" para no acumular penalización de centrado.

          2. Regularización del giro: se usa la diferencia entre la velocidad 
             de las dos ruedas (accion está normalizada en [-1, 1] por rueda, 
             así que la diferencia máxima es 2).

          3. Penalización fuerte por choque (el blob
             ocupa ya casi toda la imagen)

          4. Pequeña recompensa por paso mientras se está dentro de la
             banda objetivo (distancia y centrado a la vez)

        """
        if not visible:
            r = -1.0
        else:
            r = -abs(tam - TAM_OBJETIVO)
            r -= 0.5 * abs(blob_x)

        diferencia_ruedas = abs(float(accion[1]) - float(accion[0])) / 2.0
        r -= 0.02 * diferencia_ruedas

        if choque:
            r -= 10.0
        if perdido:
            r -= 5.0
        if en_banda:
            r += 0.3

        return float(r)

    # -----------------------------------------------------------------
    # Interfaz de Gymnasium
    # -----------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._detener()

        if self.cli_reset is not None:
            # Reinicia la escena completa
            self._reiniciar_escena()
            time.sleep(0.5)

        self.pasos = 0
        self.en_banda = 0
        self.pasos_perdido = 0
        self.ultima_accion = np.zeros(2, dtype=np.float32)

        # Se limpia la última detección para no arrastrar una lectura de antes del
        # reinicio, y se confía en _blob_visible() para el resto.
        self._blob_t_ultimo = -1.0
        rclpy.spin_once(self.nodo, timeout_sec=0.1)

        return self._observacion(), {}

    def step(self, accion):
        accion = np.clip(np.asarray(accion, dtype=np.float32), -1.0, 1.0)
        self.ultima_accion = accion

        # Reescalar velocidad de cada rueda por separado
        v_izq = V_RUEDA_MIN + (accion[0] + 1.0) * 0.5 * (V_RUEDA_MAX - V_RUEDA_MIN)
        v_der = V_RUEDA_MIN + (accion[1] + 1.0) * 0.5 * (V_RUEDA_MAX - V_RUEDA_MIN)
        self._mover_ruedas(v_izq, v_der)
        
        # Como el topic de blobs es por eventos hace falta seguir haciendo
        # spin durante toda la ventana del paso, o se puede perder la
        # única detección que llegue en ese intervalo.
        t0 = time.time()
        while time.time() - t0 < PASO_S:
            rclpy.spin_once(self.nodo, timeout_sec=0.02)

        self.pasos += 1

        visible, blob_x, blob_tam = self._leer_blob()
        if visible:
            self.pasos_perdido = 0
        else:
            self.pasos_perdido += 1

        choque = visible and blob_tam >= TAM_CHOQUE
        perdido = self.pasos_perdido >= PASOS_PERDIDO_MAX

        en_banda_ahora = (visible
                          and abs(blob_tam - TAM_OBJETIVO) <= TAM_TOLERANCIA
                          and abs(blob_x) <= X_TOLERANCIA)
        self.en_banda = self.en_banda + 1 if en_banda_ahora else 0

        recompensa = self._recompensa(
            blob_x, blob_tam, visible, accion, perdido, choque, en_banda_ahora)

        terminated = bool(choque or perdido)
        truncated = bool(self.pasos >= self.pasos_max)

        if terminated or truncated:
            self._detener()

        distancia_real = None
        if self._pose is not None and self._pose_robot_objetivo is not None:
            distancia_real = float(np.hypot(
                self._pose[0] - self._pose_robot_objetivo[0],
                self._pose[1] - self._pose_robot_objetivo[1]))

        info = {
            'blob_x': blob_x,
            'blob_tam': blob_tam,
            'visible': visible,
            'choque': choque,
            'perdido': perdido,
            'en_banda_consecutivos': self.en_banda,
            'pose_sim': self._pose,
            'pose_sim_robot_objetivo': self._pose_robot_objetivo,
            'distancia_real_mm': distancia_real,
        }

        if self.verbose:
            print('paso {:3d}  vis={}  x={:+.3f}  tam={:.3f}  '
                  'v_izq={:+.3f}  v_der={:+.3f}  r={:+.3f}{}'
                  .format(self.pasos, int(visible), blob_x, blob_tam,
                          v_izq, v_der, recompensa,
                          '  CHOQUE' if choque else
                          ('  PERDIDO' if perdido else '')))

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


# =====================================================================
# Utilidad de calibración
# =====================================================================

def calibrar(avanzar=True):
    """Mide blob_x y blob_tam a distintas distancias del robot líder.

    Con avanzar=True (por defecto): asume que el otro está quieto, ya
    colocado delante del seguidor y dentro de su campo de visión, y hace
    que el seguidor avance hacia él a pulsos cortos, imprimiendo una fila
    por pulso con el tamaño del blob. Se detiene si el blob se pierde, si
    el tamaño deja de crecer varios pulsos seguidos, o si indica que está a punto de chocar

    Con avanzar=False: no mueve el robot, sólo imprime blob_x y blob_tam
    en continuo. Útil para calibrar varias distancias moviendo al líder
    entre lecturas sin que el seguidor se mueva de en medio

    De la tabla salen las constantes:

      BLOB_TAM_SATURACION   el tamaño máximo visto, con el líder pegado
                            al seguidor.
      TAM_OBJETIVO          el tamaño de la fila en la que la distancia es
                            la deseada (d), dividido por el anterior.
      TAM_CHOQUE            un valor intermedio entre el objetivo y 1,0.
    """
    if not HAY_INTERFACES_ROBOT:
        raise RuntimeError(AVISO_SIN_ROBOT)

    if not rclpy.ok():
        rclpy.init()
    nodo = Node('calibracion_seguir')

    estado = {'blob_x': None, 'blob_tam': None, 'blob_t': -1.0,
              'pose': None, 'pose_robot_objetivo': None}

    def cb_blob(m):
        for blob in m.blobs:
            if blob.color == COLOR_BLOB:
                estado['blob_x'] = float(blob.x)
                estado['blob_tam'] = float(blob.size)
                estado['blob_t'] = time.time()
                break

    nodo.create_subscription(BlobArray, NS_SMARTPHONE + '/color_blobs', cb_blob, 10)

    if HAY_INTERFACES_SIM:
        nodo.create_subscription(
            RobotLocation, NS_SIM + '/robot_location',
            lambda m: estado.__setitem__('pose', (m.position.x, m.position.z)), 1)
        nodo.create_subscription(
            RobotLocation, NS_SIM_ROBOT_OBJETIVO + '/robot_location',
            lambda m: estado.__setitem__('pose_robot_objetivo', (m.position.x, m.position.z)), 1)

    cli_move_wheels = nodo.create_client(MoveWheels, NS_BASE + '/move_wheels')
    cli_stop_wheels = nodo.create_client(StopWheels, NS_BASE + '/stop_wheels')
    if not cli_move_wheels.wait_for_service(timeout_sec=5.0):
        raise RuntimeError(
            'No responde el servicio {}/move_wheels. ¿Está en marcha '
            'robobo_container?'.format(NS_BASE))

    def blob_visible_ahora():
        return (estado['blob_t'] >= 0
                and (time.time() - estado['blob_t']) <= BLOB_VIGENCIA_S)

    def mover(v, segundos):
        peticion = MoveWheels.Request()
        peticion.right_speed = float(v)
        peticion.left_speed = float(v)
        futuro = cli_move_wheels.call_async(peticion)
        rclpy.spin_until_future_complete(nodo, futuro, timeout_sec=1.0)

        t0 = time.time()
        while time.time() - t0 < segundos:
            rclpy.spin_once(nodo, timeout_sec=0.02)

        futuro = cli_stop_wheels.call_async(StopWheels.Request())
        rclpy.spin_until_future_complete(nodo, futuro, timeout_sec=1.0)

    def distancia_real():
        if estado['pose'] and estado['pose_robot_objetivo']:
            return ((estado['pose'][0] - estado['pose_robot_objetivo'][0]) ** 2 +
                     (estado['pose'][1] - estado['pose_robot_objetivo'][1]) ** 2) ** 0.5
        return None

    cabecera = '{:>4s}  {:>7s}  {:>7s}  {:>4s}  {:>8s}'.format(
        'paso', 'blob_x', 'tam', 'vis', 'distmm')
        
    try:
        if not avanzar:
            print(cabecera)
            while rclpy.ok():
                rclpy.spin_once(nodo, timeout_sec=0.1)
                vis = blob_visible_ahora()
                d = distancia_real()
                print('{:4s}  {:7s}  {:7s}  {:4d}  {:>8s}'.format(
                    '--',
                    '{:.1f}'.format(estado['blob_x']) if vis else '--',
                    '{:.0f}'.format(estado['blob_tam']) if vis else '--',
                    int(vis),
                    '{:.0f}'.format(d) if d is not None else '--'))
                time.sleep(0.2)
            return

        print(cabecera)
        maximo, estancado, paso = 0.0, 0, 0

        while paso < 200 and estancado < 15:
            mover(0.03, PASO_S)
            paso += 1

            vis = blob_visible_ahora()
            tam_actual = estado['blob_tam'] if vis else 0.0
            d = distancia_real()

            print('{:4d}  {:7.1f}  {:7.0f}  {:4d}  {:>8s}'.format(
                paso, estado['blob_x'] if vis else -1.0, tam_actual, int(vis),
                '{:.0f}'.format(d) if d is not None else '--'))

            if vis and tam_actual >= 0.9 * BLOB_TAM_SATURACION:
                print('\n-> tamaño del blob ya muy cercano al límite de '
                      'saturación, deteniendo la calibración.')
                break

            if tam_actual > maximo:
                maximo, estancado = tam_actual, 0
            else:
                estancado += 1

        mover(-0.05, 1.0)

        print('\ntamaño máximo visto  {:6.0f}   <- candidato a '
              'BLOB_TAM_SATURACION'.format(maximo))
        print('\nTAM_OBJETIVO se elige mirando la tabla: el tamaño de la '
              'fila en la que\nel seguidor está a la distancia d deseada, '
              'dividido por el máximo.\nSi tenéis el módulo sim cargado, '
              'la columna distmm da la distancia real\nen milímetros entre '
              'los dos robots para esa misma fila, que es la forma\nmás '
              'directa de fijar d.')

    except KeyboardInterrupt:
        pass
    finally:
        try:
            futuro = cli_stop_wheels.call_async(StopWheels.Request())
            rclpy.spin_until_future_complete(nodo, futuro, timeout_sec=1.0)
        except Exception:
            pass
        nodo.destroy_node()
