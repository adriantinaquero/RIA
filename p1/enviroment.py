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
import sys 
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node

# Interfaces propias del puente para poder reiniciar la escena
from robobo_ros2_interfaces.msg import RobotLocation    
from robobo_ros2_interfaces.srv import ResetSimulation


try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from rclpy.utilities import remove_ros_args
    from rcl_interfaces.msg import ParameterDescriptor
    from geometry_msgs.msg import Twist

    # Import required service and action types
    from robobo_ros2_interfaces.srv import StopWheels, MoveWheels, MoveWheelsTime as MoveWheelsTimeSrv, MoveWheelsDegrees
    from robobo_ros2_interfaces.msg import BlobArray
    from robobo_ros2_interfaces.action import MoveTilt, MoveWheelsTime as MoveWheelsTimeAction

    from std_msgs.msg import Int32
    HAY_INTERFACES_ROBOT = True
except ImportError as e:
    sys.stderr.write(
        f"[ERROR] Failed to import ROS 2 dependencies: {e}\n"
        "Please ensure your ROS 2 environment and workspace are sourced:\n"
        "  Windows:      .\\install\\setup.ps1\n"
        "  Linux/macOS:  source install/setup.bash\n"
    )
    sys.exit(1)
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
                 pasos_max=600,
                 usar_simulador=True,
                 verbose=False,
                 robot_name='0'):
        super().__init__()

        self.pasos_max = pasos_max
        self.verbose = verbose
        self.robot_name = robot_name

        # -------------------------------------------------- ROS 2
        if not rclpy.ok():
            rclpy.init()

        self.nodo = Node('seguir_entorno_rl')

        if not HAY_INTERFACES_ROBOT:
            raise RuntimeError(AVISO_SIN_ROBOT)

        self._blob_posx = None
        self._dist = None
        self._last_blob_x = None
        self._last_seen_blob_x = 0.0
        self._blob_t_ultimo = -1.0   # time.time() de la última detección

        self.base_ns = f'/robobo/robot_{self.robot_name}/base'
        self.smartphone_ns = f'/robobo/robot_{self.robot_name}/smartphone'

        # self.nodo.get_logger().info("==========================================")
        # self.nodo.get_logger().info("       Robobo ROS 2 Standalone Demo       ")
        # self.nodo.get_logger().info("==========================================")
        # self.nodo.get_logger().info(f"Target Robot Name : {self.robot_name}")
        # self.nodo.get_logger().info(f"Target Base NS    : {self.base_ns}")
        # self.nodo.get_logger().info("==========================================")

        self.stop_wheels_client = self.nodo.create_client(
            StopWheels, f'{self.base_ns}/stop_wheels'
        )
        self.move_wheels_client = self.nodo.create_client(
            MoveWheels, f'{self.base_ns}/move_wheels'
        )


        self.move_wheels_degrees_client = self.nodo.create_client(
            MoveWheelsDegrees, f'{self.base_ns}/move_wheels_degrees'
        )

        self.latest_color_blob = None
        self.color_blob_sub = self.nodo.create_subscription(
            BlobArray, f'{self.smartphone_ns}/color_blobs', self._cb_blob, 10
        )

        self.move_wheels_time_srv_client = self.nodo.create_client(
            MoveWheelsTimeSrv, f'{self.base_ns}/move_wheels_time'
        )


        self.move_wheels_time_action_client = ActionClient(
            self.nodo, MoveWheelsTimeAction, f'{self.base_ns}/move_wheels_time'
        )

        self.move_tilt_action_client = ActionClient(
            self.nodo, MoveTilt, f'{self.base_ns}/move_tilt'
        )

        self.latest_right_wheel_speed = None
        self.latest_left_wheel_speed = None

        self.right_wheel_speed = self.nodo.create_subscription(
            Int32, f'{self.base_ns}/wheel/right/speed', self._right_wheel_speed_callback, 10
        )
        self.left_wheel_speed = self.nodo.create_subscription(
            Int32, f'{self.base_ns}/wheel/left/speed', self._left_wheel_speed_callback, 10
        )

        self.pub_vel = self.nodo.create_publisher(Twist, NS_BASE + '/cmd_vel', 10)

        # -------------------------------------------------- simulador
        self.cli_reset = None
        self._pose = None
        self._pose_robot_objetivo = None
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
        # [blob_x, dist, visible, ultimo_blob_x, ultima_accion_v, ultima_accion_w]
        low=np.array([-1.0, -1.0, 0.0, -1, -1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0, 1, 1.0, 1.0], dtype=np.float32),
        dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # -------------------------------------------------- estado
        self.pasos = 0
        self.en_banda = 0
        self.pasos_perdido = 0
        self._n_blob_msgs = 0
        self._n_ir = 0
        self.ultima_accion = np.zeros((2,), dtype=np.float32)


    # -----------------------------------------------------------------
    # Comunicación con ROS 2
    # -----------------------------------------------------------------
    def _right_wheel_speed_callback(self, msg):
        """Store the most recent right wheel speed reading (int)."""
        self.latest_right_wheel_speed = msg.data
    
    def _left_wheel_speed_callback(self, msg):
        """Store the most recent left wheel speed reading (int)."""
        self.latest_left_wheel_speed = msg.data

    def _cb_blob(self, msg):
        """Se ejecuta cada vez que llega una detección del blob verde.

        Este topic no es periódico: sólo publica
        cuando hay blob detectado. Por eso no se cuenta "lecturas nuevas"
        para esperar aquí; se guarda el momento de la detección y
        _blob_visible() decide, por antigüedad, si todavía es de fiar.
        """
        for blob in msg.blobs:
            self._n_blob_msgs += 1
            if blob.color == COLOR_BLOB:
                self._last_blob_x = self._blob_posx 
                self._blob_posx = float(blob.x)
                self._dist = (-238.76 * np.log(blob.size) + 1557.47)/1000
                self._blob_t_ultimo = time.time()
                self._n_ir += 1
                break

    def _cb_pose(self, msg):
        self._pose = (msg.position.x, msg.position.z, msg.rotation.y)

    def _cb_pose_robot_objetivo(self, msg):
        self._pose_robot_objetivo = (msg.position.x, msg.position.z, msg.rotation.y)

    def move_tilt(self, angle, speed=20.0):
        """Send tilt motor action goal synchronously."""
        # self.nodo.get_logger().info(f"Moving tilt motor to {angle}° (speed: {speed})...")
        goal_msg = MoveTilt.Goal()
        goal_msg.angle = float(angle)
        goal_msg.speed = float(speed)

        send_goal_future = self.move_tilt_action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.nodo, send_goal_future, timeout_sec=5.0)

        if not send_goal_future.done():
            self.nodo.get_logger().error("  -> Timed out sending tilt goal")
            return False

        goal_handle = send_goal_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.nodo.get_logger().error("  -> Tilt movement goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.nodo, result_future, timeout_sec=10.0)

        if not result_future.done():
            self.nodo.get_logger().error("  -> Timed out waiting for tilt result")
            return False

        result = result_future.result()
        success = result.result.success if result and result.result else False
        # self.nodo.get_logger().info(f"  -> Tilt movement completed (success: {success})")
        return success

    def read_right_wheel_speed(self, timeout_sec=2.0):
        """Return the most recent right wheel speed reading (int)."""
        # self.nodo.get_logger().info(f"Reading right wheel speed (topic: {self.base_ns}/wheel/right/speed)...")
        self.latest_right_wheel_speed = None
        start_time = time.time()

        while self.latest_right_wheel_speed is None and (time.time() - start_time) < timeout_sec:
            rclpy.spin_once(self.nodo, timeout_sec=timeout_sec)

        if self.latest_right_wheel_speed is None:
            self.nodo.get_logger().warning(
                f"  -> No right wheel speed data received within {timeout_sec:.1f}s"
            )
            return None

        # self.nodo.get_logger().info(f"  -> Wheel speed values (raw): {self.latest_right_wheel_speed}")
        return self.latest_right_wheel_speed

    def read_left_wheel_speed(self, timeout_sec=2.0):
        """Return the most recent left wheel speed reading (int)."""
        # self.nodo.get_logger().info(f"Reading left wheel speed (topic: {self.base_ns}/wheel/left/speed)...")
        self.latest_left_wheel_speed = None
        start_time = time.time()

        while self.latest_left_wheel_speed is None and (time.time() - start_time) < timeout_sec:
            rclpy.spin_once(self.nodo, timeout_sec=timeout_sec)

        if self.latest_left_wheel_speed is None:
            self.nodo.get_logger().warning(
                f"  -> No left wheel speed data received within {timeout_sec:.1f}s"
            )
            return None

        # self.nodo.get_logger().info(f"  -> Wheel speed values (raw): {self.latest_left_wheel_speed}")
        return self.latest_left_wheel_speed

    def move_wheels_time(self, right_speed, left_speed, duration):
        """Send wheel movement action goal synchronously."""
        # self.nodo.get_logger().info(
        #     f"Moving wheels: right={right_speed}, left={left_speed} for {duration}s..."
        # )
        goal_msg = MoveWheelsTimeAction.Goal()
        goal_msg.right_speed = float(right_speed)
        goal_msg.left_speed = float(left_speed)
        goal_msg.time = float(duration)

        send_goal_future = self.move_wheels_time_action_client.send_goal_async(
            goal_msg
        )
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=5.0)

        if not send_goal_future.done():
            self.nodo.get_logger().error("  -> Timed out sending wheel movement goal")
            return False

        goal_handle = send_goal_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.nodo.get_logger().error("  -> Wheel movement goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=duration + 10.0
        )

        if not result_future.done():
            self.nodo.get_logger().error("  -> Timed out waiting for wheel movement result")
            return False

        result = result_future.result()
        success = result.result.success if result and result.result else False
        # self.nodo.get_logger().info(f"  -> Wheel movement completed (success: {success})")
        return success

    def move_left_wheel_degrees(self, degrees, speed):
        """Send left wheel movement action goal synchronously."""
        # self.nodo.get_logger().info(
        #     f"Moving left wheel: {degrees} degrees at speed {speed}..."
        # )
        req = MoveWheelsDegrees.Request()
        req.wheel = 'L'
        req.left_degrees = float(degrees)
        req.left_speed = float(speed)

        # self.nodo.get_logger().info(f"Setting left wheel degrees to: {req.left_degrees}, speed: {req.left_speed}...")

        future = self.move_wheels_degrees_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.done():
            try:
                response = future.result()
                if response and response.success:
                    # self.nodo.get_logger().info(f"  -> Wheels moved successfully: {response}")
                    return True
                else:
                    msg = response.message if response else "Empty response"
                    self.nodo.get_logger().error(f"  -> Failed to move wheels: {msg}")
            except Exception as e:
                self.nodo.get_logger().error(f"  -> Error reading move_wheels response: {e}")
        else:
            self.nodo.get_logger().error("  -> Call to move_wheels service timed out")
        return False

    def move_right_wheel_degrees(self, degrees, speed):
        """Send right wheel movement action goal synchronously."""
        # self.nodo.get_logger().info(
        #     f"Moving right wheel: {degrees} degrees at speed {speed}..."
        # )
        req = MoveWheelsDegrees.Request()
        req.wheel = 'R'
        req.right_degrees = float(degrees)
        req.right_speed = float(speed)

        future = self.move_wheels_degrees_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        self.nodo.get_logger().info(f"Setting right wheel degrees to: {req.right_degrees}, speed: {req.right_speed}...")

        if future.done():
            try:
                response = future.result()
                if response and response.success:
                    # self.nodo.get_logger().info(f"  -> Wheels moved successfully: {response}")
                    return True
                else:
                    msg = response.message if response else "Empty response"
                    self.nodo.get_logger().error(f"  -> Failed to move wheels: {msg}")
            except Exception as e:
                self.nodo.get_logger().error(f"  -> Error reading move_wheels response: {e}")
        else:
            self.nodo.get_logger().error("  -> Call to move_wheels service timed out")
        return False

    def move_wheels(self, right_speed, left_speed):
        """Send wheel movement action goal synchronously."""
        req = MoveWheels.Request()
        req.right_speed = float(right_speed)
        req.left_speed = float(left_speed)
        # self.nodo.get_logger().info(f"Setting wheels speed to R: {right_speed},  L: {left_speed}...")

        future = self.move_wheels_client.call_async(req)
        rclpy.spin_until_future_complete(self.nodo, future, timeout_sec=5.0)

        if future.done():
            try:
                response = future.result()
                if response and response.success:
                    # self.nodo.get_logger().info(f"  -> Wheels moved successfully: {response}")
                    return True
                else:
                    msg = response.message if response else "Empty response"
                    self.nodo.get_logger().error(f"  -> Failed to move wheels: {msg}")
            except Exception as e:
                self.nodo.get_logger().error(f"  -> Error reading move_wheels response: {e}")
        else:
            self.nodo.get_logger().error("  -> Call to move_wheels service timed out")
        return False

    def _publicar_velocidad(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub_vel.publish(msg)

    def stop_robot(self):
        """Safely stop wheels and reset LEDs (useful on abort / shutdown)."""
        try:
            if self.stop_wheels_client.wait_for_service(timeout_sec=0.5):
                req = StopWheels.Request()
                future = self.stop_wheels_client.call_async(req)
                rclpy.spin_until_future_complete(self.nodo, future, timeout_sec=1.0)
        except Exception:
            pass

    def _reiniciar_escena(self):
        self.nodo.get_logger().info("Reiniciando escena...")
        futuro = self.cli_reset.call_async(ResetSimulation.Request())
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=5.0)
        if futuro.result() is None or not futuro.result().success:
            raise RuntimeError('El reinicio de la escena ha fallado.')
    
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
        objetivo = self._n_blob_msgs + 1
        t0 = time.time()
        while self._n_blob_msgs < objetivo:
            rclpy.spin_once(self.nodo, timeout_sec=0.05)
            if time.time() - t0 > ESPERA_MAX_S:
                raise RuntimeError(
                    'No llegan lecturas de {}/color_blobs.\n'
                    'Comprobar que robobo_container está en ejecución y '
                    'conectado al simulador.'.format(NS_BASE)
                    if inicial else
                    'Se ha interrumpido el flujo de lecturas de los '
                    'blobs durante el episodio.')
    # -----------------------------------------------------------------
    # Observación
    # -----------------------------------------------------------------

    def _blob_visible(self):
        if self._dist == float('inf'):
            return False
        return (time.time() - self._blob_t_ultimo) <= BLOB_VIGENCIA_S

    def _leer_blob(self):
        """Devuelve (visible, blob_x, blob_tam) ya normalizados."""
        visible = self._blob_visible()
        if visible:
            blob_x = float(np.clip(
                (self._blob_posx - BLOB_X_CENTRO) / BLOB_X_CENTRO, -1.0, 1.0))
            dist = float(np.clip(
                (self._dist - DIST_TARGET_M) / DIST_TARGET_M, -1.0, 1.0))
        else:
            blob_x, dist = 0.0, 0.0
        return visible, blob_x, dist


    def _observacion(self):
        self._last_seen_blob_x = self._last_blob_x if self._blob_visible() else self._last_seen_blob_x
        visible, blob_x, dist = self._leer_blob()
        return np.array(
            [blob_x, dist, 1.0 if visible else 0.0, self._last_seen_blob_x] +
            list(self.ultima_accion),
            dtype=np.float32)

    # -----------------------------------------------------------------
    # Recompensa
    # -----------------------------------------------------------------

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
            r_posicion += 0.75
        
        
        r_posicion -= abs(accion_prev[1]-accion[1])/10

        if BANDA_OBJETIVO_M[0] <= dist <= BANDA_OBJETIVO_M[1]:
            r_posicion -= abs(accion_prev[0]-accion[0])**2
        
        return float(r_posicion)

    # -----------------------------------------------------------------
    # Interfaz de Gymnasium
    # -----------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.stop_robot()

        if self.cli_reset is not None:
            # Reinicia la escena completa
            self._reiniciar_escena()
            time.sleep(0.5)

        self.pasos = 0
        self.en_banda = 0
        self.pasos_perdido = 0
        self._phi_prev = None
        self.ultima_accion = np.zeros((2,), dtype=np.float32)

        # Se limpia la última detección para no arrastrar una lectura de antes del
        # reinicio, y se confía en _blob_visible() para el resto.
        self._blob_t_ultimo = -1.0
        rclpy.spin_once(self.nodo, timeout_sec=0.1)
        self.move_tilt(110, speed=20.0)  # mirar hacia abajo para ver el blob
        return self._observacion(), {}

    def step(self, accion):
        accion_prev = self.ultima_accion
        accion = np.clip(np.asarray(accion, dtype=np.float32), -1.0, 1.0)
        self.ultima_accion = accion
            
        # 1. Reescalar la acción normalizada a unidades del SI y enviarla.
        #    El puente traduce el Twist a velocidades de rueda.
        v = V_MIN + (accion[0] + 1.0) * 0.5 * (V_MAX - V_MIN)
        w = float(accion[1]) * W_MAX
        self._publicar_velocidad(v, w)

        # Como el topic de blobs es por eventos hace falta seguir haciendo
        # spin durante toda la ventana del paso, o se puede perder la
        # única detección que llegue en ese intervalo.
        time.sleep(PASO_S)

        self._vaciar_cola()
        self._esperar_lectura_nueva()

        self.pasos += 1

        visible, blob_x, dist = self._leer_blob()

        if visible:
            self.pasos_perdido = 0
        else:
            self.pasos_perdido += 1

        d_m = self._dist
        choque = visible and d_m <= DIST_CHOQUE_M
        en_banda = visible and BANDA_OBJETIVO_M[0] <= d_m <= BANDA_OBJETIVO_M[1]
        
        self.en_banda = self.en_banda + 1 if en_banda else 0
        perdido = self.pasos_perdido >= PASOS_PERDIDO_MAX

        recompensa = self._recompensa(
            blob_x, dist, visible, accion, accion_prev,
            perdido, choque)

        terminated = bool(choque or perdido)
        truncated = bool(self.pasos >= self.pasos_max)

        if terminated or truncated:
            self.stop_robot()

        distancia_real = None
        if self._pose is not None and self._pose_robot_objetivo is not None:
            distancia_real = float(np.hypot(
                self._pose[0] - self._pose_robot_objetivo[0],
                self._pose[1] - self._pose_robot_objetivo[1]))

        info = {
            'blob_x': blob_x,
            'dist': dist,
            'visible': visible,
            'choque': choque,
            'perdido': perdido,
            'en_banda_consecutivos': self.en_banda,
            'pose_sim': self._pose,
            'distancia_real_mm': d_m,
        }
        if self.verbose:
            print('paso {:3d}  vis={}  x={:+.3f}  dist={:.3f}  '
                  'v={:+.3f}  w={:+.3f}  r={:+.3f}{}'
                  .format(self.pasos, int(visible), blob_x, d_m,
                          v, w, recompensa,
                          '  CHOQUE' if choque else
                          ('  PERDIDO' if perdido else '')))

        return self._observacion(), recompensa, terminated, truncated, info

    def close(self):
        try:
            self.stop_robot()
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
    """Mide blob_x y blob_tam a distintas distancias del robot líder."""
    if not HAY_INTERFACES_ROBOT:
        raise RuntimeError(AVISO_SIN_ROBOT)

    if not rclpy.ok():
        rclpy.init()
    nodo = Node('calibracion_seguir')

    estado = {
        'blob_x': None,
        'blob_tam': None,
        'blob_t': -1.0,
        'pose': None,
        'pose_robot_objetivo': None
    }

    def cb_blob(m):
        for blob in m.blobs:
            if blob.color == COLOR_BLOB:
                estado['blob_x'] = float(blob.x)
                estado['blob_tam'] = float(blob.size)
                estado['blob_t'] = time.time()
                break

    nodo.create_subscription(BlobArray, NS_SMARTPHONE + '/color_blobs', cb_blob, 10)

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

    def spin_for_duration(duration_s):
        """Procesa callbacks de ROS activamente durante una ventana de tiempo."""
        t0 = time.time()
        while time.time() - t0 < duration_s:
            rclpy.spin_once(nodo, timeout_sec=0.02)

    def mover(v, segundos):
        peticion = MoveWheels.Request()
        peticion.right_speed = float(v)
        peticion.left_speed = float(v)
        futuro = cli_move_wheels.call_async(peticion)
        rclpy.spin_until_future_complete(nodo, futuro, timeout_sec=1.0)

        # Procesa ROS mientras se mueve
        spin_for_duration(segundos)

        futuro = cli_stop_wheels.call_async(StopWheels.Request())
        rclpy.spin_until_future_complete(nodo, futuro, timeout_sec=1.0)

    def distancia_real():
        print(estado['pose'], estado['pose_robot_objetivo'])
        if estado['pose'] and estado['pose_robot_objetivo']:
            return ((estado['pose'][0] - estado['pose_robot_objetivo'][0]) ** 2 +
                    (estado['pose'][1] - estado['pose_robot_objetivo'][1]) ** 2) ** 0.5
        return None

    cabecera = '{:>4s}  {:>7s}  {:>7s}  {:>4s}  {:>8s}'.format(
        'paso', 'blob_x', 'tam', 'vis', 'distmm')

    try:
        # Esperar 0.5s girando callbacks para recibir la detección inicial de la cámara
        spin_for_duration(0.5)

        if not avanzar:
            print(cabecera)
            while rclpy.ok():
                spin_for_duration(0.2)
                vis = blob_visible_ahora()
                d = distancia_real()
                print('{:4s}  {:7s}  {:7s}  {:4d}  {:>8s}'.format(
                    '--',
                    '{:.1f}'.format(estado['blob_x']) if vis else '--',
                    '{:.0f}'.format(estado['blob_tam']) if vis else '--',
                    int(vis),
                    '{:.0f}'.format(d) if d is not None else '--'))
            return

        print(cabecera)
        maximo, estancado, paso = 0.0, 0, 0

        while paso < 200 and estancado < 15:
            mover(15, PASO_S)
            
            # Un pequeño spin extra para dar tiempo a capturar el frame justo tras parar
            spin_for_duration(0.1)

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

        mover(-20, 1.0)

        print('\ntamaño máximo visto  {:6.0f}   <- candidato a '
              'BLOB_TAM_SATURACION'.format(maximo))

    except KeyboardInterrupt:
        pass
    finally:
        try:
            futuro = cli_stop_wheels.call_async(StopWheels.Request())
            rclpy.spin_until_future_complete(nodo, futuro, timeout_sec=1.0)
        except Exception:
            pass
        nodo.destroy_node()