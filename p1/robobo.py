import sys
import time
import argparse
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from rclpy.utilities import remove_ros_args
    from rcl_interfaces.msg import ParameterDescriptor

    # Import required service and action types
    from robobo_ros2_interfaces.srv import StopWheels, MoveWheels, MoveWheelsTime as MoveWheelsTimeSrv
    from robobo_ros2_interfaces.msg import BlobArray
    from robobo_ros2_interfaces.action import MoveTilt, MoveWheelsTime as MoveWheelsTimeAction
    # Aggregate IR (infrared proximity) sensor topic:
    #   /robobo/robot_<n>/base/ir  ->  std_msgs/msg/Int32MultiArray
    from std_msgs.msg import Int32MultiArray, Int32

except ImportError as e:
    sys.stderr.write(
        f"[ERROR] Failed to import ROS 2 dependencies: {e}\n"
        "Please ensure your ROS 2 environment and workspace are sourced:\n"
        "  Windows:      .\\install\\setup.ps1\n"
        "  Linux/macOS:  source install/setup.bash\n"
    )
    sys.exit(1)

def parse_cli_args(args=None):
    """Parse application-level command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Standalone demonstration script for Robobo robot capabilities.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--robot-name', '-r', '--robot_name',
        type=str,
        default='0',
        dest='robot_name',
        help='Robot name for ROS namespace (/robobo/robot_<name>/base)'
    )
    parser.add_argument(
        '--timeout', '-t',
        type=float,
        default=15.0,
        help='Maximum seconds to wait for robobo_container to be ready'
    )
    parser.add_argument(
        '--ip',
        type=str,
        default=None,
        help='(Informational) Robot IP address. Note: IP is configured when launching robobo_container'
    )
    parser.add_argument(
        '--robot-id', '--robot_id',
        type=int,
        default=None,
        dest='robot_id',
        help='(Informational) Robot ID. Note: Robot ID is configured when launching robobo_container'
    )
    return parser.parse_known_args(args)


class RoboboDemo(Node):
    def __init__(self, robot_name='0', timeout=15.0):
        super().__init__('robobo_demo')

        # Declare parameters for connection configuration
        self.declare_parameter('robot_name', robot_name, ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('timeout', timeout)
        self.declare_parameter('ip', '127.0.0.1')
        self.declare_parameter('robot_id', 0)

        # Resolve parameter values (ROS params override CLI defaults if passed via --ros-args)
        self.robot_name = str(self.get_parameter('robot_name').value)
        self.timeout = float(self.get_parameter('timeout').value)
        self.ip = str(self.get_parameter('ip').value)
        self.robot_id = int(self.get_parameter('robot_id').value)

        self.base_ns = f'/robobo/robot_{self.robot_name}/base'
        self.smartphone_ns = f'/robobo/robot_{self.robot_name}/smartphone'

        self.get_logger().info("==========================================")
        self.get_logger().info("       Robobo ROS 2 Standalone Demo       ")
        self.get_logger().info("==========================================")
        self.get_logger().info(f"Target Robot Name : {self.robot_name}")
        self.get_logger().info(f"Target Base NS    : {self.base_ns}")
        self.get_logger().info("==========================================")

        self.stop_wheels_client = self.create_client(
            StopWheels, f'{self.base_ns}/stop_wheels'
        )
        self.move_wheels_client = self.create_client(
            MoveWheels, f'{self.base_ns}/move_wheels'
        )

        self.move_tilt_action_client = ActionClient(
            self, MoveTilt, f'{self.base_ns}/move_tilt'
        )

        # Subscribe to the aggregate IR (infrared) proximity sensor topic
        self.latest_irs = None
        self.irs_sub = self.create_subscription(
            Int32MultiArray, f'{self.base_ns}/ir', self._irs_callback, 10
        )

        self.latest_color_blob = None
        self.color_blob_sub = self.create_subscription(
            BlobArray, f'{self.smartphone_ns}/color_blobs', self._color_blob_callback, 10
        )

        self.move_wheels_time_srv_client = self.create_client(
            MoveWheelsTimeSrv, f'{self.base_ns}/move_wheels_time'
        )


        self.move_wheels_time_action_client = ActionClient(
            self, MoveWheelsTimeAction, f'{self.base_ns}/move_wheels_time'
        )

        self.latest_right_wheel_speed = None
        self.latest_left_wheel_speed = None

        self.right_wheel_speed = self.create_subscription(
            Int32, f'{self.base_ns}/wheel/right/speed', self._right_wheel_speed_callback, 10
        )
        self.left_wheel_speed = self.create_subscription(
            Int32, f'{self.base_ns}/wheel/left/speed', self._left_wheel_speed_callback, 10
        )

    def _irs_callback(self, msg):
        """Store the most recent IR sensor reading (list of 8 ints)."""
        self.latest_irs = list(msg.data)

    def _color_blob_callback(self, msg):
        """Store the most recent color blob detection reading."""
        self.latest_color_blob = msg

    def _right_wheel_speed_callback(self, msg):
        """Store the most recent right wheel speed reading (int)."""
        self.latest_right_wheel_speed = msg.data
    
    def _left_wheel_speed_callback(self, msg):
        """Store the most recent left wheel speed reading (int)."""
        self.latest_left_wheel_speed = msg.data

    def wait_for_ready(self, timeout_sec=None):
        """Wait for required services and action servers to become available on the base node."""
        if timeout_sec is None:
            timeout_sec = self.timeout

        self.get_logger().info(
            f"Checking for running 'robobo_container' (timeout: {timeout_sec:.1f}s)..."
        )
        start_time = time.time()
        last_log_time = 0.0

        while time.time() - start_time < timeout_sec:
            wheels_time_ready = self.move_wheels_time_action_client.wait_for_server(
                timeout_sec=0.5
            )
            wheels_ready = self.move_wheels_client.wait_for_service(timeout_sec=0.5)
            tilt_ready = self.move_tilt_action_client.wait_for_server(timeout_sec=0.5)
            
            if wheels_ready and tilt_ready and wheels_time_ready:
                self.get_logger().info("-> robobo_container detected! All required services and actions are ready.")
                return True

            elapsed = time.time() - start_time
            if elapsed - last_log_time >= 3.0:
                self.get_logger().info(
                    f"Waiting for robobo_container... ({elapsed:.0f}/{timeout_sec:.0f}s)"
                )
                last_log_time = elapsed

        # Final check of individual interfaces for detailed diagnostic
        missing = []
        if not self.set_led_client.service_is_ready():
            missing.append(f"Service : {self.base_ns}/set_led")
        if not self.move_wheels_time_action_client.server_is_ready():
            missing.append(f"Action  : {self.base_ns}/move_wheels_time")
        if not self.move_pan_action_client.server_is_ready():
            missing.append(f"Action  : {self.base_ns}/move_pan")
        if not self.move_tilt_action_client.server_is_ready():
            missing.append(f"Action  : {self.base_ns}/move_tilt")

        self.get_logger().error(
            f"\n"
            f"****************************************************************\n"
            f"[ERROR] Could not connect to 'robobo_container' after {timeout_sec:.1f}s.\n"
            f"Target namespace: '{self.base_ns}'\n"
            f"Missing interface(s):\n  " + "\n  ".join(missing) + "\n\n"
            f"Please ensure 'robobo_container' is running in another terminal:\n"
            f"  ros2 launch robobo_ros2 robobo.launch.py robot_name:={self.robot_name}\n"
            f"  # or:\n"
            f"  ros2 run robobo_ros2 robobo_container --ros-args -p robot_name:={self.robot_name}\n"
            f"****************************************************************"
        )
        return False

    def move_tilt(self, angle, speed=20.0):
        """Send tilt motor action goal synchronously."""
        self.get_logger().info(f"Moving tilt motor to {angle}° (speed: {speed})...")
        goal_msg = MoveTilt.Goal()
        goal_msg.angle = float(angle)
        goal_msg.speed = float(speed)

        send_goal_future = self.move_tilt_action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=5.0)

        if not send_goal_future.done():
            self.get_logger().error("  -> Timed out sending tilt goal")
            return False

        goal_handle = send_goal_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("  -> Tilt movement goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=10.0)

        if not result_future.done():
            self.get_logger().error("  -> Timed out waiting for tilt result")
            return False

        result = result_future.result()
        success = result.result.success if result and result.result else False
        self.get_logger().info(f"  -> Tilt movement completed (success: {success})")
        return success

    def read_right_wheel_speed(self, timeout_sec=2.0):
        """Return the most recent right wheel speed reading (int)."""
        self.get_logger().info(f"Reading right wheel speed (topic: {self.base_ns}/wheel/right/speed)...")
        self.latest_right_wheel_speed = None
        start_time = time.time()

        while self.latest_right_wheel_speed is None and (time.time() - start_time) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=timeout_sec)

        if self.latest_right_wheel_speed is None:
            self.get_logger().warning(
                f"  -> No right wheel speed data received within {timeout_sec:.1f}s"
            )
            return None

        self.get_logger().info(f"  -> Wheel speed values (raw): {self.latest_right_wheel_speed}")
        return self.latest_right_wheel_speed

    def read_left_wheel_speed(self, timeout_sec=2.0):
        """Return the most recent left wheel speed reading (int)."""
        self.get_logger().info(f"Reading left wheel speed (topic: {self.base_ns}/wheel/left/speed)...")
        self.latest_left_wheel_speed = None
        start_time = time.time()

        while self.latest_left_wheel_speed is None and (time.time() - start_time) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=timeout_sec)

        if self.latest_left_wheel_speed is None:
            self.get_logger().warning(
                f"  -> No left wheel speed data received within {timeout_sec:.1f}s"
            )
            return None

        self.get_logger().info(f"  -> Wheel speed values (raw): {self.latest_left_wheel_speed}")
        return self.latest_left_wheel_speed

    def move_wheels_time(self, right_speed, left_speed, duration):
        """Send wheel movement action goal synchronously."""
        self.get_logger().info(
            f"Moving wheels: right={right_speed}, left={left_speed} for {duration}s..."
        )
        goal_msg = MoveWheelsTimeAction.Goal()
        goal_msg.right_speed = float(right_speed)
        goal_msg.left_speed = float(left_speed)
        goal_msg.time = float(duration)

        send_goal_future = self.move_wheels_time_action_client.send_goal_async(
            goal_msg
        )
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=5.0)

        if not send_goal_future.done():
            self.get_logger().error("  -> Timed out sending wheel movement goal")
            return False

        goal_handle = send_goal_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("  -> Wheel movement goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=duration + 10.0
        )

        if not result_future.done():
            self.get_logger().error("  -> Timed out waiting for wheel movement result")
            return False

        result = result_future.result()
        success = result.result.success if result and result.result else False
        self.get_logger().info(f"  -> Wheel movement completed (success: {success})")
        return success

    def read_color_blob(self, color='green', timeout_sec=2.0):
        """Wait for and log a fresh reading from the color blob topic.

        Returns a ColorBlob message with position and size information.
        """
        self.get_logger().info(f"Reading color blob for '{color}' (topic: {self.smartphone_ns}/color_blobs)...")
        self.latest_color_blob = None
        start_time = time.time()

        while self.latest_color_blob is None and (time.time() - start_time) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.latest_color_blob is None:
            self.get_logger().warning(
                f"  -> No color blob data received within {timeout_sec:.1f}s"
            )
            return None
        for blob in self.latest_color_blob.blobs:
            if blob.color == color:
                self.latest_color_blob = blob
                break
        self.get_logger().info(f"  -> Color blob values: x={self.latest_color_blob.x}, y={self.latest_color_blob.y}, size={self.latest_color_blob.size}")
        return self.latest_color_blob


    def read_irs(self, timeout_sec=2.0):
        """Wait for and log a fresh reading from the IR sensor topic.

        Returns a list of 8 raw ints from std_msgs/Int32MultiArray on
        '{base_ns}/ir'. The exact ordering of the 8 sensors is not
        guaranteed here - subscribe to the individual
        '{base_ns}/ir/<sensor>' topics (frontc, frontl, frontll,
        frontr, frontrr, backc, backl, backr) if you need to identify
        a specific one.
        """
        self.get_logger().info(f"Reading IR sensors (topic: {self.base_ns}/ir)...")
        self.latest_irs = None
        start_time = time.time()

        while self.latest_irs is None and (time.time() - start_time) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.latest_irs is None:
            self.get_logger().warning(
                f"  -> No IR data received within {timeout_sec:.1f}s"
            )
            return None

        self.get_logger().info(f"  -> IR values (raw): {self.latest_irs}")
        return self.latest_irs

    def move_wheels(self, right_speed, left_speed):
        """Send wheel movement action goal synchronously."""
        req = MoveWheels.Request()
        req.right_speed = float(right_speed)
        req.left_speed = float(left_speed)
        self.get_logger().info(f"Setting wheels speed to R: {right_speed},  L: {left_speed}...")

        future = self.move_wheels_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.done():
            try:
                response = future.result()
                if response and response.success:
                    self.get_logger().info(f"  -> Wheels moved successfully: {response}")
                    return True
                else:
                    msg = response.message if response else "Empty response"
                    self.get_logger().error(f"  -> Failed to move wheels: {msg}")
            except Exception as e:
                self.get_logger().error(f"  -> Error reading move_wheels response: {e}")
        else:
            self.get_logger().error("  -> Call to move_wheels service timed out")
        return False

    def stop_robot(self):
        """Safely stop wheels and reset LEDs (useful on abort / shutdown)."""
        try:
            if self.stop_wheels_client.wait_for_service(timeout_sec=0.5):
                req = StopWheels.Request()
                future = self.stop_wheels_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        except Exception:
            pass

        try:
            if self.set_led_client.wait_for_service(timeout_sec=0.5):
                req = SetLed.Request()
                req.led = 'All'
                req.color = 'OFF'
                future = self.set_led_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        except Exception:
            pass