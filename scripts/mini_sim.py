"""

Path tracking simulation with LQR speed and steering control

author Atsushi Sakai (@Atsushi_twi)

"""
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import cubic_spline_planner
import time
import rclpy
from rclpy.node import Node
from autoware_control_msgs.msg import Control
from autoware_planning_msgs.msg import Trajectory,TrajectoryPoint
from crp_msgs.msg import Ego


show_animation = True
max_steer = np.deg2rad(45.0)  # maximum steering angle[rad]
dt = 0.1  # time tick[s]

L = 0.5 # Wheel base of vehicle [m]


def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)

    Parameters
    ----------
    x : float or array_like
        A angle or an array of angles. This array is flattened for
        the calculation. When an angle is provided, a float angle is returned.
    zero_2_2pi : bool, optionaldef pi_2_pi(angle):
    return angle_mod(angle)
        Change angle modulo range to [0, 2pi)
        Default is False.
    degree : bool, optional
        If True, then the given angles are assumed to be in degrees.
        Default is False.

    Returns
    -------
    ret : float or ndarray
        an angle or an array of modulated angle.

    Examples
    --------
    >>> angle_mod(-4.0)
    2.28318531

    >>> angle_mod([-4.0])
    np.array(2.28318531)

    >>> angle_mod([-150.0, 190.0, 350], degree=True)
    array([-150., -170.,  -10.])

    >>> angle_mod(-60.0, zero_2_2pi=True, degree=True)
    array([300.])

    """
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle
    

def pi_2_pi(angle):
    return angle_mod(angle)

class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0,steer=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.steer = steer


class ROSMiniSim(Node):

    def __init__(self):

        super().__init__('ROS_Mini_Sim')
        
        self.cx = []
        self.cy = []
        self.cyaw = []
        self.ck = []
        self.s = []

        self.state = State(x=0.0, y=0.0, yaw=0.0, v=0.0, steer=0.0)

        self.generate_trajectory()

        self.ctrl_subscirber = self.create_subscription(Control, '/control/command/contol_cmd', self.update, 10)
        self.traj_publisher = self.create_publisher(Trajectory, '/plan/trajetory', 10)
        self.ego_publisher = self.create_publisher(Ego, '/ego', 10)

        self.trajectory_timer = self.create_timer(0.05, self.send_trajectory)
        self.ego_timer = self.create_timer(0.05, self.send_ego)


    def generate_trajectory(self,):

        ax = [0.0, 20.0, 40.0, 80.0, 100.0, 80.0, 60.0]
        ay = [0.0, 0.0, 20.0, 20.0, 40.0, 60.0, 60.0]
        goal = [ax[-1], ay[-1]]


        self.cx, self.cy, self.cyaw, self.ck, self.s = cubic_spline_planner.calc_spline_course(
            ax, ay, ds=0.1)
        target_speed = 10.0 / 3.6  # simulation parameter km/h -> m/s

        self.sp = self.calc_speed_profile(self.cyaw, target_speed)

    def calc_speed_profile(self, cyaw, target_speed):
        speed_profile = [target_speed] * len(cyaw)

        direction = 1.0

        # Set stop point
        for i in range(len(cyaw) - 1):
            dyaw = abs(cyaw[i + 1] - cyaw[i])
            switch = math.pi / 4.0 <= dyaw < math.pi / 2.0

            if switch:
                direction *= -1

            if direction != 1.0:
                speed_profile[i] = - target_speed
            else:
                speed_profile[i] = target_speed

            if switch:
                speed_profile[i] = 0.0

        # speed down
        for i in range(40):
            speed_profile[-i] = target_speed / (50 - i)
            if speed_profile[-i] <= 1.0 / 3.6:
                speed_profile[-i] = 1.0 / 3.6

        return speed_profile

    def send_trajectory(self):

        
        traj = Trajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.header.frame_id = 'map'


        for i in range(len(self.cx)):
            point = TrajectoryPoint()
            point.pose.position.x = self.cx[i]
            point.pose.position.y = self.cy[i]
            point.pose.orientation.z = self.cyaw[i]
            point.longitudinal_velocity_mps = self.sp[i]

            traj.points.append(point)

        self.traj_publisher.publish(traj)

    def send_ego(self):

        ego = Ego()
        ego.header.stamp = self.get_clock().now().to_msg()
        ego.header.frame_id = 'base_link'

        ego.pose.pose.position.x = self.state.x
        ego.pose.pose.position.y = self.state.y
        ego.pose.pose.orientation.z = self.state.yaw  # quick get around for now, TODO to quaternion
        ego.twist.twist.linear.x = self.state.v

        self.ego_publisher.publish(ego)


    def update(state, msg):

        a = msg.longitudinal.acceleration

        delta = msg.lateral.steering_tire_angle

        if delta >= max_steer:
            delta = max_steer
        if delta <= - max_steer:
            delta = - max_steer

        state.x = state.x + state.v * math.cos(state.yaw) * dt
        state.y = state.y + state.v * math.sin(state.yaw) * dt
        state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
        state.v = state.v + a * dt
        state.steer = delta

        return state

def main(args=None):
    rclpy.init(args=args)
    node = ROSMiniSim()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # Clean up and shutdown
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
