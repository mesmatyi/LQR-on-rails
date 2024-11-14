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

# === Parameters =====

# LQR parameter
lqr_Q = np.eye(5)
lqr_R = np.eye(2)
dt = 0.1  # time tick[s]
L = 2.9  # Wheel base of the vehicle [m]
max_steer = np.deg2rad(25.0)  # maximum steering angle[rad]

lqr_Q[0, 0] = 0.1
lqr_Q[1, 1] = 0.01
lqr_Q[2, 2] = 0.1
lqr_Q[3, 3] = 0.01
lqr_Q[4, 4] = 0.6

lqr_R[0, 0] = 15.0
lqr_R[1, 1] = 15.0

show_animation = True

def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)

    Parameters
    ----------
    x : float or array_like
        A angle or an array of angles. This array is flattened for
        the calculation. When an angle is provided, a float angle is returned.
    zero_2_2pi : bool, optional
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


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


def update(state, a, delta):

    if delta >= max_steer:
        delta = max_steer
    if delta <= - max_steer:
        delta = - max_steer

    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    state.v = state.v + a * dt

    return state


def calculate_lateral_acceleration(v, delta, L):
    return (v ** 2 / L) * math.tan(delta)


def pi_2_pi(angle):
    return angle_mod(angle)


def solve_dare(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    x = Q
    x_next = Q
    max_iter = 150
    eps = 0.01

    for i in range(max_iter):
        x_next = A.T @ x @ A - A.T @ x @ B @ \
                 la.inv(R + B.T @ x @ B) @ B.T @ x @ A + Q
        if (abs(x_next - x)).max() < eps:
            break
        x = x_next

    return x_next


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_dare(A, B, Q, R)

    # compute the LQR gain
    K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    eig_result = la.eig(A - B @ K)

    return K, X, eig_result[0]



def lqr_speed_steering_control(state, cx, cy, cyaw, ck, pe, pth_e,jerk_factor, sp, Q, R):
    ind, e = calc_nearest_index(state, cx, cy, cyaw)

    tv = sp[ind]

    k = ck[ind]
    v = state.v
    th_e = pi_2_pi(state.yaw - cyaw[ind])

    # A = [1.0, dt, 0.0, 0.0, 0.0
    #      0.0, 0.0, v, 0.0, 0.0]
    #      0.0, 0.0, 1.0, dt, 0.0]
    #      0.0, 0.0, 0.0, 0.0, 0.0]
    #      0.0, 0.0, 0.0, 0.0, 1.0]
    A = np.zeros((5, 5))
    A[0, 0] = 1.0
    A[0, 1] = dt
    A[1, 2] = v
    A[2, 2] = 1.0
    A[2, 3] = dt
    A[4, 4] = 1.0

    # B = [0.0, 0.0
    #     0.0, 0.0
    #     0.0, 0.0
    #     v/L, 0.0
    #     0.0, dt]
    B = np.zeros((5, 2))
    B[3, 0] = v / L
    B[4, 1] = dt

    K, _, _ = dlqr(A, B, Q, R)

    # state vector
    # x = [e, dot_e, th_e, dot_th_e, delta_v]
    # e: lateral distance to the path
    # dot_e: derivative of e
    # th_e: angle difference to the path
    # dot_th_e: derivative of th_e
    # delta_v: difference between current speed and target speed
    x = np.zeros((5, 1))
    x[0, 0] = e
    x[1, 0] = 0
    x[2, 0] = th_e
    x[3, 0] = (th_e - pth_e) / dt
    x[4, 0] = v - tv

    # input vector
    # u = [delta, accel]
    # delta: steering angle
    # accel: acceleration
    ustar = -K @ x

    # calc steering input
    ff = math.atan2(L * k, 1)  # feedforward steering angle
    fb = pi_2_pi(ustar[0, 0])  # feedback steering angle
    delta = fb + ff

    # calc accel input
    accel = ustar[1, 0]

    return delta, ind, e, th_e, accel


def calc_nearest_index(state, cx, cy, cyaw):
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind)

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    # Project RMS error onto vehicle
    vehicle_vector = [-np.cos(state.yaw + np.pi / 2),
                      -np.sin(state.yaw + np.pi / 2)]
    lateral_error_from_vehicle = np.dot([dx[ind], dy[ind]], vehicle_vector)

    return ind, -lateral_error_from_vehicle


def do_simulation(cx, cy, cyaw, ck, speed_profile, goal):
    T = 500.0  # max simulation time
    goal_dis = 3.0
    stop_speed = 0.05

    state = State(x=-0.0, y=-0.0, yaw=0.0, v=0.0)

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    delta = [0.0]
    acell = [0.0]
    errror_history = [0.0]
    jerk_history = [0.0]
    lateral_acceleration = 0.0
    prev_lat_acc = 0.0
    dl = 0.0

    e, e_th = 0.0, 0.0


    while T >= time:

        lateral_acceleration = calculate_lateral_acceleration(state.v, dl, L)
        jerk_factor = (lateral_acceleration - prev_lat_acc) / dt

        dl, target_ind, e, e_th, ai = lqr_speed_steering_control(
            state, cx, cy, cyaw, ck, e, e_th, jerk_factor, speed_profile, lqr_Q, lqr_R)
        
        
        prev_lat_acc = lateral_acceleration

        state = update(state, ai, dl)


        if abs(state.v) <= stop_speed:
            target_ind += 1

        time = time + dt

        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        if math.hypot(dx, dy) <= goal_dis:
            print("Goal")
            break

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        delta.append(dl)
        acell.append(ai)
        errror_history.append(e)
        jerk_history.append(jerk_factor)

        if target_ind % 1 == 0 and show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("speed[km/h]:" + str(round(state.v * 3.6, 2))
                      + ",target index:" + str(target_ind))
            plt.pause(0.0001)

    return t, x, y, yaw, v, delta, acell, errror_history, jerk_history


def calc_speed_profile(cyaw, target_speed):
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


def main():
    print("LQR steering control tracking start!!")
    ax = [0.0, 20.0, 40.0,  60.0, 120.0, 180.0, 280.0, 320.0, 340.0, 380.0, 400.0]
    ay = [0.0, 0.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 150.0, 160.0, 180.0]



    goal = [ax[-1], ay[-1]]

    # time the following function call

    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=0.1)
    target_speed = 50.0 / 3.6  # simulation parameter km/h -> m/s


    sp = calc_speed_profile(cyaw, target_speed)

    t, x, y, yaw, v, delta, acell, error_history,jerk_history = do_simulation(cx, cy, cyaw, ck, sp, goal)

    if show_animation:  # pragma: no cover
        plt.close()
        plt.subplots(1)
        plt.plot(ax, ay, "xb", label="waypoints")
        plt.plot(cx, cy, "-r", label="target course")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()
        plt.subplots(1)

        plt.plot(t, np.array(v)*3.6, label="speed")
        plt.grid(True)
        plt.xlabel("Time [sec]")
        plt.ylabel("Speed [m/s]")
        plt.legend()

        plt.subplots(1)
        plt.plot(s, [np.rad2deg(iyaw) for iyaw in cyaw], "-r", label="yaw")
        plt.grid(True)
        plt.legend()
        plt.xlabel("line length[m]")
        plt.ylabel("yaw angle[deg]")

        plt.subplots(1)
        plt.plot(s, ck, "-r", label="curvature")
        plt.grid(True)
        plt.legend()
        plt.xlabel("line length[m]")
        plt.ylabel("curvature [1/m]")

        plt.subplots(1)
        plt.plot(t, np.rad2deg(np.array(delta)), "-r", label="steering angle")
        plt.grid(True)
        plt.legend()
        plt.xlabel("Time [sec]")
        plt.ylabel("Steering angle[deg]")

        plt.subplots(1)
        plt.plot(t, acell, "-r", label="acceleration")
        plt.grid(True)
        plt.legend()
        plt.xlabel("Time [sec]")
        plt.ylabel("acceleration[m/ss]")

        plt.subplots(1)
        plt.plot(t, error_history, "-r", label="lateral error")
        plt.grid(True)
        plt.legend()    
        plt.xlabel("Time [sec]")
        plt.ylabel("lateral error[m]")

        plt.subplots(1)
        plt.plot(t, np.array(jerk_history), "-r", label="Jerk over time")
        plt.grid(True)
        plt.legend()
        plt.xlabel("Time [sec]")
        plt.ylabel("Jerk [m/s^3]")

        plt.show()


if __name__ == '__main__':
    main()