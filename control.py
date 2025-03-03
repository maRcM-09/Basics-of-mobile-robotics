import numpy as np

class Controller:
    def __init__(self):
        self.path = []  # Path the robot follows
        self.waypoint = 1  # Current waypoint index
        self.prev_angle = 0

    def get_controller_commands(self, x_est, theta_est, x_des):
        """
        This function implements a P-controller for orienting the robot
        and switches to straight motion when the orientation error is small.

        """
        # Controller parameters
        K_P = 0.4  # Proportional gain for angle control
        ALIGNMENT_THRESHOLD = 10  # Degrees, below which the robot moves straight
        BASE_SPEED = 150  # Base speed when moving straight
        K_P_ROTATION = 2

        # Current orientation of the robot
        thymio_angle = theta_est  # Degrees

        # Convert desired position to numpy array
        x_des = np.array(x_des)
        x_est = np.array(x_est)

        # Calculate the angle to the target position
        rho = x_des - x_est  # Difference vector
        gamma = (np.arctan2(rho[1], rho[0])+np.pi)
        if gamma > np.pi:
            gamma -= 2*np.pi
        gamma = np.rad2deg(gamma)
        angle_error = (gamma - thymio_angle+180)  # Normalize to [-180, 180]
        if angle_error > 180:
            angle_error-=360

        # If the angle error is small, move straight
        if abs(angle_error) < ALIGNMENT_THRESHOLD:
            # Apply P-controller for angular velocity
            angular_correction = 0.02 * angle_error
            w_l = BASE_SPEED + angular_correction
            w_r = BASE_SPEED - angular_correction
        else:
            # Apply P-controller for angular velocity
            
            angular_correction = K_P_ROTATION * angle_error

            # Calculate motor speeds
            if angle_error < 0:

                w_l =   - np.abs(angular_correction)
                w_r =   + np.abs(angular_correction)
            else:
                w_l =   + np.abs(angular_correction)
                w_r =   - np.abs(angular_correction)

        return int(w_l), int(w_r)

