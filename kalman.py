import numpy as np

class EKFLocalization:
    def __init__(self, dt, Q, R):
        """
        Extended Kalman Filter for localization.
        
        :param dt: Time step (s)
        :param process_noise_std: Standard deviation of process noise [x, y, theta, v, omega]
        :param measurement_noise_std: Standard deviation of measurement noise [x, y, theta, v, omega]
        """
        self.dt = dt
        self.x = np.array([0.0, 0.0, 0.0])  # Initial state: [x, y, theta, v, omega]
        self.P = np.eye(3) * 0.1  # Initial state covariance
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.z = np.array([0.0,0.0,0.0])

    def motion_model(self, state, control):
        """
        Nonlinear motion model.
        :param state: Current state [x, y, theta, v, omega]
        :param control: Control inputs [v, omega]
        :return: Predicted next state
        """
        x, y, theta = state
        v, omega = control  # Control inputs: velocity and angular velocity
        new_state = np.array([
            x + v * np.cos(theta) * self.dt,
            y + v * np.sin(theta) * self.dt,
            theta + omega * self.dt,
        ])
        while np.abs(new_state[2]) > np.pi:
            if new_state[2] < 0:
                new_state[2]+=2*np.pi
            else:
                new_state[2]-=2*np.pi

        return new_state

    def jacobian_F(self, state, control):
        """
        Jacobian of the motion model with respect to the state.
        :param state: Current state [x, y, theta, v, omega]
        :param control: Control inputs [v, omega]
        :return: Jacobian matrix (5x5)
        """
        _, _, theta = state
        v , _ = control
        return np.array([
            [1, 0, -v * np.sin(theta) * self.dt],
            [0, 1,  v * np.cos(theta) * self.dt],
            [0, 0, 1]
        ])

    def measurement_model(self, state):
        return state  # Identity mapping for simplicity

    def jacobian_H(self, state):
        return np.eye(3)

    def predict(self, control):
        """
        EKF predict step.
        :param control: Control inputs [v, omega]
        """
        # Compute Jacobian of the motion model
        F = self.jacobian_F(self.x, control)

        # Predict state
        self.x = self.motion_model(self.x, control)

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self):
        """
        EKF update step.
        :param z: Measurement vector [x, y, theta, v, omega]
        """
        # Compute Jacobian of the measurement model
        H = self.jacobian_H(self.x)

        # Compute Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        y = self.z - self.measurement_model(self.x)  # Measurement residual
        if np.abs(y[2]) > np.pi:
            if y[2]>0:
                y[2]-=2*np.pi
            elif y[2]<0:
                y[2]+=2*np.pi
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P