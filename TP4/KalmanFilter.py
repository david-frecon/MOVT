import numpy as np

class KalmanFilter():
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.u = np.array([u_x, u_y])
        self.xk = np.zeros((4, 1))
        self.A = np.identity(4)
        self.A[0][2] = dt
        self.A[1][3] = dt
        self.B = np.zeros((4, 2))
        self.B[0][0] = 0.5 * dt * dt
        self.B[1][1] = 0.5 * dt * dt
        self.B[2][0] = dt   
        self.B[3][1] = dt
        self.H = np.zeros((2, 4))
        self.H[0][0] = 1
        self.H[1][1] = 1
        Q = [[0.25 * dt**4, 0, 0.5 * dt**3, 0],
             [0, 0.25 * dt**4, 0, 0.5 * dt**3],
             [0.5 * dt**3, 0, dt**2, 0],
             [0, 0.5 * dt**3, 0, dt**2]]
        self.Q = np.array(Q) * std_acc**2
        self.R = np.array([[x_std_meas**2, 0],
                           [0, y_std_meas**2]])
        self.P = np.identity(4)
        self.xk_h = np.zeros((4, 1))
        self.Pk_h = np.identity(4)
        
    def predict(self):
        self.xk_hat = self.A @ self.xk + self.B @ self.u
        self.Pk_hat = self.A @ self.P @ self.A.T + self.Q

    def update(self, zk):
        Sk = self.H @ self.Pk_hat @ self.H.T + self.R
        Kk = self.Pk_hat @ self.H.T @ np.linalg.inv(Sk)

        self.xk = self.xk_hat + Kk @ (zk - self.H @ self.xk_hat)
        self.Pk = (np.identity(4) - Kk @ self.H) @ self.Pk_hat
