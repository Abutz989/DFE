import matplotlib.pyplot as plt
from Physics_Models.Drone_physics import *
import numpy as np
import scipy 

class PID():
    def __init__(self, Gains=None, dt=0.01, N=100, sat=30):
        if Gains is None:
            Gains = [1., 1., 1.]
        self.dt = dt
        self.Kp = Gains[0]
        self.Ki = Gains[1]
        self.Kd = Gains[2]
        
        self.prev_e = [0.0, 0.0]
        self.prev_u = [0.0, 0.0]
        self.sat = sat
        self.N = N

    def reset(self):
        self.prev_e = [0.0, 0.0]
        self.prev_u = [0.0, 0.0]

    def culcControl(self, err):
        Ts = self.dt
        N = self.N
        Kp = self.Kp
        Ki = self.Ki
        Kd = self.Kd
        a0 = (1 + N * Ts)
        a1 = -(2 + N * Ts)
        a2 = 1.0
        b0 = Kp * (1 + N * Ts) + Ki * Ts * (1 + N * Ts) + Kd * N
        b1 = -(Kp * (2 + N * Ts) + Ki * Ts + 2 * Kd * N)
        b2 = Kp + Kd * N

        ku1, ku2, ke0, ke1, ke2 = a1 / a0, a2 / a0, b0 / a0, b1 / a0, b2 / a0
        e0 = err
        u1, u2 = self.prev_u[0], self.prev_u[1]
        e1, e2 = self.prev_e[0], self.prev_e[1]
        u0 = -ku1 * u1 - ku2 * u2 + ke0 * e0 + ke1 * e1 + ke2 * e2

        self.prev_u[0], self.prev_u[1] = u0, u1
        self.prev_e[0], self.prev_e[1] = e0, e1

        return np.clip(u0,-self.sat, self.sat)


class MovingAvarge():
    def __init__(self, windowsize):
        self.cell = [1] * windowsize

    def roll(self, value):
        cell = self.cell
        cell.pop(0)
        self.cell = list((*cell, value))

    def mean(self):
        return sum(self.cell) / len(self.cell)

class LPF():
    def __init__(self, Wn =10, dt=0.01, sat=30):
        
        self.dt = dt
        b, a = scipy.signal.iirfilter(N=2, Wn=Wn, fs=1/dt, btype="low", ftype="butter")
        self.a = a
        self.b = b
        self.prev_x = [0.0, 0.0]
        self.prev_y = [0.0, 0.0]
        self.sat = sat
        

    def reset(self):
        self.prev_x = [0.0, 0.0]
        self.prev_y = [0.0, 0.0]

    def culcFilt(self, x):
        
        a0 = self.a[0]
        a1 = self.a[1]
        a2 = self.a[2]
        b0 = self.b[0]
        b1 = self.b[1]
        b2 = self.b[2]

        ky1, ky2, kx0, kx1, kx2 = a1 / a0, a2 / a0, b0 / a0, b1 / a0, b2 / a0
        x0 = x
        y1, y2 = self.prev_y[0], self.prev_y[1]
        x1, x2 = self.prev_x[0], self.prev_x[1]
        y0 = -ky1 * y1 - ky2 * y2 + kx0 * x0 + kx1 * x1 + kx2 * x2

        self.prev_y[0], self.prev_y[1] = y0, y1
        self.prev_x[0], self.prev_x[1] = x0, x1

        return np.clip(y0, -self.sat, self.sat)


class Dis_Der():
    def __init__(self, Var_len=1, h=0.001, sat=10000):
        Gains = [11 / 6, -3., 3 / 2, -1 / 3]
        self.h = h
        self.u = np.zeros((Var_len, 4))
        self.gains = np.array(Gains).reshape(-1, 1)
        self.Var_len = Var_len
        self.du = np.zeros((Var_len,))
        self.sat = sat

    def culcDer(self, u):
        u = np.array(u).reshape(-1, 1)
        u1, u2, u3 = self.u[:, 0].reshape(-1, 1), self.u[:, 1].reshape(-1, 1), self.u[:, 2].reshape(-1, 1)
        self.u = np.concatenate((u, u1, u2, u3), 1)
        derivative = np.clip(((self.u @ self.gains) / self.h), -self.sat, self.sat)
        self.du = derivative
        return derivative


class Dis_Intgral():
    def __init__(self, Var_len=1, h=0.001, sat=10000):
        Gains = [.5, .5]
        self.h = h
        self.u = np.zeros((Var_len, 1))
        self.gains =  np.array(Gains).reshape(-1, 1)
        self.Var_len = Var_len
        self.Iu = np.zeros((Var_len))
        self.sat = sat

    def culcInt(self, u):
        u = np.array(u).reshape(-1, 1)
        u1 = self.u.reshape(-1, 1)
        self.u = u1
        self.Iu += np.clip((u+u1)/2*self.h, -self.sat, self.sat).flatten()
        return np.copy(self.Iu)


def Wrap2Pi(u):
    u = (u + 1 * np.pi) % (1 * np.pi)
    return u


def Fold2HalfPi(u):
    #     if u > np.pi/2:
    u = np.array(u).reshape(-1, 1)
    relv_idx = np.where(u > np.pi / 2)
    u[relv_idx] = np.pi / 2 - u[relv_idx] % (np.pi / 2)
    return u


        

    
class VelCommand(PhysicsSim):
    def __init__(self, Physics, dt=0.01):
        self.dt = dt
        self.pyhsics = Physics
        # one time calculation
        l, b, d = self.pyhsics.get_properties()[2:5]
        T = np.array([[b, b, b, b],
                      [0, -l * b, 0, l * b],
                      [-l * b, 0, l * b, 0],
                      [-d, d, -d, d]])
        self.Inv_T = np.linalg.inv(T)
        self.Reset_Controllers()

    def Reset_Controllers(self):
        self.X_cntrl = PID([20, 4, 10], self.dt, N=100, sat=30)
        self.Y_cntrl = PID([20, 4, 10], self.dt, N=100, sat=30)
        self.Z_cntrl = PID([40, 4, 10], self.dt, N=100, sat=30)
        self.psi_cntrl = PID([20, 1, 15], self.dt, N=100, sat=1)
        self.theta_cntrl = PID([20, 1, 15], self.dt, N=100, sat=1)
        self.phi_cntrl = PID([20, 1, 15], self.dt, N=100, sat=1)
       
    def step(self, ref, State, reset=False):
        mass = self.pyhsics.get_properties()[0]
        x, y, z, phi, theta, psi = State

        err_x = ref[0] - x
        err_y = ref[1] - y
        err_z = ref[2] - z

        if reset:
            self.Reset_Controllers()

        x_2dot_d = self.X_cntrl.culcControl(err_x)
        y_2dot_d = self.Y_cntrl.culcControl(err_y)
        z_2dot_d = self.Z_cntrl.culcControl(err_z) + mass * 9.81

        eta = self.euler_conv(x_2dot_d, y_2dot_d, z_2dot_d, psi_d=0)
        err_phi = ref[3] + eta[0] - phi
        err_theta = ref[4] + eta[1] - theta
        err_psi = ref[5] + eta[2] - psi

        theta_2dot_d = self.theta_cntrl.culcControl(err_theta)
        phi_2dot_d = self.phi_cntrl.culcControl(err_phi)
        psi_2dot_d = self.psi_cntrl.culcControl(err_psi)

        t_d = self.torque_conv(phi_2dot_d, theta_2dot_d, psi_2dot_d)
        f_d = self.force_conv(z_2dot_d)

        omega = self.Get_Omega(t_d, f_d * 0)

        rotors_speed = np.sqrt(np.abs(omega)) * np.sign(omega)
        rotors_speed = rotors_speed.flatten('C')

        omega = self.Get_Omega(t_d * 0, f_d)

        rotors_speed_trust = np.sqrt(np.abs(omega)) * np.sign(omega)
        rotors_speed_trust = rotors_speed_trust.flatten('C')
        rotors_speed += rotors_speed_trust
        rotors_speed = rotors_speed*np.array([1., -1., 1., -1.])
        return rotors_speed, np.array([ref[0], ref[1], ref[2], eta[0], eta[1], eta[2]])

    def euler_conv(self, x_2dot_d, y_2dot_d, z_2dot_d, psi_d):
        d = np.sqrt(x_2dot_d * x_2dot_d + y_2dot_d * y_2dot_d + z_2dot_d * z_2dot_d)

        if (-0.001 < d) and (d < 0.001):
            phi_d = 0
        else:
            phi_d = np.arcsin((x_2dot_d * np.sin(psi_d) - y_2dot_d * np.cos(psi_d)) / (d + 1e-5))
        theta_d = np.arctan2(x_2dot_d * np.cos(psi_d) + y_2dot_d * np.sin(psi_d), z_2dot_d)
        sat_ang = np.pi / 6
        phi_d = np.clip(phi_d, -sat_ang, sat_ang )
        theta_d = np.clip(theta_d, -sat_ang, sat_ang )
        return [phi_d, theta_d, psi_d]

    def torque_conv(self, phi_2dot_d, theta_2dot_d, psi_2dot_d):
        x, y, z, phi, theta, psi = self.pyhsics.get_state()[0]
        mass, Inertia = self.pyhsics.get_properties()[0:2]
        I = np.diag(Inertia)
        tr_d = np.array([phi_2dot_d, theta_2dot_d, psi_2dot_d])
        Conv = np.array([[1, 0, -S(theta)], [0, C(phi), S(phi) * C(theta)], [0, -S(phi), C(phi) * C(theta)]])
        return (I @ Conv) @ tr_d

    def force_conv(self, z_2dot_d):
        mass = self.pyhsics.get_properties()[0]
        x, y, z, phi, theta, psi = self.pyhsics.get_state()[0]
        R = body_to_earth_frame(phi, theta, psi)
        #         R = earth_to_body_frame (phi,theta,psi)
        u = np.array([[0], [0], [z_2dot_d]])
        d = mass * R @ u
        return d[2]

    def Get_Omega(self, t_d, f_d):
        u = np.array([[f_d[0]], [t_d[0]], [t_d[1]], [t_d[2]]])
        return self.Inv_T @ u

    def Soft_Error(self, u, tol):
        if np.abs(u) <= tol:
            u = 0
        return u


class MotorCommand(PhysicsSim):
    def __init__(self, Physics, dt=0.01):
        self.dt = dt
        self.pyhsics = Physics

        window_size = 30
        Filt_f = 5
        self.Me1 = MovingAvarge(window_size)
        self.Me2 = MovingAvarge(window_size)
        self.Me3 = MovingAvarge(window_size)
        self.Me0 = MovingAvarge(window_size)
        self.Le1 = LPF(Wn = Filt_f, dt=self.dt, sat=4)
        self.Le2 = LPF(Wn = Filt_f, dt=self.dt, sat=4)
        self.Le3 = LPF(Wn = Filt_f, dt=self.dt, sat=4)
        self.Le0 = LPF(Wn = Filt_f, dt=self.dt, sat=4)
        self.Master_position = 0.
        self.Reset_Controllers()

    def Reset_Controllers(self):
        self.M0_cntrl = PID([0.05, 1., 0.], self.dt, N=1000, sat=12)
        self.M1_cntrl = PID([0.05, 1., 0.], self.dt, N=1000, sat=12)
        self.M2_cntrl = PID([0.05, 1., 0.], self.dt, N=1000, sat=12)
        self.M3_cntrl = PID([0.05, 1., 0.], self.dt, N=1000, sat=12)
        

    def step(self, refPhase, ref_rotors_speed):

        rotor_position, rotors_speed = self.pyhsics.get_rotors_state()
        rotors_speed_error = ref_rotors_speed - rotors_speed

        V_cmd = np.array([0., 0., 0., 0.])
        V_cmd[0] = self.M0_cntrl.culcControl(rotors_speed_error[0])
        V_cmd[1] = self.M1_cntrl.culcControl(rotors_speed_error[1])
        V_cmd[2] = self.M2_cntrl.culcControl(rotors_speed_error[2])
        V_cmd[3] = self.M3_cntrl.culcControl(rotors_speed_error[3])
        # V_cmd = ref_rotors_speed
        Master = rotor_position[0]
        PhaseErr = np.array([0., 0., 0., 0.])
        PhaseErr[0] = Wrap2Pi(Wrap2Pi(Master) - Wrap2Pi(rotor_position[0]))
        PhaseErr[1] = Wrap2Pi(Wrap2Pi(refPhase - Wrap2Pi(Master)) - Wrap2Pi(rotor_position[1]))
        PhaseErr[2] = Wrap2Pi(Wrap2Pi(Master) - Wrap2Pi(rotor_position[2]))
        PhaseErr[3] = Wrap2Pi(Wrap2Pi(refPhase - Wrap2Pi(Master)) - Wrap2Pi(rotor_position[3]))

#         self.Me0.roll(PhaseErr[0])
#         self.Me1.roll(PhaseErr[1])
#         self.Me2.roll(PhaseErr[2])
#         self.Me3.roll(PhaseErr[3])

#         PhaseErr[0] = self.Me0.mean()
#         PhaseErr[1] = self.Me1.mean()
#         PhaseErr[2] = self.Me2.mean()
#         PhaseErr[3] = self.Me3.mean()
        
        # PhaseErr[0] = self.Le0.culcFilt(PhaseErr[0])
        # PhaseErr[1] = self.Le1.culcFilt(PhaseErr[1])
        # PhaseErr[2] = self.Le2.culcFilt(PhaseErr[2])
        # PhaseErr[3] = self.Le3.culcFilt(PhaseErr[3])
            
    
        return V_cmd, PhaseErr , rotors_speed_error  # added new output


class Telemetry(PhysicsSim):

    def __init__(self, Physics):
        # Goal
        self.pyhsics = Physics
        self.time = []
        self.x = []
        self.y = []
        self.z = []
        self.psi = []
        self.theta = []
        self.phi = []
        self.rotors = []
        self.ref_x = []
        self.ref_y = []
        self.ref_z = []
        self.ref_phi = []
        self.ref_theta = []
        self.ref_psi = []
        self.phaseE0 = []
        self.phaseE1 = []
        self.phaseE2 = []
        self.phaseE3 = []

    def record(self, ref, rotor_position_err):
        x, y, z, phi, theta, psi = self.pyhsics.get_state()[0]
        time = self.pyhsics.get_state()[2]

        self.x.append(x)
        self.y.append(y)
        self.z.append(z)
        self.psi.append(psi)
        self.theta.append(theta)
        self.phi.append(phi)
        self.time.append(time)
        self.ref_x.append(ref[0])
        self.ref_y.append(ref[1])
        self.ref_z.append(ref[2])
        self.ref_phi.append(ref[3])
        self.ref_theta.append(ref[4])
        self.ref_psi.append(ref[5])
        
        self.phaseE0.append([rotor_position_err[0]])
        self.phaseE1.append(Fold2HalfPi(rotor_position_err[1])[0])
        self.phaseE2.append(Fold2HalfPi(rotor_position_err[2])[0])
        self.phaseE3.append(Fold2HalfPi(rotor_position_err[3])[0])
        # self.phaseE1.append([rotor_position_err[1]])
        # self.phaseE2.append([rotor_position_err[2]])
        # self.phaseE3.append([rotor_position_err[3]])

    def PlotZ(self, axs):
        axs.plot(self.time, self.z, 'b', self.time, self.ref_z, '--r')
        axs.set_ylabel('z')
        axs.grid(True)

    def PlotX(self, axs):
        axs.plot(self.time, self.x, 'b', self.time, self.ref_x, '--r')
        axs.set_ylabel('x')
        axs.grid(True)

    def PlotY(self, axs):
        axs.plot(self.time, self.y, 'b', self.time, self.ref_y, '--r')
        axs.set_ylabel('y')
        axs.grid(True)

    def PlotRoll(self, axs):
        axs.plot(self.time, self.phi, 'b', self.time, self.ref_phi, '--r')
        axs.set_ylabel('Roll')
        axs.grid(True)

    def PlotPitch(self, axs):
        axs.plot(self.time, self.theta, 'b', self.time, self.ref_theta, '--r')
        axs.set_ylabel('Pitch')
        axs.grid(True)

    def PlotYaw(self, axs):
        axs.plot(self.time, self.psi, 'b', self.time, self.ref_psi, '--r')
        axs.set_ylabel('Yaw')
        axs.grid(True)

    def PlotE0(self, axs):
        axs.plot(self.time, self.phaseE0, 'b')
        axs.set_ylabel('Phase Error rotor 0')
        axs.grid(True)

    def PlotE1(self, axs):
        axs.plot(self.time, self.phaseE1, 'b')
        axs.set_ylabel('Phase Error rotor 1')
        axs.grid(True)

    def PlotE2(self, axs):
        axs.plot(self.time, self.phaseE2, 'b')
        axs.set_ylabel('Phase Error rotor 2')
        axs.grid(True)

    def PlotE3(self, axs):
        axs.plot(self.time, self.phaseE3, 'b')
        axs.set_ylabel('Phase Error rotor 3')
        axs.grid(True)

    def Plot3(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # Data for a three-dimensional line
        zline = self.z
        xline = self.x
        yline = self.y
        ax.plot3D(xline, yline, zline, 'gray')


