import numpy as np
import csv


def C(x):
    return np.cos(x)


def S(x):
    return np.sin(x)


def earth_to_body_frame(ii, jj, kk):
    # C^b_n
    # roll - ii , pitch - jj, yaw - kk
    R = [[C(kk) * C(jj), C(kk) * S(jj) * S(ii) - S(kk) * C(ii), C(kk) * S(jj) * C(ii) + S(kk) * S(ii)],
         [S(kk) * C(jj), S(kk) * S(jj) * S(ii) + C(kk) * C(ii), S(kk) * S(jj) * C(ii) - C(kk) * S(ii)],
         [-S(jj), C(jj) * S(ii), C(jj) * C(ii)]]
    return np.array(R)


def body_to_earth_frame(ii, jj, kk):
    # C^n_b
    return np.transpose(earth_to_body_frame(ii, jj, kk))


class PhysicsSim(object):
    def __init__(self, init_pose=None, init_velocities=None, init_angle_velocities=None, init_rotors_position=None,runtime = 5., dt = 0.01):
        self.init_pose = init_pose
        self.init_velocities = init_velocities
        self.init_angle_velocities = init_angle_velocities
        self.init_rotors_position = np.array([0.0, 0.0, 0.0, 0.0]) if init_rotors_position is None else np.copy(init_rotors_position)
        self.runtime = runtime

        self.gravity = 9.81  # m/s
        self.rho = 1.2
        self.mass = 1  # 300 g
        self.dt = dt  # Timestep
        self.b = 54.2e-6  #thrust factor
        self.d = 1.1e-6 # drag factor
        self.M0 = RotorSim(self.init_rotors_position[0],dt)
        self.M1 = RotorSim(self.init_rotors_position[1], dt)
        self.M2 = RotorSim(self.init_rotors_position[2], dt)
        self.M3 = RotorSim(self.init_rotors_position[3], dt)

        self.C_d = 0.05
        self.l_to_rotor = 0.24
        self.propeller_size = 0.1
        width, length, height = .3, .3, .15
        self.dims = np.array([width, length, height])  # x, y, z dimensions of quadcopter
        self.areas = np.array([length * height, width * height, width * length])
        I_x = 1 / 12. * self.mass * (height**2 + width**2)
        I_y = 1 / 12. * self.mass * (height**2 + length**2)
        I_z = 1 / 12. * self.mass * (width**2 + length**2)
        
        self.moments_of_inertia = np.array([I_x, I_y, I_z])  # moments of inertia

        env_bounds = 300.0
        self.lower_bounds = np.array([-env_bounds / 2, -env_bounds / 2, 0])
        self.upper_bounds = np.array([env_bounds / 2, env_bounds / 2, env_bounds])

        self.reset()

    def reset(self):
        self.time = 0.0
        self.pose = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]) if self.init_pose is None else np.copy(self.init_pose)
        self.v = np.array([0.0, 0.0, 0.0]) if self.init_velocities is None else np.copy(self.init_velocities)
        self.angular_v = np.array([0.0, 0.0, 0.0]) if self.init_angle_velocities is None else np.copy(self.init_angle_velocities)
        self.rotor_position = np.array([0.0, 0.0, 0.0, 0.0])
        self.rotor_speeds = np.array([0.0, 0.0, 0.0, 0.0])
        self.linear_accel = np.array([0.0, 0.0, 0.0])
        self.angular_accels = np.array([0.0, 0.0, 0.0])
        self.M0.reset(self.init_rotors_position[0])
        self.M1.reset(self.init_rotors_position[1])
        self.M2.reset(self.init_rotors_position[2])
        self.M3.reset(self.init_rotors_position[3])
        self.done = False
      
    def get_properties(self):
        return self.mass , self.moments_of_inertia, self.l_to_rotor,self.b,self.d 

    def get_state(self):
        x,y,z,phi,theta,psi = self.pose
        x_dot, y_dot, z_dot = self.v
        psi_dot, theta_dot ,phi_dot = self.angular_v
        rotor_position = self.rotor_position
        state = np.array([x, y, z, phi, theta, psi])
        time = self.time
        return state, rotor_position, time

    def get_rotors_state(self):

        return self.rotor_position, self.rotor_speeds

    def find_body_velocity(self):
        body_velocity = np.matmul(earth_to_body_frame(*list(self.pose[3:])), self.v)
        return body_velocity
   
    def find_linear_drag(self):
        linear_drag = 0.5 * self.rho * self.find_body_velocity()**2 * self.areas * self.C_d
        return linear_drag
    
    def find_linear_forces(self, thrusts):
        gravity_force = self.mass * self.gravity * np.array([0, 0, 1])
        thrust_body_force = thrusts[0]*np.array([0,0,1])
        drag_body_force = -self.find_linear_drag()# Drag in body frame
        body_forces = thrust_body_force + drag_body_force
        linear_forces = np.matmul(earth_to_body_frame(*list(self.pose[3:])), body_forces)
#         linear_forces = np.matmul(body_to_earth_frame(*list(self.pose[3:])), body_forces)
        linear_forces -= gravity_force # z axis faces up
        return linear_forces

    def find_moments(self, thrusts):
        thrust_moment = np.array([(thrusts[3] - thrusts[2]) * self.l_to_rotor,
                            (thrusts[1] - thrusts[0]) * self.l_to_rotor,
                            0])# (thrusts[2] + thrusts[3] - thrusts[0] - thrusts[1]) * self.T_q])  # Moment from thrust

        drag_moment =  self.C_d * 0.5 * self.rho * self.angular_v * np.absolute(self.angular_v) * self.areas * self.dims * self.dims
        moments = thrust_moment - drag_moment # + motor_inertia_moment
        return moments

    def find_propeler_thrust(self, rotor_speeds):
        b = self.b
        d = self.d
        l = self.l_to_rotor
        T = np.array([[b, b, b, b],
                      [0,-l*b, 0, l*b],
                      [-l*b, 0, l*b,0],
                      [-d, d, -d, d]])
         
        return T@np.square(rotor_speeds)

    def next_timestep(self, V):
        
        dt = self.dt

        theta0, w0 = self.M0.next_timestep(V[0])
        theta1, w1 = self.M1.next_timestep(V[1])
        theta2, w2 = self.M2.next_timestep(V[2])
        theta3, w3 = self.M3.next_timestep(V[3])
        rotor_speeds = np.array([w0,w1,w2,w3])
        # rotor_speeds = np.array([V[0], -V[1],V[2],-V[3]])
        self.rotor_speeds = rotor_speeds
        self.rotor_position = np.array([theta0,theta1,theta2,theta3])
            
        thrusts = self.find_propeler_thrust(rotor_speeds)
        self.linear_accel = self.find_linear_forces(thrusts) / self.mass

        position = self.pose[:3] + self.v * dt + 0.5 * self.linear_accel * dt**2
        self.v += self.linear_accel *dt

        moments = thrusts[1:]

        self.angular_accels = moments / self.moments_of_inertia
        angles = self.pose[3:] + self.angular_v *dt + 0.5 * self.angular_accels * dt ** 2
        self.angular_v = self.angular_v + self.angular_accels * dt

        new_positions = []
        for ii in range(3):
            if position[ii] <= self.lower_bounds[ii]:
                new_positions.append(self.lower_bounds[ii])
                self.done = True
            elif position[ii] > self.upper_bounds[ii]:
                new_positions.append(self.upper_bounds[ii])
                self.done = True
            else:
                new_positions.append(position[ii])

        self.pose = np.array(new_positions + list(angles))
        self.time += dt
        if self.time > self.runtime:
            self.done = True
        return self.done


class RotorSim(object):
    def __init__(self, init_pose=0.0,dt=0.01):

        self.k = 6.3e-3
        self.R = 0.6*0.1
        self.Jtp = 104e-6
        self.N = 3
        self.pos = init_pose
        self.w = 0.0
        self.dt = dt  # Timestep
        self.d = 1.1e-6  # drag factor

    def reset(self, init_pose=0.0):
        self.pos = init_pose
        self.w = 0.0

    def next_timestep(self, V):

        dt = self.dt
        w_dot  = (V*self.k/self.Jtp/self.R*self.N)-self.w*(self.k**2/self.Jtp/self.R*self.N**2) - (self.w**2)*(self.d/self.Jtp)
        self.w = self.w + w_dot*dt
        self.pos = self.pos + self.w*dt + 0.5*w_dot*dt**2
        self.pos = (self.pos+ 2 * np.pi) % (2 * np.pi)
        return self.pos,self.w
        