
import gym
from gym import spaces
import numpy as np
from Physics_Models.ControlAlgTestMod import *
from Physics_Models.Drone_physics import PhysicsSim

class DroneEnv(gym.Env):
    metadata = {
        'render.modes': ['rgb_array']
    }
    def __init__(self, runtime=10., dt=0.001, seed=42,nAgents = 3):
        super(DroneEnv, self).__init__()
        self.damping_time = 4
        self.runtime = runtime + self.damping_time                                    # time limit of the episode
        self.dt = dt
        self.nAgents = nAgents
        max_act = np.ones(3,)
        max_obs = np.ones((18,3)) # [Ph_err(1)X3, cntr_cmd(1)X3,vel_err(1)X3,dPos_err(6),Action(1)X3]
        # self.observation_space = spaces.Dict({str(i) : spaces.Box(low=-max_obs, high=max_obs, dtype=np.float32) for i in range(self.nAgents)})
        self.observation_space = spaces.Box(low=-max_obs, high=max_obs, dtype=np.float32)
        self.action_space = spaces.Box(low=-max_act, high=max_act, dtype=np.float32)

        
        

    def reset(self, seed=None):
     
        init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose
        init_velocities = np.array([0., 0., 0.])         # initial velocities
        init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
        init_rotors_position = np.array([1., 2., 3., 4.])
        self.Plant = PhysicsSim(init_pose, init_velocities, init_angle_velocities,
                                init_rotors_position, self.runtime, self.dt)
        self.V_cmd = VelCommand(self.Plant,self.dt)
        self.P_cmd = MotorCommand(self.Plant,self.dt)

        self.Action = np.zeros((3,))
        self.Action_k = np.zeros((3,))
        self.Action_kk = np.zeros((3,))

        self.Control_cmd =  np.zeros((3,))
        self.Control_cmd_k = np.zeros((3,))
        self.Control_cmd_kk = np.zeros((3,))

        self.Phase_err = np.zeros((3,))  # init_rotors_position[1:] - init_rotors_position[0]
        self.Phase_err_k = np.zeros((3,))
        self.Phase_err_kk = np.zeros((3,))

        self.Rotors_Vel_err = np.zeros((3,))
        self.Rotors_Vel_err_k = np.zeros((3,))
        self.Rotors_Vel_err_kk = np.zeros((3,))

        self.Pos_err = np.zeros((6,))
        self.d_Pos_err = Dis_Der(Var_len=len(self.Pos_err), h=self.dt, sat=10)
        # self.d_Phase_err = Dis_Der(Var_len=len(self.Phase_err), h=self.dt, sat=10)
        # self.LPF_phase_err  = LPF(Wn = 1, dt=self.dt, sat=4)
        
#         only for continuis action
        self.LPF_action  = LPF(Wn = 5, dt=self.dt, sat=1)
        # self.d_action = Dis_Der(Var_len = 4, h=self.dt, sat=10)
        
        self.LPF_Control_cmd  = LPF(Wn = 1, dt=self.dt, sat=12)
        self.d_Control_cmd = Dis_Der(Var_len=len(self.Control_cmd), h=self.dt, sat=120)
        

        self.curr_episode_reward = np.array([0.])
        self.TelemReset()
        self.FirstStep()

        return self._get_obs()

    def TelemReset(self):
        self.Telem = Telemetry(self.Plant)

    def FirstStep(self,ref_cmd = np.array([0., 0., 10., 0., 0., 0.])):
        refPhase = 0
        for i in range(round(self.damping_time /self.dt)):
            State, rotor_position, time = self.Plant.get_state()
            ref_rotors_speed, refValuse = self.V_cmd.step(ref_cmd, State)
            Volt_Input_cmd, rotor_position_err,rotors_speed_error = self.P_cmd.step(refPhase, ref_rotors_speed)
            self.Control_cmd = Volt_Input_cmd
            self.Rotors_Vel_err = rotors_speed_error[1:]
            falling = self.Plant.next_timestep(Volt_Input_cmd)
            self.Telem.record(refValuse, rotor_position_err)
    
    
    def step(self, action , ref_cmd = np.array([0., 0., 10., 0., 0., 0.])):

        self.Action_kk = self.Action_k
        self.Action_k = self.Action
        self.Action = action

        K_action = 6.0
        action = np.concatenate(([0],K_action * action)).flatten()
        refPhase = 0
        State, rotor_position, time = self.Plant.get_state()
        ref_rotors_speed, refValuse = self.V_cmd.step(ref_cmd, State)
        Volt_Input_cmd, rotor_position_err,rotors_speed_error = self.P_cmd.step(refPhase, ref_rotors_speed)

        self.Control_cmd_kk = self.Control_cmd_k
        self.Control_cmd_k = self.Control_cmd
        self.Control_cmd = Volt_Input_cmd[1:]

        LPF_action = self.LPF_action.culcFilt(action)
        Volt_Input = np.clip(Volt_Input_cmd + LPF_action, -12, 12)
        # Volt_Input = np.clip(Volt_Input_cmd + action, -12, 12)
        falling = self.Plant.next_timestep(Volt_Input)
        self.Telem.record(refValuse, rotor_position_err)

        self.Pos_err = ref_cmd - State
        self.d_Pos_err.culcDer(self.Pos_err)

        # LPF_ph_err = rotor_position_err[1:] 
        # LPF_ph_err = self.LPF_phase_err.culcFilt(rotor_position_err[1:])
        self.Phase_err_kk = self.Phase_err_k
        self.Phase_err_k = self.Phase_err
        self.Phase_err = rotor_position_err[1:]

        self.Rotors_Vel_err_kk = self.Rotors_Vel_err_k
        self.Rotors_Vel_err_k = self.Rotors_Vel_err
        self.Rotors_Vel_err = rotors_speed_error[1:]

        obs = self._get_obs()
        reward = np.copy(self.get_reward()).astype(np.single).item()
        done = self.is_done(falling)
        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        if mode != 'rgb_array':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        self.plot()

    def close(self):
        pass

    def plot(self):
        fig, axs = plt.subplots(nrows=3, ncols=2)
        fig.set_figwidth(20)
        fig.set_figheight(10)
        self.Telem.PlotX(axs[0, 0])
        self.Telem.PlotY(axs[1, 0])
        self.Telem.PlotZ(axs[2, 0])
        self.Telem.PlotRoll(axs[0, 1])
        self.Telem.PlotPitch(axs[1, 1])
        self.Telem.PlotYaw(axs[2, 1])

        fig1, axs1 = plt.subplots(3)
        fig1.set_figwidth(20)
        fig1.set_figheight(10)
        self.Telem.PlotE1(axs1[0])
        self.Telem.PlotE2(axs1[1])
        self.Telem.PlotE3(axs1[2])

    def _get_obs(self, action=np.zeros(4),):
        ph_err = np.copy(self.fix_phase(self.Phase_err)).reshape(-1, 1) / 3.142/2
        ph_err_k = np.copy(self.fix_phase(self.Phase_err_k)).reshape(-1, 1) / 3.142/2
        ph_err_kk = np.copy(self.fix_phase(self.Phase_err_kk)).reshape(-1, 1) / 3.142/2
        cntr_cmd = np.copy(self.Control_cmd).reshape(-1, 1) / 12.0
        cntr_cmd_k = np.copy(self.Control_cmd_k).reshape(-1, 1) / 12.0
        cntr_cmd_kk = np.copy(self.Control_cmd_kk).reshape(-1, 1) / 12.0
        rotor_vel_err = np.clip(np.copy(self.Rotors_Vel_err).reshape(-1, 1), -30, 30) / 30
        rotor_vel_err_k = np.clip(np.copy(self.Rotors_Vel_err_k).reshape(-1, 1), -30, 30) / 30
        rotor_vel_err_kk = np.clip(np.copy(self.Rotors_Vel_err_kk).reshape(-1, 1), -30, 30) / 30
        act = np.copy(self.Action).reshape(-1, 1)
        act_k = np.copy(self.Action_k).reshape(-1, 1)
        act_kk = np.copy(self.Action_kk).reshape(-1, 1)
        d_pos_err = np.copy(self.d_Pos_err.du).reshape(-1, 1) / 10.0

        # curr_state = {str(i): np.concatenate((ph_err[i],ph_err_k[i], ph_err_kk[i],
        #                                       cntr_cmd[i],cntr_cmd_k[i],cntr_cmd_kk[i],
        #                                       rotor_vel_err[i],rotor_vel_err_k[i],rotor_vel_err_kk[i],
        #                                       act[i],act_k[i], act_kk[i],
        #                                       d_pos_err), axis=0).flatten().astype(np.float32)
        #               for i in range(self.nAgents)}

        curr_state = [np.concatenate((ph_err[i],ph_err_k[i], ph_err_kk[i],cntr_cmd[i],cntr_cmd_k[i],cntr_cmd_kk[i],rotor_vel_err[i],rotor_vel_err_k[i],rotor_vel_err_kk[i],act[i],act_k[i], act_kk[i],d_pos_err.flatten()), axis=0).flatten().astype(np.float32)
                      for i in range(self.nAgents)]

        curr_state = np.array(curr_state)
        return curr_state

    def get_reward(self):
        ## sparse reward
        pos_err = self.Pos_err
        phase_err = self.Phase_err
        phase_err = Fold2HalfPi(phase_err)
        stable = all(np.abs(pos_err) < 0.25)
        balance = all(np.abs(phase_err) < 10*0.0174) and all(np.abs(pos_err) < 0.1)
        sync = all(np.abs(phase_err) < 5*0.0174) and all(np.abs(pos_err) < 0.05)
        curr_reward = 0.
        if stable:
            if sync:
                curr_reward = 5.
            elif balance:
                curr_reward = 1.
            else:
                curr_reward = 0.

        else:
            curr_reward = -10.

        phase_reward = 5.0*np.sum(np.clip(0.25-np.array(phase_err), 0, None))
        pos_reward = np.sum(np.clip(0.05 - np.abs(np.array(pos_err)), 0, None))
        curr_reward += np.array(np.sum([phase_reward, pos_reward])).flatten()[0]

        return curr_reward

    def is_done(self, falling):
        if falling:
            return True
        elif any(np.abs(self.Pos_err) > 1.0):
            return True
        return False

    def fix_phase(self,phase_err):
        f_err = Fold2HalfPi(phase_err).flatten()
        manipulate = np.array(([1 if err < 1.57 else -1 for err in phase_err]))
        return manipulate*f_err