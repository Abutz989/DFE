
import gym
from gym import spaces
import numpy as np
from Physics_Models.ControlAlgTestMod import *
from Physics_Models.Drone_physics import PhysicsSim

class DroneEnv(gym.Env):
    metadata = {
        'render.modes': ['rgb_array']
    }
    def __init__(self, runtime=10., dt=0.001, seed=42, obs_type = 2,reward_type = 3):
        super(DroneEnv, self).__init__()
        self.damping_time = 4
        self.runtime = runtime + self.damping_time                                    # time limit of the episode
        self.dt = dt
        self.obs_type = obs_type
        self.reward_type = reward_type
        max_act =np.ones(4,)
        if self.obs_type == 1:
            max_obs = np.ones(14,)   # [Ph_err, Act, ang_pos_err,cntr_cmd] 
        elif self.obs_type == 2:
            max_obs = np.ones(22,)   # [Ph_err, pos_err,d_Ph_err, d_pos_err, ,cntr_cmd] 
        elif self.obs_type == 3:
            max_obs = np.ones(26,)   # [ph_err, act, cntr_cmd,rotor_vel_err, d_ph_err, d_act, d_cntr_cmd] 
        elif self.obs_type == 4:
            hidden_num = 6
            max_act = np.ones(4+hidden_num,)
            max_obs = np.ones(30+hidden_num,)   # [Ph_err, pos_err,d_Ph_err, d_pos_err, ,cntr_cmd ,act,rotor_vel_err ,hidden_layer] 
        elif self.obs_type == 5:
            max_act =np.ones(3,)
            max_obs = np.ones(21,)   # [ph_err ,ph_err_k ,ph_err_kk, ph_err_kkk, vel_err, vel_err_k ,vel_err_kk]
        else :
            max_obs = np.ones(38,)   # [ph_err, pos_err, act, cntr_cmd,rotor_vel_err, d_ph_err, d_pos_err, d_act, d_cntr_cmd] 
        
        
        self.action_space = spaces.Box(low=-max_act, high=max_act, dtype=np.float32)
        self.observation_space = spaces.Box(low=-max_obs, high=max_obs, dtype=np.float32)
        
        

    def reset(self, seed=None):
     
        init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose
        init_velocities = np.array([0., 0., 0.])         # initial velocities
        init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
        init_rotors_position = np.array([1., 2., 3., 4.])
        self.Plant = PhysicsSim(init_pose, init_velocities, init_angle_velocities,
                                init_rotors_position, self.runtime, self.dt)
        self.V_cmd = VelCommand(self.Plant,self.dt)
        self.P_cmd = MotorCommand(self.Plant,self.dt)
        self.Control_cmd =  np.zeros((4,))
        self.Pos_err = np.zeros((6,))

        self.Phase_err = np.zeros((3,))  # init_rotors_position[1:] - init_rotors_position[0]
        self.Phase_err_k = np.zeros((3,))
        self.Phase_err_kk = np.zeros((3,))
        self.Phase_err_kkk = np.zeros((3,))
        self.Rotors_Vel_err = np.zeros((3,))
        self.Rotors_Vel_err_k = np.zeros((3,))
        self.Rotors_Vel_err_kk = np.zeros((3,))

        self.d_Pos_err = Dis_Der(Var_len=len(self.Pos_err), h=self.dt, sat=10)
        self.d_Phase_err = Dis_Der(Var_len=len(self.Phase_err), h=self.dt, sat=10)
        self.LPF_phase_err  = LPF(Wn = 3, dt=self.dt, sat=4)
        
#         only for continuis action
        self.LPF_action  = LPF(Wn = 10, dt=self.dt, sat=4)
        self.d_action = Dis_Der(Var_len = 4, h=self.dt, sat=10)
        
        self.LPF_Control_cmd  = LPF(Wn = 1, dt=self.dt, sat=12)
        self.d_Control_cmd = Dis_Der(Var_len=len(self.Control_cmd), h=self.dt, sat=120)
        

        self.curr_episode_reward = np.array([0.])
        self.TelemReset()
        self.FirstStep()

        return self._get_obs(np.zeros(self.action_space.shape))

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

        K_action = 6.0
        if self.obs_type == 5:
            action = np.concatenate(([0],action)).flatten()
        action = K_action * action
        refPhase = 0
        State, rotor_position, time = self.Plant.get_state()
        ref_rotors_speed, refValuse = self.V_cmd.step(ref_cmd, State)
        Volt_Input_cmd, rotor_position_err,rotors_speed_error = self.P_cmd.step(refPhase, ref_rotors_speed)
        self.Control_cmd = Volt_Input_cmd
        
        LPF_action = action[:4]
        # LPF_action = self.LPF_action.culcFilt(action[:4])
        Volt_Input = np.clip(Volt_Input_cmd + LPF_action, -12, 12)
        # Volt_Input = np.clip(Volt_Input_cmd + action[:4], -12, 12)
        falling = self.Plant.next_timestep(Volt_Input)
        self.Telem.record(refValuse, rotor_position_err)
        
        self.Pos_err = ref_cmd - State
        self.d_Pos_err.culcDer(self.Pos_err)
        
        self.Phase_err_kkk = self.Phase_err_kk
        self.Phase_err_kk = self.Phase_err_k
        self.Phase_err_k = self.Phase_err
        self.Phase_err = rotor_position_err[1:]
        LPF_ph_err = self.LPF_phase_err.culcFilt(rotor_position_err[1:])
        # self.Phase_err = LPF_ph_err
        
        self.d_Phase_err.culcDer(LPF_ph_err)
        LPF_action = self.LPF_action.culcFilt(action[:4]/K_action)
        self.d_action.culcDer(LPF_action)
        LPF_Control_cmd = self.LPF_Control_cmd.culcFilt(self.Control_cmd)
        self.d_Control_cmd.culcDer(LPF_Control_cmd)
        
        self.Rotors_Vel_err_kk = self.Rotors_Vel_err_k
        self.Rotors_Vel_err_k = self.Rotors_Vel_err
        self.Rotors_Vel_err = rotors_speed_error[1:]
        
        obs = self._get_obs(action/K_action)
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

        act = np.copy(action).reshape(-1, 1) /1.0
        d_act = np.copy(self.d_action.du).reshape(-1, 1)/10.0
        ph_err = np.copy(self.Phase_err).reshape(-1, 1) / 3.142
        d_ph_err = np.copy(self.d_Phase_err.du).reshape(-1, 1) / 10.0
        pos_err = np.copy(self.Pos_err).reshape(-1, 1) / 1.0
        d_pos_err = np.copy(self.d_Pos_err.du).reshape(-1, 1) / 10.0
        cntr_cmd = np.copy(self.Control_cmd).reshape(-1, 1) / 12.0
        d_cntr_cmd = np.copy(self.d_Control_cmd.du).reshape(-1, 1) / 120.0
        rotor_vel_err = np.clip(np.copy(self.Rotors_Vel_err).reshape(-1, 1),-30,30) / 30
        
        if self.obs_type == 1:# [Ph_err, Act, ang_pos_err,cntr_cmd] 
            curr_state = np.concatenate((ph_err, act[:4], pos_err[3:], cntr_cmd), axis=0).flatten().astype(np.float32)   
        elif self.obs_type == 2: # [Ph_err, pos_err,d_Ph_err, d_pos_err, ,cntr_cmd] 
            curr_state = np.concatenate((ph_err, pos_err, d_ph_err, d_pos_err,cntr_cmd), axis=0).flatten().astype(np.float32)   
        elif self.obs_type == 3: # [ph_err, act, cntr_cmd, rotor_vel_err, d_ph_err, d_act, d_cntr_cmd] 
            curr_state = np.concatenate((ph_err, act[:4], cntr_cmd,rotor_vel_err, d_ph_err, d_act, d_cntr_cmd), axis=0).flatten().astype(np.float32)   
        elif self.obs_type == 4: # [Ph_err, pos_err,d_Ph_err, d_pos_err, ,cntr_cmd] 
            curr_state = np.concatenate((ph_err, pos_err, d_ph_err, d_pos_err,cntr_cmd,rotor_vel_err,act), axis=0).flatten().astype(np.float32)   
        elif self.obs_type == 5:
            ph_err = np.copy(self.fix_phase(self.Phase_err)).reshape(-1, 1) / 3.142/2
            ph_err_k = np.copy(self.fix_phase(self.Phase_err_k)).reshape(-1, 1) / 3.142/2
            ph_err_kk = np.copy(self.fix_phase(self.Phase_err_kk)).reshape(-1, 1) / 3.142/2
            ph_err_kkk = np.copy(self.fix_phase(self.Phase_err_kkk)).reshape(-1, 1) / 3.142/2
            rotor_vel_err = np.clip(np.copy(self.Rotors_Vel_err).reshape(-1, 1), -30, 30) / 30
            rotor_vel_err_k = np.clip(np.copy(self.Rotors_Vel_err_k).reshape(-1, 1), -30, 30) / 30
            rotor_vel_err_kk = np.clip(np.copy(self.Rotors_Vel_err_kk).reshape(-1, 1), -30, 30) / 30
            curr_state = np.concatenate((ph_err, ph_err_k, ph_err_kk, ph_err_kkk,rotor_vel_err, rotor_vel_err_k, rotor_vel_err_kk), axis=0).flatten().astype(np.float32)
        else:
            curr_state = np.concatenate((ph_err, pos_err, act, cntr_cmd,rotor_vel_err, d_ph_err, d_pos_err, d_act, d_cntr_cmd), axis=0).flatten().astype(np.float32)   
        
        
        return curr_state

    def get_reward(self):
        reward_type = self.reward_type
        
        if reward_type == 1:
            # original
            pos_err = self.Pos_err[:3]
            rotor_position_err = self.Phase_err
            K_out_pos = 10.0
            K_out_phase = 2.0
            K_time = 1.0
            pos_tol = 5 * 0.01  # 5 [cm]
            phase_tol = 5 * 0.0174  # 5 [deg]
            loc_err_tol = 0.5  # 0.5 [m]
            rad_err_tol = 10*0.0175  # [rad]
            rotor_position_err = Fold2HalfPi(rotor_position_err)
            phase_reward = np.sum(K_out_phase * (rad_err_tol - np.array(rotor_position_err)))

            if any(np.abs(rotor_position_err) > rad_err_tol):
                phase_reward += 2.0 * (rad_err_tol - np.max(np.abs(rotor_position_err)))
            elif all(np.abs(rotor_position_err) < phase_tol):
                phase_reward += 200.0 * (phase_tol -np.max(np.abs(rotor_position_err)))

            pos_reward = np.sum(K_out_pos * np.abs(np.array(pos_err)))
            if any(np.abs(pos_err) > loc_err_tol):
                pos_reward += 50.0 * (loc_err_tol - np.max(np.abs(pos_err)))

            curr_reward = np.array(np.sum([phase_reward, pos_reward])).flatten()[0]

        elif reward_type == 2:
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
        
        elif reward_type == 3:
            pos_err = self.Pos_err
            phase_err = self.Phase_err
            rotor_vel_err = self.Rotors_Vel_err
            phase_err = Fold2HalfPi(phase_err)
            sync = all(np.abs(phase_err) < 5*0.0174) and all(np.abs(pos_err) < 0.05)
            stable = all(np.abs(pos_err) < 0.25)
            
            pos_reward = np.sum(np.clip(0.05 - np.abs(np.array(pos_err)), 0, None))/6.0
            phase_reward = 5.0*np.sum(np.clip(10*0.0174 - np.array(phase_err), 0, None))
            rotor_vel_reward = 0.25*np.sum(np.clip(4.0-np.abs(rotor_vel_err), 0, None))/4.0
            bonus1 = 10.0 if sync else 0.0
            bonus2 = np.sum(np.array(phase_err < 10*0.0174).astype(float))
            penlty = 0.0 if stable else -10.0
            
            curr_reward = np.array(np.sum([phase_reward, pos_reward, rotor_vel_reward, bonus1, bonus2, penlty])).flatten()[0]
            
        else:
            curr_reward = 0.0
        
        return curr_reward

    def is_done(self, falling):
        if falling:
            return True
        elif any(np.abs(self.Pos_err) > 1.0):
            return True
        return False
    
    def fix_phase(self,phase_err):
        f_err = Fold2HalfPi(phase_err).flatten()
        manipoate = np.array(([1 if err < 1.57 else -1 for err in phase_err]))
        return manipoate*f_err