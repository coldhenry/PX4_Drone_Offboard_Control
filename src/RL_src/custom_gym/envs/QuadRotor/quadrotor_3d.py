# no waypoint #=============================
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import numpy as np
from math import cos, sin
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from .custom_Quadrotor import CustomQuadrotor
from PythonRobotics.AerialNavigation.drone_3d_trajectory_following.TrajectoryGenerator import TrajectoryGenerator
from mpl_toolkits.mplot3d import Axes3D

"""
Based on PythonRobotics/AerialNavigation/drone_3d_trajectory_following.
Extended to OpenAI Gym Interface.

[Ref]
https://github.com/AtsushiSakai/PythonRobotics/blob/master/AerialNavigation/drone_3d_trajectory_following/drone_3d_trajectory_following.py
"""

# Simulation parameters
g = 9.81  # gravity (m/s^2)
m = 0.65  # mass (kg)
L = 0.23  # distance from the motor to the center body (m)

# intertia [jx, jy, jz] (kg.m^-2)
jx = 7.5e-1
jy = 7.5e-1
jz = 1.3

kt = 31.3e-2   # drag force coefficient (N.s2)
kd = 7.5e-4    # reverse moment coefficient (N.ms2)


T = 4 # simulation time



"""
References :
[1]Backstepping Sliding Approach for Controlling a Quadrotor Using Barrier Lyapunov Functions
"""
class QuadRotorEnv(gym.Env):
    
    def __init__(self):
        """
        Calculates the necessary thrust and torques for the quadrotor to
        follow the trajectory described by the sets of coefficients
        x_c, y_c, and z_c.
        """
        self.x_pos = 0
        self.y_pos = 0
        self.z_pos = 0
        self.x_vel = 0
        self.y_vel = 0
        self.z_vel = 0
#         self.x_acc = 0
#         self.y_acc = 0
#         self.z_acc = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.roll_vel = 0
        self.pitch_vel = 0
        self.yaw_vel = 0        
#         self.des_yaw = 0
        
        self.dt = 0.1
#         self.dt = 0.001
        
        self.q = CustomQuadrotor(x=self.x_pos, y=self.y_pos, z=self.z_pos,\
                roll=self.roll,pitch=self.pitch, yaw=self.yaw,\
                size=1)

        self.plot_first_called = True
        self.record = False
        self.episode_id = -1 #episode becomes 0 when reset() is called for the first time.
        
        low = np.full(9, -float('inf'), dtype=np.float32)
        high = np.full(9, float('inf'), dtype=np.float32)

        low_a = np.full(4, 0, dtype=np.float32)
        high_a = np.full(4, 50, dtype=np.float32)
        

        self.observation_space= spaces.Box(low, high, dtype=np.float32)
        self.action_space= spaces.Box(low_a, high_a, dtype=np.float32)
        
        
    def custom_Monitor(self,save_folder, video_callable = False, video_file_name = "episode"):
        self.video_callable = video_callable
        self.video_file_name = save_folder + "/" + video_file_name
        os.makedirs(save_folder, exist_ok = True)
        self.record = True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def state_plot(self):
        return [self.x_pos, self.y_pos, self.z_pos]

    def step(self, action):
        #action: speed of each rotor.
        
        
        action = np.clip(action, 0.0, 100)
        acc = action
        w1 = acc[0]
        w2 = acc[1]
        w3 = acc[2]
        w4 = acc[3]
        
        omeg = w2+w4-w1-w3
        
        u1 = kt*(w1**2 + w2**2 + w3**2 + w4**2)
        u2 = kt*(w1**2 - w3**2)
        u3 = kt*(w2**2 - w4**2)
        u4 = kd*(w1**2 + w3**2 - w2**2 - w4**2)
        
        
        self.x_vel += ((cos(self.roll)*sin(self.pitch)*cos(self.yaw)+sin(self.roll)*sin(self.yaw))*u1/m)*self.dt
        self.y_vel += ((cos(self.roll)*sin(self.pitch)*sin(self.yaw)-sin(self.roll)*cos(self.yaw))*u1/m)*self.dt
        self.z_vel += (-g+(cos(self.roll)*cos(self.pitch))*u1/m)* self.dt

        self.x_pos += self.x_vel * self.dt
        self.y_pos += self.y_vel * self.dt
        self.z_pos += self.z_vel * self.dt
        
        self.traj()
        
        roll_torque = self.yaw_vel*self.pitch_vel*(jy-jz)/ jx + L*u2/jx
        pitch_torque = self.yaw_vel*self.roll_vel*(jz-jx)/ jy + L*u3/jy
        yaw_torque = self.pitch_vel*self.roll_vel*(jx-jy)/ jz + L*u4/jz

        self.roll_vel += roll_torque * self.dt
        self.pitch_vel += pitch_torque * self.dt
        self.yaw_vel += yaw_torque * self.dt

        self.roll += self.roll_vel * self.dt
        self.pitch += self.pitch_vel * self.dt
        self.yaw += self.yaw_vel * self.dt
        
       
        self.q.update_pose(self.x_pos, self.y_pos, self.z_pos, \
                self.roll, self.pitch, self.yaw)

        self.t += self.dt
 
        if self.t< T: 
            done = False
        else:
            done = True
        if self.record_episode:
            self._save_action(self.x_pos, self.y_pos, self.z_pos, \
                self.roll, self.pitch, self.yaw)

        return self._get_obs(), self._get_reward(done), done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    
    def reset(self):
        self.episode_id += 1
        self.t = 0
        self.check_point = 0

        bound = 0.01 
#================================================================ random initial point
        self.x_pos = 0 + self.np_random.uniform(-bound, bound)
        self.y_pos = 0 + self.np_random.uniform(-bound, bound)
        self.z_pos = 0 + self.np_random.uniform(-bound, bound)
#         self.x_vel = 0 + self.np_random.uniform(-bound, bound)
#         self.y_vel = 0 + self.np_random.uniform(-bound, bound)
#         self.z_vel = 0 + self.np_random.uniform(-bound, bound)
#         self.roll = 0 + self.np_random.uniform(-bound, bound)
#         self.pitch = 0 + self.np_random.uniform(-bound, bound)
#         self.yaw = 0 + self.np_random.uniform(-bound, bound)
#================================================================ without random
#         self.x_pos = 0 
#         self.y_pos = 0 
#         self.z_pos = 0 
        self.x_vel = 0 
        self.y_vel = 0 
        self.z_vel = 0 
        self.roll = 0 
        self.pitch = 0 
        self.yaw = 0 
#================================================================
        self.roll_vel = 0
        self.pitch_vel = 0
        self.yaw_vel = 0
        
        self.traj()

        self.q.reset_trajectory_plot()
        self.q.update_pose(self.x_pos, self.y_pos, self.z_pos, \
                self.roll, self.pitch, self.yaw)

        if self.record:
            self.record_episode = self._ini_movie(self.episode_id)
            if self.record_episode:
                self._save_action(self.x_pos, self.y_pos, self.z_pos,\
                        self.roll, self.pitch, self.yaw)
        else:
            self.record_episode = False


        return self._get_obs()
   
    
    def render(self, mode = 'human', close = True):
        if self.plot_first_called:
            self.q.initialize_rendering()
            self.plot_first_called = False
    
        self.q.plot()
        

    
    def close(self):
        pass


    """
    _get_obs() and _get_reward() are needed to be modified
    depending on applying tasks.
    """

    def _get_obs(self):
        # 9obs
        obs = np.array([(self.des_x_pos-self.x_pos), (self.des_y_pos-self.y_pos), (self.des_z_pos-self.z_pos), \
                        self.roll, self.pitch, self.yaw, \
                        (self.des_x_vel - self.x_vel), (self.des_y_vel - self.y_vel),(self.des_z_vel - self.z_vel)])
        return  obs
   
    
    def _get_reward(self, done):

        r = (-abs(self.des_x_pos-self.x_pos)-abs(self.des_y_pos-self.y_pos)-abs(self.des_z_pos-self.z_pos))-abs(self.yaw)-abs(self.roll)-abs(self.pitch) 
        
#         r = (-abs(self.des_x_pos-self.x_pos)-abs(self.des_y_pos-self.y_pos)-abs(self.des_z_pos-self.z_pos))-abs(self.yaw)-abs(self.roll)-abs(self.pitch)- 0.5*(abs(self.des_x_vel - self.x_vel)+abs(self.des_y_vel - self.y_vel)+abs(self.des_z_vel - self.z_vel))
        
        if done:
            return 0.0
        else:
            return r
            
    def _ini_movie(self,episode_id):
        if not self.video_callable(episode_id):
            return False
        self.act_history = []
        return True


    def _save_action(self,x_pos, y_pos, z_pos, roll, pitch, yaw):
        self.act_history.append((x_pos,y_pos,z_pos, roll,pitch,yaw))


    def _save_movie(self, episode_id):
        act_history = np.asarray(self.act_history)

        FFMpegWriter = manimation.writers['ffmpeg']
        moviewriter = FFMpegWriter(fps = 60)
        file_name = self.video_file_name + str(episode_id) + ".mp4"

        if self.plot_first_called:
            self.q.initialize_rendering()
            self.plot_first_called = False
        
        self.q.reset_trajectory_plot()

        with moviewriter.saving(self.q.fig,file_name,dpi = 100):
            for act in act_history:
                self.q.update_pose(act[0], act[1], act[2],\
                        act[3], act[4], act[5])
                self.q.plot()
                moviewriter.grab_frame()
        
        print("--Episode{} has recorded.--".format(str(episode_id)))

    def traj(self):
        "Define desired trajectories"

        # takeoff
        self.des_x_pos = 0
        self.des_y_pos = 0
        self.des_z_pos = 2*self.t
        self.des_x_vel = 0
        self.des_y_vel = 0
        self.des_z_vel = 2 
        self.des_x_acc = 0
        self.des_y_acc = 0
        self.des_z_acc = 0


#         # circle
#         self.des_x_pos = 5 * sin(0.2*np.pi*self.t)    
#         self.des_y_pos = 5 * cos(0.2*np.pi*self.t)
#         self.des_z_pos = 4
#         self.des_x_vel = 1 *np.pi * cos(0.2*np.pi*self.t)
#         self.des_y_vel = -1* np.pi*sin(0.2*np.pi*self.t)
#         self.des_z_vel = 0
#         self.des_x_acc = -0.2*np.pi*np.pi*sin(0.2*np.pi*self.t)
#         self.des_y_acc = -0.2*np.pi*np.pi*sin(0.2*np.pi*self.t)
#         self.des_z_acc = 0


#         # sin wave in x direction 
#         self.des_x_pos = self.t
#         self.des_y_pos = 1
#         self.des_z_pos = 1* sin(0.5* np.pi* self.t)+ 2
#         self.des_x_vel = 1
#         self.des_y_vel = 0
#         self.des_z_vel = 0.5* np.pi * cos(0.5* np.pi*self.t)
#         self.des_x_acc = 0
#         self.des_y_acc = 0
#         self.des_z_acc = -0.5*np.pi*np.pi*sin(0.5* np.pi*self.t)

            
