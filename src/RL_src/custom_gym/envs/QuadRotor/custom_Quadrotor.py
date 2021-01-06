import numpy as np
import matplotlib.pyplot as plt
from PythonRobotics.AerialNavigation.drone_3d_trajectory_following.Quadrotor import Quadrotor

"""
Quadrotor Inheritance class for OpenAI Gym Interface.
 """

class CustomQuadrotor(Quadrotor):
    
    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, size=0.25):
            self.show_animation = False #always set false.
        
            self.p1 = np.array([size / 2, 0, 0, 1]).T
            self.p2 = np.array([-size / 2, 0, 0, 1]).T
            self.p3 = np.array([0, size / 2, 0, 1]).T
            self.p4 = np.array([0, -size / 2, 0, 1]).T
   
            self.x_data = []
            self.y_data = []
            self.z_data = []

            self.update_pose(x, y, z, roll, pitch, yaw)


    def initialize_rendering(self):
        plt.ion()
        self.fig = plt.figure()
        # for stopping simulation with the esc key.
        self.fig.canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        
        self.ax = self.fig.add_subplot(111, projection='3d')

    def reset_trajectory_plot(self):
        self.x_data = []
        self.y_data = []
        self.z_data = []