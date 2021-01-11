import numpy as np
from math import sin, cos


def trajGenerator(t, option, pos=[0, 0, 0]):

    if option is "fixed position":
        des_pos = np.array([pos]).T
        des_vel = np.array([[0, 0, 0]]).T
        des_acc = np.array([[0, 0, 0]]).T

    elif option is "spiral":
        des_pos = np.array([[0.4 * t, 0.4 * sin(np.pi * t), 0.6 * cos(np.pi * t)]]).T
        des_vel = np.array(
            [[0.4, 0.4 * np.pi * cos(np.pi * t), -0.6 * np.pi * sin(np.pi * t)]]
        ).T
        des_acc = np.array(
            [
                [
                    0,
                    -0.4 * np.pi * np.pi * sin(np.pi * t),
                    -0.6 * np.pi * np.pi * cos(np.pi * t),
                ]
            ]
        ).T

    elif option is "circle": 
        # circle
    
        des_pos = np.array([[5 * sin(np.pi * t), 5* cos(np.pi * t), 3]]).T
        des_vel = np.array(
            [[5 * np.pi * cos(np.pi * t), -5 * np.pi * sin(np.pi * t), 0]]
        ).T
        des_acc = np.array(
            [
                [
                    -5 * np.pi * np.pi * sin(np.pi * t),           
                    -5 * np.pi * np.pi * cos(np.pi * t),
                    0

                ]
            ]
        ).T

    else:
        # default setpoint
        des_pos = np.array([[0, 0, 5]]).T
        des_vel = np.array([[0, 0, 0]]).T
        des_acc = np.array([[0, 0, 0]]).T

    return des_pos, des_vel, des_acc


if __name__ == "__main__":
    pass
