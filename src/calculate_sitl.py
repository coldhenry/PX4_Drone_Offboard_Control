#!/usr/bin/env python
import argparse
import rospy
import math
from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from matplotlib import animation
from std_msgs.msg import String
from mavros_msgs.msg import AttitudeTarget
from geometry_msgs.msg import PoseStamped, TwistStamped  ## ???

from geometric_controller import geometric_controller as geo_ctl


class callback:
    def __init__(self):

        ### Sub / Pub
        rospy.Subscriber("reference/pose", PoseStamped, self.setpointPose_cb)
        rospy.Subscriber(
            "/mavros/local_position/pose", PoseStamped, self.feedbackPose_cb
        )
        rospy.Subscriber(
            "mavros/local_position/velocity_local", TwistStamped, self.mavtwist_cb
        )
        self.pubCmd = rospy.Publisher("computed_cmd", AttitudeTarget, queue_size=10)

        ## Main loop
        self.timer = rospy.Timer(rospy.Duration(0.1), self.mainLoop)

        ## Variables list
        self.time = 0.0
        self.state = np.zeros((18,))
        self.x, self.y, self.z = 0.0, 0.0, 0.0
        self.qx, self.qy, self.qz, self.qw = 0.0, 0.0, 0.0, 0.0
        self.x_d, self.y_d, self.z_d = 0.0, 0.0, 0.0
        self.qx_d, self.qy_d, self.qz_d, self.qw_d = 0.0, 0.0, 0.0, 0.0
        self.vx, self.vy, self.vz = 0.0, 0.0, 0.0
        self.vx_d, self.vy_d, self.vz_d = 0.0, 0.0, 0.0
        self.roll_v, self.pitch_v, self.yaw_v = 0.0, 0.0, 0.0
        self.rotmat = Rot.from_dcm(np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]))
        self.rotmat_d = np.zeros((3, 3))
        self.euler = np.zeros((3,))
        self.euler_d = np.zeros((3,))
        self.roll, self.pitch, self.yaw = 0.0, 0.0, 0.0
        self.roll_d, self.pitch_d, self.yaw_d = 0.0, 0.0, 0.0

    def mainLoop(self, timer):
        
        # PD controller
        # self.getError()
        # self.PDcontroller()
        

        # geometric controller
        #t = np.linspace(self.time, self.time + 0.1, 1)
        t = self.time
        rotmat = np.array(self.rotmat.as_dcm())
        self.state = np.concatenate((self.x, self.y, self.z, self.vx, self.vy, self.vz, rotmat, self.roll_v, self.pitch_v, self.yaw_v), axis=None)
        #print(self.state)
        f, M, _, _, _, _ = geo_ctl(self.state, t)
        print("f: {f}\n M: {M}\n".format(f = f, M = M))
        self.time += 0.1
        self.pubRLcmd(f, M)

    def feedbackPose_cb(self, data):
        self.x, self.y, self.z = (
            data.pose.position.x,
            data.pose.position.y,
            data.pose.position.z,
        )
        self.qx, self.qy, self.qz, self.qw = (
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z,
            data.pose.orientation.w,
        )
        
        self.rotmat = Rot.from_quat([self.qx, self.qy, self.qz, self.qw])
        self.euler = self.rotmat.as_euler("zxy", degrees=False)  # changed here
        # print('euler',self.euler)
        # self.euler = self.euler % (2 * np.pi)
        self.roll, self.pitch, self.yaw = self.euler[1], self.euler[0], self.euler[2]

    def setpointPose_cb(self, data):
        self.x_d, self.y_d, self.z_d = (
            data.pose.position.x,
            data.pose.position.y,
            data.pose.position.z,
        )
        self.qx_d, self.qy_d, self.qz_d, self.qw_d = (
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z,
            data.pose.orientation.w,
        )
        self.rotmat_d = Rot.from_quat([self.qx_d, self.qy_d, self.qz_d, self.qw_d])
        self.euler_d = self.rotmat_d.as_euler("zxy", degrees=False)
        # self.roll_d, self.pitch_d, self.yaw_d = (
        #     self.euler_d[1],
        #     self.euler_d[0],
        #     self.euler_d[2],
        # )

        self.roll_d, self.pitch_d, self.yaw_d = (
            0.0,
            0.0,
            0.0,
        )

    def mavtwist_cb(self, data):
        self.vx = data.twist.linear.x
        self.vy = data.twist.linear.y
        self.vz = data.twist.linear.z
        self.roll_v = data.twist.angular.x
        self.pitch_v = data.twist.angular.y
        self.yaw_v = data.twist.angular.z

    def ref_cb(self, data):
        self.vx_d, self.vy_d, self.vz_d = (
            data.velocity.x,
            data.velocity.y,
            data.velocity.z,
        )

    def getError(self):
        """
        observation
        type: tensor
        name: [err_x, err_y, err_z, err_roll, err_pitch, err_yaw, err_vx, err_vy, err_vz]
        shape: [1,9]

        """
        self.err_x = self.x_d - self.x
        self.err_y = self.y_d - self.y
        self.err_z = self.z_d - self.z
        self.err_roll = float(self.roll_d - self.roll)
        self.err_pitch = float(self.pitch_d - self.pitch)
        self.err_yaw = float(self.yaw_d - self.yaw)
        # print('des',self.roll_d, self.pitch_d, self.yaw_d)
        # print('attitude',self.roll,self.pitch,self.yaw)
        self.err_vx = self.vx_d - self.vx
        self.err_vy = self.vy_d - self.vy
        self.err_vz = self.vz_d - self.vz
        npdata = np.array(
            [
                self.err_x,
                self.err_y,
                self.err_z,
                self.roll,
                self.pitch,
                self.yaw,
                self.err_vx,
                self.err_vy,
                self.err_vz,
            ]
        )
        return npdata

    def PDcontroller(self):
        g = 9.81
        m = 1.56  # 0.03   # 1.56
        jx = 0.029125  # 1.43e-5 #
        jy = 0.029125  # 1.43e-5 #0.029125
        jz = 0.05522  # 2.89e-5 

        # position controller params
        Kp = np.array([[15, 15, 30]]) * 1
        Kd = np.array([[12, 12, 10]]) * 1
        # attitude controller params
        KpM = np.ones((3, 1)) * 3000
        KdM = np.ones((3, 1)) * 300

        acc_des1 = 0 + Kd[0][0] * self.err_vx + Kp[0][0] * self.err_x
        acc_des2 = 0 + Kd[0][1] * self.err_vy + Kp[0][1] * self.err_y
        acc_des3 = 0 + Kd[0][2] * self.err_vz + Kp[0][2] * self.err_z

        # Desired roll, pitch and yaw
        phi_des = (1 / g) * (acc_des1 * sin(self.yaw_d)) - acc_des2 * cos(self.yaw_d)
        theta_des = (1 / g) * (acc_des1 * cos(self.yaw_d)) - acc_des2 * sin(self.yaw_d)
        psi_des = self.yaw_d

        # Thurst
        self.F = m * (g + acc_des3)
        #print("F:", self.F)
        # self.F = np.clip(self.F, 0.0, 100000.0)
        # Moment

        self.M1 = jx * KdM[0][0] * (self.roll_v - 0) + KpM[0][0] * (phi_des - self.roll)
        self.M2 = jy * KdM[1][0] * (self.pitch_v - 0) + KpM[1][0] * (
            theta_des - self.pitch
        )
        self.M3 = jz * KdM[2][0] * (self.yaw_v - 0) + KpM[2][0] * (psi_des - self.yaw)
        # print('M',self.M1,self.M2, self.M3)
        # print('RPY_v',self.roll_v,self.pitch_v,self.yaw_v)
        # print('des',phi_des,theta_des,psi_des)
        # print('RPY',self.roll,self.pitch,self.yaw)
        self.M1 = 0  # phi_des
        self.M2 = 0  # theta_des
        self.M3 = 0  # psi_des #self.F/m -g
        # self.M1 = np.clip(self.M1, 0.0, 100000.0)
        # self.M2 = np.clip(self.M2, 0.0, 100000.0)
        # self.M3 = np.clip(self.M3, 0.0, 100000.0)
        # print('M',self.M1,self.M2, self.M3)
        # pubPDcmd [F, M1, M2, M3]

    def pubRLcmd(self, f, M):

        msg = AttitudeTarget()

        # self.cmdConversion()

        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        # ============================PD controller
        # msg.body_rate.x = self.M1
        # msg.body_rate.y = self.M2
        # msg.body_rate.z = self.M3
        # msg.type_mask = 128
        # # Ignore orientation messages
        # msg.thrust = self.F
        # ============================PD controller

        # ============================RL controller
        # action = self.action.numpy()[0, 0, 0]
        # # action = np.clip(action, 0.0, 10.0)
        # print("action", action)
        # msg.body_rate.x = action[1] * 1e-3
        # msg.body_rate.y = action[2] * 1e-3
        # msg.body_rate.z = action[3] * 1e-3
        # msg.type_mask = 128
        # msg.thrust = action[0] * 2
        # # Ignore orientation messages
        # # if action[0] == 0:
        # #     msg.thrust = action[0] + 10
        # # else :
        # #     msg.thrust = action[0]
        # print("F", msg.thrust)
        # ============================RL controller

        # ============================Geo controller
        msg.body_rate.x = M[0]
        msg.body_rate.y = M[1]
        msg.body_rate.z = M[2]
        msg.type_mask = 128
        msg.thrust = f
        print("F", msg.thrust)
        # ============================RL controller

        self.pubCmd.publish(msg)


def main():

    rospy.init_node("calculate", anonymous=True)
    rospy.loginfo("Started!")

    cb = callback()

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == "__main__":

    main()
