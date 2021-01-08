#!/usr/bin/env python
# ROS python API
import rospy

# 3D point & Stamped Pose msgs
from geometry_msgs.msg import Point, PoseStamped
from mavros_msgs.msg import AttitudeTarget

# import all mavros messages and services
from mavros_msgs.msg import *
from mavros_msgs.srv import *

# Flight modes class
# Flight modes are activated using ROS services
class fcuModes:
    def __init__(self):
        pass

    def setTakeoff(self):
        rospy.wait_for_service("mavros/cmd/takeoff")
        try:
            takeoffService = rospy.ServiceProxy(
                "mavros/cmd/takeoff", mavros_msgs.srv.CommandTOL
            )
            takeoffService(altitude=3)
        except rospy.ServiceException, e:
            print "Service takeoff call failed: %s" % e

    def setArm(self):
        rospy.wait_for_service("mavros/cmd/arming")
        try:
            armService = rospy.ServiceProxy(
                "mavros/cmd/arming", mavros_msgs.srv.CommandBool
            )
            armService(True)
        except rospy.ServiceException, e:
            print "Service arming call failed: %s" % e

    def setDisarm(self):
        rospy.wait_for_service("mavros/cmd/arming")
        try:
            armService = rospy.ServiceProxy(
                "mavros/cmd/arming", mavros_msgs.srv.CommandBool
            )
            armService(False)
        except rospy.ServiceException, e:
            print "Service disarming call failed: %s" % e

    def setStabilizedMode(self):
        rospy.wait_for_service("mavros/set_mode")
        try:
            flightModeService = rospy.ServiceProxy(
                "mavros/set_mode", mavros_msgs.srv.SetMode
            )
            flightModeService(custom_mode="STABILIZED")
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Stabilized Mode could not be set." % e

    def setOffboardMode(self):
        rospy.wait_for_service("mavros/set_mode")
        try:
            flightModeService = rospy.ServiceProxy(
                "mavros/set_mode", mavros_msgs.srv.SetMode
            )
            flightModeService(custom_mode="OFFBOARD")
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Offboard Mode could not be set." % e

    def setAltitudeMode(self):
        rospy.wait_for_service("mavros/set_mode")
        try:
            flightModeService = rospy.ServiceProxy(
                "mavros/set_mode", mavros_msgs.srv.SetMode
            )
            flightModeService(custom_mode="ALTCTL")
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Altitude Mode could not be set." % e

    def setPositionMode(self):
        rospy.wait_for_service("mavros/set_mode")
        try:
            flightModeService = rospy.ServiceProxy(
                "mavros/set_mode", mavros_msgs.srv.SetMode
            )
            flightModeService(custom_mode="POSCTL")
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Position Mode could not be set." % e

    def setAutoLandMode(self):
        rospy.wait_for_service("mavros/set_mode")
        try:
            flightModeService = rospy.ServiceProxy(
                "mavros/set_mode", mavros_msgs.srv.SetMode
            )
            flightModeService(custom_mode="AUTO.LAND")
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Autoland Mode could not be set." % e


class Controller:
    # initialization method
    def __init__(self):
        # Drone state
        self.state = State()
        # Instantiate a setpoints message
        self.sp = PositionTarget()
        # Instantiate a attitude message
        self.at = AttitudeTarget()
        # set the flag to use position setpoints and yaw angle
        self.sp.type_mask = int("010111111000", 2)
        # LOCAL_NED
        self.sp.coordinate_frame = 1

        # We will fly at a fixed altitude for now
        # Altitude setpoint, [meters]
        self.ALT_SP = 3.00
        # update the setpoint message with the required altitude
        self.sp.position.z = self.ALT_SP
        # Step size for position update
        self.STEP_SIZE = 2.0
        # Fence. We will assume a square fence for now
        self.FENCE_LIMIT = 5.0

        # A Message for the current local position of the drone
        self.local_pos = Point(0.0, 0.0, 0.0)

        # initial values for setpoints
        self.sp.position.x = 0.0
        self.sp.position.y = 0.0

        # speed of the drone is set using MPC_XY_CRUISE parameter in MAVLink
        # using QGroundControl. By default it is 5 m/s.

    # Callbacks

    ## command callback
    def cmdCb(self, msg):
        self.at.header.stamp = msg.header.stamp
        self.at.header.frame_id = msg.header.frame_id
        self.at.body_rate.x = msg.body_rate.x
        self.at.body_rate.y = msg.body_rate.y
        self.at.body_rate.z = msg.body_rate.z
        self.at.type_mask = msg.type_mask
        self.at.thrust = msg.thrust

    ## local position callback
    def posCb(self, msg):
        self.local_pos.x = msg.pose.position.x
        self.local_pos.y = msg.pose.position.y
        self.local_pos.z = msg.pose.position.z

    ## Drone State callback
    def stateCb(self, msg):
        self.state = msg

    ## Update setpoint message
    def updateSp(self):
        self.sp.position.x = self.local_pos.x
        self.sp.position.y = self.local_pos.y
        self.sp.position.z = self.local_pos.z

    def x_dir(self):
        self.sp.position.x = self.local_pos.x + 5
        self.sp.position.y = self.local_pos.y

    def neg_x_dir(self):
        self.sp.position.x = self.local_pos.x - 5
        self.sp.position.y = self.local_pos.y

    def y_dir(self):
        self.sp.position.x = self.local_pos.x
        self.sp.position.y = self.local_pos.y + 5

    def neg_y_dir(self):
        self.sp.position.x = self.local_pos.x
        self.sp.position.y = self.local_pos.y - 5


# Main function
def main():

    # initiate node
    rospy.init_node("setpoint_node", anonymous=True)

    # flight mode object
    modes = fcuModes()

    # controller object
    cnt = Controller()

    # ROS loop rate
    rate = rospy.Rate(20.0)

    # Subscribe to drone state
    rospy.Subscriber("mavros/state", State, cnt.stateCb)

    # Subscribe to drone's local position
    rospy.Subscriber("mavros/local_position/pose", PoseStamped, cnt.posCb)

    # Subscribe to control of geometric controller
    rospy.Subscriber("computed_cmd", AttitudeTarget, cnt.cmdCb)

    # attitude publisher
    at_pub = rospy.Publisher("mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=1)

    # wait for FCU connection
    while not rospy.is_shutdown() and not cnt.state.connected:
        print ("wait for FCU...")
        rate.sleep()

    # We need to send few setpoint messages, then activate OFFBOARD mode, to take effect
    k = 0
    print ("send few commands")
    while k < 10:
        sp_pub.publish(cnt.sp)
        rate.sleep()
        k = k + 1

    # activate OFFBOARD mode
    print ("set to OFFBOARD mode")
    modes.setOffboardMode()

    # Make sure the drone is armed
    while not cnt.state.armed:
        print ("ready to arm")
        modes.setArm()
        rate.sleep()

    # ROS main loop
    if not rospy.is_shutdown():
        print ("start main task...")
    while not rospy.is_shutdown():
        # update and publish
        at_pub.publish(cnt.at)
        rate.sleep()

    print ("finished")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
