#!/usr/bin/env python
import rospy
import csv
from std_msgs.msg import String
from mavros_msgs.msg import AttitudeTarget

record =[]

def att_cb(data):

    msgs = [data.body_rate.x, data.body_rate.y, data.body_rate.z, data.thrust]
    rospy.loginfo(rospy.get_caller_id() + " {}, {}, {}".format(msgs[0], msgs[1], msgs[2]))
    rospy.loginfo(rospy.get_caller_id() + " %.3f", msgs[3])
    record.append(msgs)
    
def listener():

    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/mavros/setpoint_raw/attitude", AttitudeTarget, att_cb)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
    """
    with open('/home/coldhenry/src/Drone/testing/geometric_sinewave_log.csv', 'w') as file:
        print("writing")
        writer = csv.writer(file)
        writer.writerows(record)
    print("Writed")
    """