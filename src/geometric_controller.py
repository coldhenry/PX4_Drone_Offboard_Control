# coding: utf-8
from scipy.integrate import odeint, solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import cos, sin
import scipy
import matplotlib.pyplot as plt

# Simulation parameters
g = 9.81
m = 1.56
jx = 0.029125
jy = 0.029125
jz = 0.055225
J = np.diag([jx, jy, jz]) # moment of Inertia in Kg(m^2)


# Interial fram axis
e1 = np.array([[1,0,0]]).T
e2 = np.array([[0,1,0]]).T
e3 = np.array([[0,0,1]]).T
d = 0.315    # distance of center of mass from fram center in m
c = 8.004*10e-4 # fixed constant in m


#########################################################################


def hat_map(v):
    v = np.squeeze(v, axis=1)
    return np.array([[0, -v[2], v[1]], [v[2], 0 ,-v[0]], [-v[1], v[0], 0]])

def vee_map(R):
    return np.array([R[2, 1],R[0, 2],R[1, 0]])

def vec_cross(a, b):
#     print('a,b', a,b)
    tmp = np.array([[(a[1]*b[2]-a[2]*b[1]),(-a[0]*b[2] + a[2]*b[0]),(a[0]*b[1] - a[1]*b[0])]]).T
#     print('temp',tmp.shape)
    ans = np.squeeze(tmp, axis=0)
#     print('ans',ans)
    return ans

#########################################################################

# ode function 
def step(y,t): #odeint 
    x_curr = y[0:3]
    x_curr.resize(3,1)
    v_curr = y[3:6]
    v_curr.resize(3,1)
    R = y[6:15].reshape((3,3))
    w = y[15:18]
    w.resize(3,1)
    b3 = R[:,2] 
    b3 = b3[...,np.newaxis]
    
    # parameters
    k1 = np.diag([5, 5 ,9])
    k2 = np.diag([5, 5 ,10])
    kR = 200
    kOm = 1


    # desired states
    des_pos = np.array([[0.4*t, 0.4*sin(np.pi*t), 0.6*cos(np.pi*t)]]).T
    des_vel = np.array([[0.4, 0.4* np.pi* cos(np.pi*t), -0.6*np.pi*sin(np.pi*t)]]).T
    des_acc = np.array([[0, -0.4*np.pi*np.pi*sin(np.pi*t),-0.6*np.pi*np.pi*cos(np.pi*t)]]).T 

    b1d = np.array([[cos(np.pi*t), sin(np.pi*t), 0]]).T
    w_desired = np.array([[0,0,0]]).T

    # Position Control
    pos_e = np.subtract(x_curr, des_pos)
    vel_e = np.subtract(v_curr, des_vel)
    A = (-np.dot(k1,pos_e) -np.dot(k2,vel_e) + np.dot(m,des_acc) + np.dot(m*g,e3))

    b3_desired = A/np.linalg.norm(A) 
    f = np.dot(A.T,b3)             # Thurst
#     print('f',f)

    
    # Attitude Control

    b2d = vec_cross(b3_desired,b1d)
    b2d = b2d/np.linalg.norm(b2d) 
    projection_b1d = -vec_cross(b3_desired,b2d)
    check = vec_cross(b2d,b3_desired)
    Rd_middle = np.concatenate((check, b2d), axis=1)
    Rd = np.concatenate((Rd_middle, b3_desired), axis=1)
    
    # calculating error in angular velocity and attitude
    tmp = np.subtract(np.dot(Rd.T, R), np.dot(R.T, Rd))
    R_e = 1/2 * vee_map(tmp)
    R_e = R_e[..., np.newaxis]
#     print('R_e',R_e)
    
    Om_e = w - (R.T @ (Rd @ w_desired))
#     print('Om_e',Om_e)
    B = (R.T @ (Rd @ w_desired))
        
    C = np.dot(hat_map(w), B) - B
    
    M = - (kR * R_e) - (kOm * Om_e) + vec_cross(w, (J @ w)) - (J @ C)  # Moment {wx, wy, wz}

    v_dot = np.dot(-g,e3) + np.dot((f/m)*R, e3)

    R_dot = (R @ hat_map(w))    

    tmp = (-vec_cross(w, (J @ w)) + M)

    Omega_dot = np.dot(np.linalg.inv(J), tmp)

    dydt = np.concatenate((v_curr, v_dot, R_dot.reshape((9,1)),Omega_dot), axis=0)
    dydt = np.squeeze(dydt, axis=1)

    return dydt


######################################################################### 


def main():
    
    # initial pt
    R0 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    y0 = np.concatenate((0, 0, 0, 0, 0, 0, R0,0, 0, 0), axis=None)
    
    # slove ode
    n = 40
    t = np.linspace(0, 4, n)
    sol = odeint(step, y0, t)

    # plot solution 
    
    x = []
    y = []
    z = [] 

    for i in range(len(t)):
        x.append(0.4*t[i])
        y.append(0.4*sin(np.pi*t[i]))
        z.append(0.6*cos(np.pi*t[i]))
    
    x.append(0.4*t[n-1]) 
    y.append(0.4*sin(np.pi*t[n-1])) 
    z.append(0.6*cos(np.pi*t[n-1]))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], 'b', label='Real trajectory')
    ax.plot(x, y, z, 'r-', linewidth = 0.8, label='Desired trajectory')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-2,2)
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
