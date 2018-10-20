#!/usr/bin/python
#-*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import RK45


class Simulator:
    def __init__(self):
        self.last_global_state = None
        self.last_local_state = None
        self.current_action = None
        self.steps = 0
        self.time_span = 10          # 2 seconds for each iteration
        self.number_iterations = 100  # 100 iterations for each step
        self.integrator = None
        self.rk_mode = 'scipy_rk'

        ##Vessel Constants

        self.M = 115000      *10**3
        self.M11 = 14840.4   * 10**3
        self.M22 = 174050    * 10**3
        self.M26 = 38369.6   * 10**3
        self.M66 = 364540000 * 10**3
        self.Iz = 414000000  * 10**3
        self.L = 244.74 #length
        self.Draft = 15.3
        self.x_g = 2.2230# center mass
        self.x_prop = -112 #propulsor position
        self.force_prop_max = 1.6 * 10**6 # max porpulsor force
        self.x_rudder = -115 # rudder position
        self.rudder_area = 68

        self.Cy = 0.06           # coeff de arrasto lateral
        self.lp = 7.65 # cross-flow center
        self.Cb = 0.85           # block coefficient
        self.B = 42             # Beam
        self.S = 27342        # wet surface


        ## Water constants
        self.pho = 1.025 * 10**3# water density
        self.mi = 10**-3  # water viscosity

    def reset_start_pos(self, global_vector):
        x0, y0, theta0, vx0, vy0, theta_dot0 = global_vector[0], global_vector[1], global_vector[2], global_vector[3], global_vector[4], global_vector[5]
        self.last_global_state = np.array([x0, y0, theta0, vx0, vy0, theta_dot0])
        self.last_local_state = self._global_to_local(self.last_global_state)
        if self.rk_mode == 'scipy_rk':
            self.current_action = np.zeros(2)
            self.integrator = self.scipy_runge_kutta(self.simulate_scipy, self.get_state(), t_bound=self.time_span)

    def step(self, angle_level, rot_level):
        self.current_action = np.array([angle_level, rot_level])
        if self.rk_mode == 'ours_rk':
            for i in range(self.number_iterations):
                self.last_global_state = self.runge_kutta(self.get_state(), self.simulate_in_global, 6, self.time_span/self.number_iterations)
            return self.last_global_state

        if self.rk_mode == 'scipy_rk':
            while not (self.integrator.status == 'finished'):
                self.integrator.step()

            self.last_global_state = self.integrator.y
            self.last_local_state = self._global_to_local(self.last_global_state)
            self.integrator = self.scipy_runge_kutta(self.simulate_scipy, self.get_state(), t0=self.integrator.t, t_bound=self.integrator.t+self.time_span)
            return self.last_global_state

    def simulate_scipy(self, t, global_states):
        local_states = self._global_to_local(global_states)
        return self._local_ds_global_ds(global_states[2], self.simulate(local_states))

    def simulate_in_global(self, global_states):
        local_states = self._global_to_local(global_states)
        return self._local_ds_global_ds(global_states[2], self.simulate(local_states))

    def simulate(self, local_states):
        """
        Suppose that the vessel is a simple bloc with the following dynamics:
        u :  local coordinate aligned with the vessel
        v :  local coordinate aligned with the vessel

        u = cos(theta) X + sin(theta) Y
        v = -sin(theta) X + cos(theta) Y
        dtheta : equal for both


        (M+M11) * ddu     - (M+M22) * dv * dtheta        - (M*xg + M26) * dtheta**2    =  F1u + Fpx
        (M+M22) * ddv     + (M*xg + M26)*ddtheta + (M + M11)*du * dtheta       = F1v + Fpy
        (Iz+M66)* ddtheta + (M*xg+M26)*ddy       + (M*xg + M26) * dudtheta     = F1v + Tprop

        Where:

        F1u = 0.5 * pho * Vc**2 * L * Draft * C1
            C1 = C0*cos(gamma)+( cos(3*gamma - cos(gamma) ) ) * pi * Draft / (8 * L)
                C0 = 0.0094 * S /( Draft * L) / (log10(Re)-2)**2
                    Re = pho*Vc*L/mi

        F1v = 0.5 * pho * Vc**2 * L * Draft * C2
            C2 = (Cy - 0.5*pi*Draft/L)*sin(gamma)*mod(sin(gamma)) + 0.5*pi*Draft/L * (sin(gamma)**3) + pi*Draft/L*(1+0.4*Cb*B/Draft)*sin(gamma)*abs(cos(gamma))

        F1z = 0.5 * pho * Vc**2 * L * Draft * C6
            C6 =    -lp/L * Cy * sin(gamma) * mod(sin(gamma))
            C6 = C6 -pi*Draft/L * sin(gamma) * cos(gamma)
            C6 = C6 - ( 0.5 + 0.5*mod( cos(gamma) ) )**2 * pi*Draft/L*(0.5 - 2.4*Draft/L)*sin(gamma)* mod(cos(gamma))

        # Suppose no natural waterflow
        Vc = sqrt(u**2+v**2)
        gamma = pi + atan2(v, u)

        # modelo simlificado de propulsão
        beta = actions[0]
        alpha = actions[1]
        Fpx = cos(beta) * force_prop_max * alpha
        Fpy = sin(beta) * force_prop_max * alpha
        Fpz = sin(beta) * force_prop_max * alpha * x_rudder

        x1= u
        x2 = du

        x3 = v
        x4 = dv

        x5 = th
        x6 = dth

        fx1 =   x2
        fx2 =  (F1u + Fpx + (M+M22) * x4 * x6  - (M*xg + M26) * x6**2 ) / (M+M11)

        fx3 = x4
        fx5 = x6

        a11 = (M+M22)
        a12 = (M*xg + M26)
        a21 = (Iz+M66)
        a22 = (M*xg+M26)
        b1 =  -(M + M11)*x2 * x6  + F1v + Fpy
        b2 =  -(M*xg + M26) * x2*x6 + F1v + Tprop

        A = np.array([[a11; a12];[a21; a22])
        B = np.array([b1; b2])
        fx46 = A.inv*B
        fx4 = fx46[0]
        fx6 = fx46[1]

        fx = [fx1 fx2 fx3 fx4]




        :param local_states: Space state
        :return df_local_states
        """
        x1 = local_states[0] #u
        x2 = local_states[1] #v
        x3 = local_states[2] #theta (not used)
        x4 = local_states[3] #du
        x5 = local_states[4] #dv
        x6 = local_states[5] #dtheta
        beta = self.current_action[0]*np.pi/12   #leme
        alpha = self.current_action[1]    #propulsor

        vc = np.sqrt(x4 ** 2 + x5 ** 2)
        gamma = np.pi + np.arctan2(x5, x4)

        # Composing resistivity forces
        Re = self.pho * vc * self.L / self.mi
        if Re == 0:
            C0=0
        else:
            C0 = 0.0094 * self.S / (self.Draft * self.L) / (np.log10(Re) - 2) ** 2
        C1 = C0 * np.cos(gamma) + (np.cos(3 * gamma) - np.cos(gamma)) * np.pi * self.Draft / (8 * self.L)
        F1u = 0.5 * self.pho * vc ** 2 * self.L * self.Draft * C1

        C2 = (self.Cy - 0.5 * np.pi * self.Draft / self.L) * np.sin(gamma) * np.abs(np.sin(gamma)) + 0.5 * np.pi * self.Draft / self.L * (
                np.sin(gamma) ** 3) + np.pi * self.Draft / self.L * (1 + 0.4 * self.Cb * self.B / self.Draft) * np.sin(gamma) * np.abs(np.cos(gamma))
        F1v = 0.5 * self.pho * vc ** 2 * self.L * self.Draft * C2

        C6 = -self.lp / self.L * self.Cy * np.sin(gamma) * np.abs(np.sin(gamma))
        C6 = C6 - np.pi * self.Draft / self.L * np.sin(gamma) * np.cos(gamma)
        C6 = C6 - (0.5 + 0.5 * np.abs(np.cos(gamma))) ** 2 * np.pi * self.Draft / self.L * (0.5 - 2.4 * self.Draft / self.L) * np.sin(gamma) * np.abs(np.cos(gamma))
        F1z = 0.5 * self.pho * vc ** 2 * self.L * self.Draft * C6

        # Propulsion model
        Fpx = np.cos(beta) * self.force_prop_max * alpha
        Fpy = -np.sin(beta) * self.force_prop_max * alpha
        Fpz = -np.sin(beta) * self.force_prop_max * self.x_rudder

        # without resistence
        #F1u, F1v, F1z = 0, 0, 0
        # Derivative function

        fx1 = x4
        fx2 = x5
        fx3 = x6
        # simple model
        # fx4 = (F1u + Fpx)/(self.M + self.M11)
        # fx5 = (F1v + Fpy)/(self.M + self.M22)
        # fx6 = (F1z + Fpz)/(self.Iz + self.M66)

        a11 = (self.M)
        a12 = (self.M * self.x_g)
        a21 = (self.M * self.x_g)
        a22 = (self.Iz )
        b1 = -(self.M) * x4 * x6 + F1v + Fpy
        b2 = -(self.M * self.x_g)* x6 + F1z + Fpz
        A = np.array([[a11, a12], [a21, a22]])
        B = np.array([b1, b2])
        fx56 = np.dot(np.linalg.inv(A), B.transpose())

        fx4 = x6 * x5 + x6 ** 2 + (F1u + Fpx) / self.M
        fx5 = fx56[0]
        fx6 = fx56[1]



        fx = np.array([fx1, fx2, fx3, fx4, fx5, fx6])
        return fx

    def scipy_runge_kutta(self, fun, y0, t0=0, t_bound=10):
        return RK45(fun, t0, y0, t_bound,  rtol=self.time_span/self.number_iterations, atol=1e-5)

    def runge_kutta(self, x, fx, n, hs):
        k1 = []
        k2 = []
        k3 = []
        k4 = []
        xk = []
        ret = np.zeros([n])
        for i in range(n):
            k1.append(fx(x)[i]*hs)
        for i in range(n):
            xk.append(x[i] + k1[i]*0.5)
        for i in range(n):
            k2.append(fx(xk)[i]*hs)
        for i in range(n):
            xk[i] = x[i] + k2[i]*0.5
        for i in range(n):
            k3.append(fx(xk)[i]*hs)
        for i in range(n):
            xk[i] = x[i] + k3[i]
        for i in range(n):
            k4.append(fx(xk)[i]*hs)
        for i in range(n):
            ret[i] = x[i] + (k1[i] + 2*(k2[i] + k3[i]) + k4[i])/6
        return ret

    def get_state(self):
        return self.last_global_state

    def get_local_state(self):
        return self.last_local_state

    def _local_to_global(self, local_state):
        # local_state: [ux, uy, theta, uxdot, uydot, thetadot]
        theta = local_state[2]
        c, s = np.cos(theta), np.sin(theta)
        A = np.array([[c, -s], [s, c]])
        B_l_pos = np.array([local_state[0], local_state[1]])
        B_l_vel = np.array([local_state[3], local_state[4]])

        B_g_pos = np.dot(A, B_l_pos.transpose())
        B_g_vel = np.dot(A, B_l_vel.transpose())
        return np.array([B_g_pos[0], B_g_pos[1], local_state[2], B_g_vel[0], B_g_vel[1], local_state[5]])

    def _global_to_local(self, global_state):
        # global_states: [x, y, theta, vx, vy, thetadot]
        theta = global_state[2]
        c, s = np.cos(theta), np.sin(theta)
        A = np.array([[c, s], [-s, c]])
        B_g_pos = np.array([global_state[0], global_state[1]])
        B_g_vel = np.array([global_state[3], global_state[4]])

        B_l_pos = np.dot(A, B_g_pos.transpose())
        B_l_vel = np.dot(A, B_g_vel.transpose())
        return np.array([B_l_pos[0], B_l_pos[1], global_state[2], B_l_vel[0], B_l_vel[1], global_state[5]])

    def _local_ds_global_ds(self, theta, local_states):
        """
        The function recieves two local states, one reffering to the state before the runge-kutta and other refering to a
        state after runge-kutta and then compute the global state bated on the transiction

        :param local_states_0: Local state before the transition
        :param local_states_1: Local state after the transition
        :return: global states
        """
        c, s = np.cos(theta), np.sin(theta)
        A = np.array([[c, -s], [s, c]])
        B_l_pos = np.array([local_states[0], local_states[1]])
        B_l_vel = np.array([local_states[3], local_states[4]])

        B_g_pos = np.dot(A, B_l_pos.transpose())
        B_g_vel = np.dot(A, B_l_vel.transpose())

        return np.array([B_g_pos[0], B_g_pos[1], local_states[2], B_g_vel[0], B_g_vel[1], local_states[5]])