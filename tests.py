import unittest
from simulator import *
import numpy as np
from scipy.integrate import RK45


class TestSimulator(unittest.TestCase):

    def test_global_to_local(self):
        xg = np.array([1, 1, np.pi / 4, -1, -1, 0])
        sim1 = Simulator()
        x1 = sim1._global_to_local(xg)
        xref = np.array([np.sqrt(2), 0, np.pi/4, -np.sqrt(2), 0, 0])
        self.assertTrue((x1 == xref).all())

    def test_local_to_global(self):
        xg = np.array([np.sqrt(2), 0,  np.pi / 4, -np.sqrt(2), 0, 1.5])
        sim1 = Simulator()
        x1 = sim1._local_to_global(xg)
        xref = np.array([1, 1, np.pi/4, -1, -1, 1.5])
        self.assertTrue(np.allclose(x1, xref))

    def test_test_reset_pos(self):
        x0 = np.array([1, 1, np.pi / 4, -1, -1, 0])
        sim = Simulator()
        sim.reset_start_pos(x0)
        x0_set = sim.get_state()

        self.assertTrue(np.allclose(x0, x0_set))

        x0_local_local = sim.get_local_state()
        x0_local_ref = np.array([np.sqrt(2), 0, np.pi/4, -np.sqrt(2), 0, 0])
        self.assertTrue(np.allclose(x0_local_local, x0_local_ref))

    def test_runge_kutta(self):
        """
        Test 2-separeted mass-spring system dynamics
        """
        states = np.array([1, 0, 1, 0])
        t0 = 0
        tmax1 = np.pi*2 # period/2 ==> opposite position
        tmax2 = np.pi*5 # period/2 ==> opposite position
        h = 0.01
        N1 = int(np.round(tmax1/h))
        N2 = int(np.round(tmax2/h))
        sim = Simulator()
        # (x, fx, n, hs)
        for i in range(N1):
            states = sim.runge_kutta(states, _mass_spring, 4, h)
        self.assertAlmostEqual(states[0], -1, places=4)
        for i in range(N2-N1):
            states = sim.runge_kutta(states, _mass_spring, 4, h)
        self.assertAlmostEqual(states[2], -1, places=4)

    def test_simulation(self):
        sim = Simulator()

        # first case: Vessel with no velocity, no action and nothing should happen
        actions = np.zeros(2)
        sim.current_action = actions
        states = np.array([10, 0, 0, 0, 0, 0])
        df = sim.simulate(states)
        self.assertTrue(np.allclose(df, np.zeros(6)))
        # if the vessel has only velocity on x, only df[0] and df[3] should be not-null
        states = np.array([10, 0, 0, 1, 0, 0])
        df = sim.simulate(states)
        self.assertTrue(df[0] > 0 and df[3] < 0)
        # we acceleration test
        states = np.array([10, 0, 0, 0, 0, 0])
        sim.current_action = np.array([0, 1])
        df = sim.simulate(states)
        self.assertTrue(df[1] == 0)
        self.assertTrue(df[3] > 0)
        self.assertTrue(df[4] == 0)
        self.assertTrue(df[5] == 0)

    def test_local_ds_global_ds(self):
        sim = Simulator()
        # first case: Vessel with no velocity, no action and nothing should happen
        local_s_0 = np.array([0, 0, 0, 0, 0, 0])
        local_s_1 = np.array([1, 1, 0.1*np.pi/4, 1, 1, 0.1])
        theta = local_s_0[2]
        global_s = sim._local_ds_global_ds(theta, local_s_1)
        self.assertTrue(np.allclose(global_s, np.array([1, 1, 0.1*np.pi/4, 1, 1, 0.1])))

        local_s_0 = np.array([np.sqrt(2), 0, np.pi/4, np.sqrt(2), 0, 0.1*np.pi/4])
        local_s_1 = np.array([1, 1, np.pi/2, 1, 1, 0.2*np.pi/4])
        theta = local_s_0[2]
        global_s = sim._local_ds_global_ds(theta, local_s_1)
        self.assertTrue(np.allclose(global_s, np.array([0, np.sqrt(2), np.pi/2, 0, np.sqrt(2), 0.2*np.pi/4])))

        local_s_0 = np.array([np.sqrt(2), 0, -np.pi / 4, np.sqrt(2), 0, -0.1 * np.pi / 4])
        local_s_1 = np.array([1, -1, -np.pi / 2, 1, -1, -0.2 * np.pi / 4])
        theta = local_s_0[2]
        global_s = sim._local_ds_global_ds(theta, local_s_1)
        self.assertTrue(np.allclose(global_s, np.array([0, -np.sqrt(2), -np.pi / 2, 0, -np.sqrt(2), -0.2 * np.pi / 4])))

    def test_step(self):
        sim = Simulator()
        states = np.array([10, 0, 0, 0, 0, 0])
        actions = np.array([0, 1])
        sim.reset_start_pos(states)
        sim.step(actions[0], actions[1])
        new_states = sim.get_state()
        self.assertTrue(new_states[0]> states[0])

    def test_rotation(self):
        sim = Simulator()
        states = np.array([10, 10, np.pi/4, 0, 0, 0])
        actions = np.array([0, 1])
        sim.reset_start_pos(states)
        sim.step(actions[0], actions[1])
        new_states = sim.get_state()
        self.assertTrue(new_states[0]> states[0])

    def test_episode(self):
        sim = Simulator()
        states = np.array([0, 100, -np.pi/4, 1, -1, 0])
        actions = np.array([0, 1])
        sim.reset_start_pos(states)
        for i in range(10):
            sim.step(actions[0], actions[1])
        new_states = sim.get_state()
        self.assertTrue(new_states[0]>0)

    def test_scipy_RK45(self):
        t0 = 0
        y0 = np.array([1, 0, 1, 0])
        tmax1 = np.pi * 2  # period/2 ==> opposite position
        tmax2 = np.pi * 5  # period/2 ==> opposite position
        h = 0.01

        integrator = RK45(_mass_spring_sp, t0, y0, rtol=h, atol=10**-6, t_bound=tmax1)
        while not (integrator.status == 'finished'):
            integrator.step()
        Y1 = integrator.y
        T1 = integrator.t
        integrator = RK45(_mass_spring_sp, T1, Y1, rtol=h, atol=10 ** -6, t_bound=tmax2)
        while not (integrator.status == 'finished'):
            integrator.step()
        Y2 = integrator.y
        T2 = integrator.t

        self.assertAlmostEqual(Y1[0], -1, places=2)
        self.assertAlmostEqual(Y2[2], -1, places=2)
        self.assertAlmostEqual(T1, tmax1, places=4)
        self.assertAlmostEqual(T2, tmax2, places=4)

def _mass_spring_sp(t, x):
    return _mass_spring(x)

def _mass_spring(x):
    """
    2-separeted mass-spring system dynamics
    """
    x1  = x[0]
    dx1 = x[1]
    x2  = x[2]
    dx2 = x[3]
    m1  = 16
    k1  = 4
    m2  = 25
    k2 = 1

    # m ddx1 + k1*x1 = 0
    # T = 2*pi sqrt(m1/k1) = 8*pi
    # m ddx2 + k2*x2 = 0
    # T = 2*pi sqrt(m2/k2) = 10*pi

    fx1 = dx1
    fx2 = -k1/m1 * x1
    fx3 = dx2
    fx4 = -k2/m2 * x2
    fx = np.array([fx1, fx2, fx3, fx4])
    return fx

if __name__ == '__main__':
    unittest.main()