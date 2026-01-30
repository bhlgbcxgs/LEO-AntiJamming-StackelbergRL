import random
from linksimulation import *


lat_u1 = 38.913611
lon_u1 = -77.013222
loc_u1 = np.array(latlon2ecef(lat_u1, lon_u1))

# user2 to the east 200km of user1
lat_u2 = lat_u1
lon_u2 = lon_u1 + 200 / (math.pi * math.cos(lon_u1 * math.pi / 180) * 6378) * 180
loc_u2 = np.array(latlon2ecef(lat_u2, lon_u2))

# user3 to the north 200km of user1
lat_u3 = lat_u1 + 200 / (math.pi * math.cos(lat_u1 * math.pi / 180) * 6378) * 180
lon_u3 = lon_u1
loc_u3 = np.array(latlon2ecef(lat_u3, lon_u3))

loc_u = np.array([loc_u1, loc_u2, loc_u3])

# jammer1 north 100km east 100km
lat_j1 = lat_u1 + 100 / (math.pi * math.cos(lat_u1 * math.pi / 180) * 6378) * 180
lon_j1 = lon_u1 + 100 / (math.pi * math.cos(lon_u1 * math.pi / 180) * 6378) * 180
loc_j1 = np.array(latlon2ecef(lat_j1, lon_j1))

# jammer2 north 100km west 100km
lat_j2 = lat_u1 + 100 / (math.pi * math.cos(lat_u1 * math.pi / 180) * 6378) * 180
lon_j2 = lon_u1 - 100 / (math.pi * math.cos(lon_u1 * math.pi / 180) * 6378) * 180
loc_j2 = np.array(latlon2ecef(lat_j2, lon_j2))

# jammer3 south 100km west 100km
lat_j3 = lat_u1 - 100 / (math.pi * math.cos(lat_u1 * math.pi / 180) * 6378) * 180
lon_j3 = lon_u1 - 100 / (math.pi * math.cos(lon_u1 * math.pi / 180) * 6378) * 180
loc_j3 = np.array(latlon2ecef(lat_j3, lon_j3))

# jammer4 south 100km east 100km
lat_j4 = lat_u1 - 100 / (math.pi * math.cos(lat_u1 * math.pi / 180) * 6378) * 180
lon_j4 = lon_u1 + 100 / (math.pi * math.cos(lon_u1 * math.pi / 180) * 6378) * 180
loc_j4 = np.array(latlon2ecef(lat_j4, lon_j4))

# jammer5 north 300km east 100km
lat_j5 = lat_u1 + 300 / (math.pi * math.cos(lat_u1 * math.pi / 180) * 6378) * 180
lon_j5 = lon_u1 + 100 / (math.pi * math.cos(lon_u1 * math.pi / 180) * 6378) * 180
loc_j5 = np.array(latlon2ecef(lat_j5, lon_j5))

# jammer6 north 100km east 300km
lat_j6 = lat_u1 + 100 / (math.pi * math.cos(lat_u1 * math.pi / 180) * 6378) * 180
lon_j6 = lon_u1 + 300 / (math.pi * math.cos(lon_u1 * math.pi / 180) * 6378) * 180
loc_j6 = np.array(latlon2ecef(lat_j6, lon_j6))

loc_j = np.array([loc_j1, loc_j2, loc_j3, loc_j4, loc_j5, loc_j6])

# v=[0,10) # 6 satellites chosen
'''
starlink_num: 12 ,sat_num: 36
starlink_num: 13 ,sat_num: 12
starlink_num: 14 ,sat_num: 37
starlink_num: 14 ,sat_num: 38
starlink_num: 15 ,sat_num: 39
starlink_num: 20 ,sat_num: 46
'''
v0 = 40 - 34.9

mean_anomaly = [72.0, 1.9, 32.6, 358.5, 65.3, 124.8, 4.3, 6.2, 8.1, 17.2, 11.9, 13.8, 23.0, 204.9, 19.6, 21.5,
                23.4,
                32.5, 27.2, 29.1, 31.1, 40.2, 34.9, 36.8, 38.7, 47.8, 42.5, 44.4, 53.6, 55.5, 50.2, 52.1]
starlink = [0.0, 11.3, 22.5, 33.8, 45.0, 56.3, 67.5, 78.8, 90.0, 101.3, 112.5, 123.8, 135.0, 146.3, 157.5,
            168.8, 180.0,
            191.3, 202.5, 213.8, 225.0, 236.3, 247.5, 258.8, 270.0, 281.3, 292.5, 303.8, 315.0, 326.3, 337.5,
            348.8]
link_num = [[12, 13, 14, 14, 15, 20],
            [12, 13, 13, 14, 21, 22],
            [12, 13, 14, 21, 22, 22],
            [12, 13, 14, 14, 15, 21],
            [12, 13, 13, 14, 15, 21],
            [13, 14, 21, 22, 22, 23],
            [12, 13, 14, 14, 15, 21],
            [12, 13, 13, 14, 15, 22],
            [13, 14, 15, 21, 22, 22],
            [13, 14, 14, 15, 21, 22],
            [13, 13, 14, 14, 15, 22],
            [13, 14, 15, 21, 22, 23],
            [13, 14, 14, 15, 21, 22],
            [13, 13, 14, 14, 15, 22],
            [13, 14, 15, 22, 23, 23],
            [13, 14, 14, 15, 15, 16],
            [13, 14, 14, 15, 22, 23],
            [13, 14, 15, 22, 23, 23],
            [13, 14, 14, 15, 15, 16],
            [13, 14, 14, 15, 16, 22],
            [13, 14, 15, 22, 23, 23],
            [13, 14, 15, 15, 16, 22],
            [14, 14, 15, 16, 22, 23],
            [14, 15, 22, 23, 23, 24],
            [13, 14, 15, 15, 16, 22],
            [14, 14, 15, 16, 23, 24],
            [14, 15, 16, 22, 23, 23],
            [14, 15, 15, 16, 22, 23],
            [14, 14, 15, 16, 23, 24],
            [14, 15, 16, 23, 24, 24],
            [14, 15, 15, 16, 17, 23],
            [14, 14, 15, 15, 16, 23],
            [14, 15, 16, 23, 24, 24],
            [14, 15, 15, 16, 16, 17],
            [15, 15, 16, 23, 24, 24],
            [14, 15, 16, 17, 23, 24],
            ]
satellite_num = [[36, 12, 37, 38, 39, 46],
                 [37, 13, 14, 39, 0, 0],
                 [39, 15, 40, 1, 1, 2],
                 [40, 16, 41, 42, 43, 2],
                 [41, 17, 18, 43, 44, 4],
                 [19, 44, 5, 5, 6, 6],
                 [44, 20, 45, 46, 47, 6],
                 [45, 21, 22, 47, 48, 8],
                 [23, 48, 49, 9, 9, 10],
                 [24, 0, 49, 1, 10, 10],
                 [25, 26, 0, 1, 2, 12],
                 [27, 2, 3, 13, 13, 14],
                 [28, 3, 4, 5, 14, 14],
                 [29, 30, 4, 5, 6, 16],
                 [31, 6, 7, 17, 18, 19],
                 [32, 7, 8, 8, 9, 10],
                 [33, 8, 9, 10, 20, 21],
                 [35, 10, 11, 21, 22, 23],
                 [36, 11, 12, 12, 13, 14],
                 [37, 12, 13, 14, 15, 24],
                 [39, 14, 15, 25, 26, 27],
                 [40, 15, 16, 17, 18, 26],
                 [16, 17, 18, 19, 28, 29],
                 [18, 19, 29, 30, 31, 31],
                 [44, 19, 20, 21, 22, 30],
                 [20, 21, 22, 23, 33, 34],
                 [22, 23, 24, 33, 34, 35],
                 [23, 24, 25, 26, 34, 35],
                 [24, 25, 26, 27, 37, 38],
                 [26, 27, 28, 38, 39, 40],
                 [27, 28, 29, 30, 32, 40],
                 [28, 29, 29, 30, 31, 41],
                 [30, 31, 32, 42, 43, 44],
                 [31, 32, 33, 33, 34, 36],
                 [33, 34, 35, 45, 46, 47],
                 [34, 35, 36, 39, 46, 47],
                 ]

class EnvWF():

    def __init__(self, snr_norm, angle_norm):
        self.satellite_num = 2
        self.snr_norm = snr_norm
        self.angle_norm = angle_norm



    def step(self, state, action, mc):
        """
        :param action: power allocation of each channel[0~1] (2 * satellite_num--means and variances)
        :param state: channel gain of each channel(total power * g_i / (jammer_power + noise_power))---alloc total=1
        :return: reward: total spectral efficiency
        """
        spec_effi = []
        for i in range(self.satellite_num):
            snr = 10 * self.snr_norm * state[i] + 10 * np.log10(action[i])
            angle = self.angle_norm * state[i+self.satellite_num]
            spec_effi.append(env_reward_cal(snr, mc[i], angle))
        return spec_effi

    def ge_rnd_state(self, v):
        v = random.random() * 360
        set_index = int(v // 10)
        link_current = link_num[set_index]
        satellite_current = satellite_num[set_index]
        phi = v * 108 / 1440
        sat_pos = []
        sat_num = 6
        user_num = 3
        for i in range(sat_num):
            sat_pos.append(
                ecef_position(v + v0 + mean_anomaly[link_current[i]] - 7.2 * satellite_current[i], phi, rs=7528, w=0, omega=starlink[link_current[i]],
                              i=53))

        mat_u = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        for u in range(user_num):
            sat_connect_cur_t = np.sum(mat_u, axis=1)
            sat_va = np.where(sat_connect_cur_t < 2)[0]
            sat_index = np.random.choice(sat_va, 2, replace=False)
            mat_u[sat_index[0]][u] = 1
            mat_u[sat_index[1]][u] = 1

        mat_j = [0, 1, 2, 3, 4, 5]
        random.shuffle(mat_j)

        sim = SatelliteSimulation(mat_u, mat_j, v)
        sim.P = np.ones((sim.U, sim.S))
        sinr = sim.calculate_sinr_optimized()
        state = []

        for u in range(user_num):
            state_u = []

            for s in range(sat_num):
                if mat_u[s][u] == 1:  # user u communicate with s
                    snr = sinr[u, s]
                    if snr < 0.001:
                        state_u.append(-3 / self.snr_norm)
                    else:
                        state_u.append(np.log10(snr) / self.snr_norm)

            for s in range(sat_num):
                if mat_u[s][u] == 1:
                    e_us = (math.pi / 2 - cal_angle(loc_u[u], sat_pos[s] - loc_u[u])) * 180 / math.pi
                    state_u.append(e_us / self.angle_norm)

            x = np.random.rand()
            if x < 0.3:
                shadowed_sat = random.randint(0, 1)
                state_u[shadowed_sat] = -3 / self.snr_norm
            state.append(state_u)

        return state


