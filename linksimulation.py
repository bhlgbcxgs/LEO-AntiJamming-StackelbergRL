from scipy.constants import c, k
from satellite_param import *
from rician import *

import time


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


class SatelliteSimulation:
    def __init__(self, mat_us, mat_js, v, num_antennas=1296, frequency=14e9):
        self.U = 3
        self.S = 6
        self.J = 6
        self.Mr = num_antennas
        self.Mt = num_antennas
        self.frequency = frequency
        self.wavelength = c / frequency
        self.antenna_spacing = 0.5 * self.wavelength

        # satellite set
        self.v = v
        self.set_index = int(self.v // 10)
        self.link_current = link_num[self.set_index]
        self.satellite_current = satellite_num[self.set_index]
        self.phi = self.v * 108 / 1440

        # initialize positions
        self.satellite_positions = []
        for i in range(self.S):
            self.satellite_positions.append(
                ecef_position(self.v + v0 + mean_anomaly[self.link_current[i]] - 7.2 * self.satellite_current[i], self.phi, rs=7528, w=0, omega=starlink[self.link_current[i]],
                              i=53))
        self.user_positions = loc_u
        self.jammer_positions = loc_j

        self.mat_us = mat_us
        self.mat_js = mat_js
        self.mat_us = np.array(self.mat_us)
        self.mat_js = np.array(self.mat_js)

        self.k_factor = 10
        self.elevation_angles = np.zeros((self.U, self.S))
        self.k_factors = np.zeros((self.U, self.S))

        # power parameters
        self.power_jammer_dbw = 30
        self.power_user_dbw = -15

        self.total_power = math.pow(10, self.power_user_dbw / 10)
        self.jammer_power = math.pow(10, self.power_jammer_dbw / 10)
        self.noise_figure = 3
        self.bandwidth = 1e6
        self.temperature = 290

        # calculate distance matrix and elevation angle
        self._calculate_geometry()

        self.G = self._compute_channel_matrices_jammer()

        self.sinr_values = np.zeros((self.U, self.S))
        self.spectral_efficiencies = np.zeros((self.U, self.S))

        self.P = np.zeros((self.U, self.S))

    def path_loss(self, distance):
        """calculate path loss"""
        return (self.wavelength / (4 * np.pi * distance)) ** 2

    def manual_cdist(self, X, Y):
        return np.sqrt(np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2))

    def _calculate_geometry(self):
        self.dist_us = np.zeros((self.U, self.S))
        self.dist_js = np.zeros((self.J, self.S))

        for u in range(self.U):
            for s in range(self.S):
                self.dist_us[u, s] = np.sqrt((self.user_positions[u] - self.satellite_positions[s]).dot(self.user_positions[u] - self.satellite_positions[s])
)
        for j in range(self.J):
            for s in range(self.S):
                self.dist_js[j, s] = np.sqrt((self.jammer_positions[j] - self.satellite_positions[s]).dot(self.jammer_positions[j] - self.satellite_positions[s])
)
        self.theta_u = np.zeros((self.U, self.S))
        self.phi_u = np.zeros((self.U, self.S))

        for u in range(self.U):
            for s in range(self.S):
                delta = self.satellite_positions[s] - self.user_positions[u]
                self.theta_u[u, s] = np.arctan2(delta[1], delta[0])
                self.phi_u[u, s] = np.arctan2(delta[2], np.sqrt(delta[0] ** 2 + delta[1] ** 2))

    def steering_vector(self, M, theta, phi, type='tx'):
        # UPA assumption
        M_az = int(np.sqrt(M))
        M_el = int(np.sqrt(M))
        d = self.antenna_spacing

        psi_az = 2 * np.pi * d * np.sin(theta) * np.cos(phi) / self.wavelength
        psi_el = 2 * np.pi * d * np.sin(phi) / self.wavelength

        m = np.arange(M_az)
        n = np.arange(M_el)
        az_phase = np.exp(1j * m * psi_az)
        el_phase = np.exp(1j * n * psi_el)

        a = np.kron(az_phase, el_phase)

        if type == 'rx':
            a = a.conj()

        return a

    def _compute_channel_matrices_jammer(self):
        """initialize the interference vector of jammer"""
        time_start = time.perf_counter_ns()
        G = np.zeros((self.S, self.J, self.Mr, 1), dtype=complex)

        for s in range(self.S):
            for j in range(self.J):
                elevation = (math.pi / 2 - cal_angle(self.satellite_positions[s] - self.jammer_positions[j],
                                                     self.jammer_positions[j])) * 180 / math.pi
                elevation_divide = [30, 60, 70, 80, 90]
                k_factors = [0, 3.9, 5.6, 9.8, 17.1]
                k_factor = k_factors[bisect_left(elevation_divide, elevation)]
                delta = self.satellite_positions[s] - self.jammer_positions[j]
                theta = np.arctan2(delta[1], delta[0])
                phi = np.arctan2(delta[2], np.sqrt(delta[0] ** 2 + delta[1] ** 2))
                a_rx = self.steering_vector(self.Mr, theta, phi, 'rx')
                G_los = a_rx.reshape(-1, 1)
                G_nlos = (np.random.randn(self.Mr, 1) + 1j * np.random.randn(self.Mr, 1)) / np.sqrt(2)
                G[s, j] = np.sqrt(k_factor / (k_factor + 1)) * G_los + np.sqrt(1 / (k_factor + 1)) * G_nlos
                path_loss = self.path_loss(self.dist_js[j, s])
                G[s, j] *= np.sqrt(path_loss)
        time_end = time.perf_counter_ns()
        print("**time**compute_channel_matrices_jammer**", time_end - time_start)
        return G

    def _compute_precoding_matrices(self, u, connection_vector):
        connected_sats = np.where(connection_vector == 1)[0]
        d_u = len(connected_sats)
        A = np.zeros((self.Mt, d_u), dtype=complex)
        for i, s in enumerate(connected_sats):
            A[:, i] = self.steering_vector(self.Mt, self.theta_u[u, s], self.phi_u[u, s], 'tx')
        if d_u > 0 :
            W_u = A @ np.linalg.pinv(A.conj().T @ A + 1e-6 * np.eye(d_u))
            # 列归一化
            norms = np.linalg.norm(W_u, axis=0)
            W_u = W_u / norms[np.newaxis, :]
        else:
            W_u = np.zeros((self.Mt, 1))  # default

        return W_u

    def _compute_receive_combining_matrices(self, s, connection_vector):
        connected_users = np.where(connection_vector == 1)[0]
        d_s = len(connected_users)
        A = np.zeros((self.Mr, d_s), dtype=complex)
        for i, u in enumerate(connected_users):
            A[:, i] = self.steering_vector(self.Mr, self.theta_u[u, s], self.phi_u[u, s], 'rx')

        if d_s > 0:
            V_s = A @ np.linalg.pinv(A.conj().T @ A + 1e-6 * np.eye(d_s))
            norms = np.linalg.norm(V_s, axis=0)
            V_s = V_s / norms[np.newaxis, :]
        else:
            V_s = np.zeros((self.Mr, 1))  # default

        return V_s

    def _water_filling_original(self):
        """
           parameters:
               connection_matrix_user: U x S user-satellite connection matrix
               connection_matrix_sat: S x U sat-user connection matrix

           return:
               P: U x S power allocation matrix
           """
        # connection matrix
        connection_matrix_sat = np.array(self.mat_us)
        connection_matrix_user = connection_matrix_sat.T

        np.random.seed(0)
        effective_gains = np.zeros((self.U, self.S))
        V_all = [self._compute_receive_combining_matrices(s, connection_matrix_sat[s])
                 for s in range(self.S)]
        W_all = [self._compute_precoding_matrices(u, connection_matrix_user[u])
                 for u in range(self.U)]

        user_sat_map = [{
            s: idx for idx, s in enumerate(np.where(connection_matrix_user[u] == 1)[0])
        } for u in range(self.U)]
        sat_user_map = [{
            u: idx for idx, u in enumerate(np.where(connection_matrix_sat[s] == 1)[0])
        } for s in range(self.S)]

        for u in range(self.U):
            connected_sats = np.where(connection_matrix_user[u] == 1)[0]
            for s in connected_sats:
                # obtain v_su and w_us of the current link
                v_su = V_all[s][:, sat_user_map[s][u]]
                w_us = W_all[u][:, user_sat_map[u][s]]

                a_rx = self.steering_vector(self.Mr, self.theta_u[u, s], self.phi_u[u, s], 'rx')
                a_tx = self.steering_vector(self.Mt, self.theta_u[u, s], self.phi_u[u, s], 'tx')

                path_loss = self.path_loss(self.dist_us[u, s])

                elevation = (math.pi / 2 - cal_angle(
                    self.satellite_positions[s] - self.user_positions[u],
                    self.user_positions[u])) * 180 / math.pi
                k_factor = self._get_k_factor(elevation)

                h_proj_los = (v_su.conj().T @ a_rx) * (a_tx.conj().T @ w_us)
                h_proj_nlos = (np.random.randn() + 1j * np.random.randn()) * np.linalg.norm(v_su) * np.linalg.norm(
                    w_us) / np.sqrt(2)
                h_proj = np.sqrt(k_factor / (k_factor + 1)) * h_proj_los + np.sqrt(1 / (k_factor + 1)) * h_proj_nlos
                h_proj *= np.sqrt(path_loss)

                effective_gains[u, s] = np.abs(h_proj) ** 2


        for u in range(self.U):
            connected_sats = np.where(connection_matrix_user[u] == 1)[0]
            if len(connected_sats) == 0:
                continue

            gamma_u = effective_gains[u, connected_sats]
            interference_power = []
            for s in connected_sats:
                v_su = V_all[s][:, sat_user_map[s][u]]
                jammer_interf = 0
                noise = k * self.temperature * self.bandwidth * 10 ** (self.noise_figure / 10)
                noise = noise * np.linalg.norm(v_su) ** 2
                for j in range(self.J):
                    if self.mat_js[j] == s:
                        jammer_interf += self.jammer_power * np.abs(
                            v_su.conj().T @ self.G[s, j]) ** 2
                interference_power.append(noise + jammer_interf[0])
            P_u = self._compute_water_filling(interference_power, self.total_power, gamma_u)
            self.P[u, connected_sats] = P_u

        return

    def _compute_water_filling(self, interference_power, total_power, g_i):

        K = len(interference_power)
        if K == 0:
            return 0
        if K == 1:
            powers_opt = np.zeros([K])
            powers_opt[0] = total_power
            return powers_opt
        else:
            channel_gain = []
            for i in range(K):
                channel_gain.append(g_i[i] / interference_power[i])

            channels_sortindex = np.argsort(channel_gain)[::-1]
            channels_sorted = np.array(channel_gain)[channels_sortindex]
            h = 1 / channels_sorted

            if (K - 1) * h[K - 1] - sum(h[np.arange(0, K - 2)]) < total_power:
                idx = K - 1
            else:
                idx_max = K - 1
                idx_min = 0

                while True:
                    idx_mid = math.floor(0.5 * (idx_max + idx_min))
                    w_tmp = idx_mid
                    h_tmp = h[idx_mid]
                    if w_tmp * h_tmp - sum(h[np.arange(0, idx_mid - 1)]) < total_power:
                        idx_min = idx_mid
                    else:
                        idx_max = idx_mid
                    if idx_min + 1 == idx_max:
                        idx = idx_min
                        break

            w_filled = idx + 1
            h_filled = (total_power + sum(h[np.arange(0, idx + 1)])) / w_filled
            p_allocate = np.zeros([K])
            p_allocate[np.arange(0, idx + 1)] = h_filled - h[np.arange(0, idx + 1)]

            powers_opt = np.zeros([K])
            powers_opt[channels_sortindex[np.arange(0, K)]] = p_allocate

        return powers_opt

    def calculate_sinr_optimized(self):
        connection_matrix_sat = self.mat_us
        connection_matrix_user = connection_matrix_sat.T

        V_all = [self._compute_receive_combining_matrices(s, connection_matrix_sat[s])
                 for s in range(self.S)]
        W_all = [self._compute_precoding_matrices(u, connection_matrix_user[u])
                 for u in range(self.U)]

        user_sat_map = [{
            s: idx for idx, s in enumerate(np.where(connection_matrix_user[u] == 1)[0])
        } for u in range(self.U)]

        sat_user_map = [{
            u: idx for idx, u in enumerate(np.where(connection_matrix_sat[s] == 1)[0])
        } for s in range(self.S)]

        SINR_matrix = np.zeros((self.U, self.S))

        for s in range(self.S):
            connected_users_s = np.where(connection_matrix_sat[s] == 1)[0]
            for u in connected_users_s:
                if connection_matrix_user[u, s] != 1:
                    continue

                v_su = V_all[s][:, sat_user_map[s][u]:sat_user_map[s][u] + 1]
                w_us = W_all[u][:, user_sat_map[u][s]:user_sat_map[u][s] + 1]

                a_rx = self.steering_vector(self.Mr, self.theta_u[u, s], self.phi_u[u, s], 'rx')
                a_tx = self.steering_vector(self.Mt, self.theta_u[u, s], self.phi_u[u, s], 'tx')

                elevation = (math.pi / 2 - cal_angle(
                    self.satellite_positions[s] - self.user_positions[u],
                    self.user_positions[u])) * 180 / math.pi
                k_factor = self._get_k_factor(elevation)

                path_loss = self.path_loss(self.dist_us[u, s])

                h_proj_los = (v_su.conj().T @ a_rx) * (a_tx.conj().T @ w_us)

                h_proj_nlos = (np.random.randn() + 1j * np.random.randn()) * np.linalg.norm(v_su) * np.linalg.norm(
                    w_us) / np.sqrt(2)
                h_proj = np.sqrt(k_factor / (k_factor + 1)) * h_proj_los + np.sqrt(1 / (k_factor + 1)) * h_proj_nlos
                h_proj *= np.sqrt(path_loss)

                signal = self.total_power * self.P[u, s] * np.abs(h_proj) ** 2

                noise = k * self.temperature * self.bandwidth * 10 ** (self.noise_figure / 10)
                noise = noise * np.linalg.norm(v_su) ** 2

                intra_interf = 0
                for u_prime in connected_users_s:
                    if u_prime == u: continue

                    elevation = (math.pi / 2 - cal_angle(
                        self.satellite_positions[s] - self.user_positions[u_prime],
                        self.user_positions[u_prime])) * 180 / math.pi
                    k_factor = self._get_k_factor(elevation)

                    w_u_prime_s = W_all[u_prime][:, user_sat_map[u_prime][s]:user_sat_map[u_prime][s] + 1]

                    a_rx_prime = self.steering_vector(self.Mr, self.theta_u[u_prime, s], self.phi_u[u_prime, s], 'rx')
                    a_tx_prime = self.steering_vector(self.Mt, self.theta_u[u_prime, s], self.phi_u[u_prime, s], 'tx')

                    path_loss_prime = self.path_loss(self.dist_us[u_prime, s])

                    h_proj_los_prime = (v_su.conj().T @ a_rx_prime) * (a_tx_prime.conj().T @ w_u_prime_s)
                    h_proj_nlos_prime = (np.random.randn() + 1j * np.random.randn()) * np.linalg.norm(
                        v_su) * np.linalg.norm(w_us) / np.sqrt(2)
                    h_proj_prime = np.sqrt(k_factor / (k_factor + 1)) * h_proj_los_prime + np.sqrt(
                        1 / (k_factor + 1)) * h_proj_nlos_prime
                    h_proj_prime *= np.sqrt(path_loss_prime)

                    intra_interf += self.total_power * self.P[u_prime, s] * np.abs(h_proj_prime) ** 2

                inter_interf = 0
                for s_prime in range(self.S):
                    if s_prime == s: continue
                    connected_users_sp = np.where(connection_matrix_sat[s_prime] == 1)[0]
                    for u_prime in connected_users_sp:

                        if s_prime not in user_sat_map[u_prime]:
                            continue

                        w_u_prime_sp = W_all[u_prime][:,
                                       user_sat_map[u_prime][s_prime]:user_sat_map[u_prime][s_prime] + 1]

                        a_rx_prime = self.steering_vector(self.Mr, self.theta_u[u_prime, s], self.phi_u[u_prime, s],
                                                          'rx')
                        a_tx_prime = self.steering_vector(self.Mt, self.theta_u[u_prime, s],
                                                          self.phi_u[u_prime, s], 'tx')

                        path_loss_prime = self.path_loss(self.dist_us[u_prime, s])

                        h_proj_los_prime = (v_su.conj().T @ a_rx_prime) * (a_tx_prime.conj().T @ w_u_prime_sp)

                        h_proj_nlos_prime = (np.random.randn() + 1j * np.random.randn()) * np.linalg.norm(
                            v_su) * np.linalg.norm(w_us) / np.sqrt(2)
                        h_proj_prime = np.sqrt(k_factor / (k_factor + 1)) * h_proj_los_prime + np.sqrt(
                            1 / (k_factor + 1)) * h_proj_nlos_prime
                        h_proj_prime *= np.sqrt(path_loss_prime)

                        inter_interf += self.total_power * self.P[u_prime, s_prime] * np.abs(h_proj_prime) ** 2

                jammer_interf = 0
                for j in range(self.J):
                    if self.mat_js[j] == s:
                        jammer_interf += self.jammer_power * np.abs(v_su.conj().T @ self.G[s, j]) ** 2

                total_interf = noise + intra_interf + inter_interf + jammer_interf
                SINR_matrix[u, s] = signal / (total_interf + 1e-16)

        return SINR_matrix

    def _get_k_factor(self, elevation):
        elevation_divide = [30, 60, 70, 80, 90]
        k_factors = [0, 3.9, 5.6, 9.8, 17.1]
        return k_factors[bisect_left(elevation_divide, elevation)]

    def _compute_spectral_efficiency(self, sinr):
        return np.log2(1 + sinr)

    def _compute_spectral_efficiency_real(self, sinr):
        se_real = np.zeros((self.U, self.S))
        for u in range(self.U):
            for s in range(self.S):
                e_us = (math.pi / 2 - cal_angle(self.user_positions[u], self.satellite_positions[s] - self.user_positions[u])) * 180 / math.pi
                effi = []
                for mc in range(13):
                    effi.append(env_reward_cal(sinr[u][s], mc, e_us))  # as for fixed MCS e_us is set to be 30
                se_real[u][s] = np.max(effi)

        return se_real
