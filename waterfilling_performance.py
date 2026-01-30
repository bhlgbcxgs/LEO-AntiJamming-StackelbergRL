# from SatelliteSimulationOri import *
# from SatelliteSimulation import *
# from SatelliteSimulationV2 import *
from linksimulation import *
import numpy as np
import copy
from satellite_param import *
from itertools import permutations


def user_sat_matching_greedy_wf(sim):
    user_counts = np.zeros(user_num, dtype=int)
    satellite_counts = np.zeros(sat_num, dtype=int)
    sim.mat_us = np.zeros((sim.S, sim.U))

    # greedy
    while np.any(user_counts < 2):
        max_reward = -np.inf
        best_user, best_satellite = -1, -1

        for u in range(sim.U):
            if user_counts[u] >= 2:
                continue
            for s in range(sim.S):
                if satellite_counts[s] >= 2:
                    continue
                if sim.mat_us[s, u] == 1:
                    continue
                sim.mat_us[s, u] = 1
                sim._water_filling_original()
                sinr = sim.calculate_sinr_optimized()
                reward_link = sim._compute_spectral_efficiency(sinr)[u, s]

                if reward_link > max_reward:
                    max_reward = reward_link
                    best_user, best_satellite = u, s

                sim.mat_us[s, u] = 0

        sim.mat_us[best_satellite, best_user] = 1
        user_counts[best_user] += 1
        satellite_counts[best_satellite] += 1

    return


def heuristic_jammer_mapping(sim):
    user_gain_proxy = 1.0 / (sim.dist_us ** 2 + 1e-6)

    sat_importance = np.max(user_gain_proxy, axis=0)

    preferred_sat_indices = np.argsort(sat_importance)[::-1]

    J_num = sim.J
    S_num = sim.S
    jammer_gains = np.zeros((J_num, S_num))

    for s in range(S_num):
        for j in range(J_num):
            g_vec = sim.G[s, j]  # shape (Mr, 1)
            gain_val = np.linalg.norm(g_vec) ** 2
            jammer_gains[j, s] = gain_val

    available_jammers = set(range(J_num))

    jammer_to_sat_mapping = np.zeros(J_num, dtype=int)

    for sat_idx in preferred_sat_indices:

        if not available_jammers:
            break

        candidates = list(available_jammers)

        current_gains = jammer_gains[candidates, sat_idx]

        best_candidate_local_idx = np.argmax(current_gains)
        best_jammer_id = candidates[best_candidate_local_idx]

        jammer_to_sat_mapping[best_jammer_id] = sat_idx

        available_jammers.remove(best_jammer_id)

    return jammer_to_sat_mapping


if __name__ == "__main__":

    link_num = [12, 13, 14, 14, 15, 20]
    satellite_num = [36, 12, 37, 38, 39, 46]
    starlink = [135.0, 146.3, 157.5, 157.5, 168.8, 225.0]
    mean_anomaly = [23.0, 204.9, 19.6, 19.6, 21.5, 31.1]
    v0 = 40 - 34.9
    v = 5
    phi = v * 108 / 1440
    sat_pos = []
    sat_num = 6
    user_num = 3
    jammer_num = 6
    for i in range(sat_num):
        sat_pos.append(
            ecef_position(v + v0 + mean_anomaly[i] - 7.2 * satellite_num[i], phi, rs=7528, w=0, omega=starlink[i],
                          i=53))

    perms = permutations([0, 1, 2, 3, 4, 5])
    perms = list(perms)
    mat_u_all = []
    mat_temp = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    sat_list = [0, 1, 2, 3, 4, 5]
    for i_u11 in range(sat_num):
        mat_temp[i_u11][0] = 1
        sat_list1 = sat_list[:i_u11] + sat_list[i_u11 + 1:]
        print("sat_list1", sat_list1)
        for i_u12 in sat_list1:
            mat_temp[i_u12][0] = 1
            for i_u21 in range(sat_num):
                mat_temp[i_u21][1] = 1
                sat_connect_cur = np.sum(mat_temp, axis=1)
                sat_va1 = np.where(sat_connect_cur < 2)[0]
                for i_u22 in range(len(sat_va1)):
                    if sat_va1[i_u22] == i_u21:
                        continue
                    mat_temp[sat_va1[i_u22]][1] = 1
                    sat_connect_cur = np.sum(mat_temp, axis=1)
                    sat_va2 = np.where(sat_connect_cur < 2)[0]
                    for i_u31 in range(len(sat_va2)):
                        mat_temp[sat_va2[i_u31]][2] = 1
                        sat_connect_cur = np.sum(mat_temp, axis=1)
                        sat_va3 = np.where(sat_connect_cur < 2)[0]
                        for i_u32 in range(len(sat_va3)):
                            if sat_va3[i_u32] == sat_va2[i_u31]:
                                continue
                            mat_temp[sat_va3[i_u32]][2] = 1
                            # print("mat_temp", mat_temp)
                            mat_add = copy.deepcopy(mat_temp)
                            mat_u_all.append(mat_add)
                            mat_temp[sat_va3[i_u32]][2] = 0
                        mat_temp[sat_va2[i_u31]][2] = 0
                    mat_temp[sat_va1[i_u22]][1] = 0
                mat_temp[i_u21][1] = 0
            mat_temp[i_u12][0] = 0
        mat_temp[i_u11][0] = 0

    utility_all_time = []
    for v in range(360):
        sim = SatelliteSimulation(mat_u_all[0], perms[0], v)
        utility_j = []
        mat_j = heuristic_jammer_mapping(sim)
        sim.mat_js = np.array(mat_j)
        user_sat_matching_greedy_wf(sim)
        sim._water_filling_original()
        SINR = sim.calculate_sinr_optimized()
        utility_j = np.sum(sim._compute_spectral_efficiency_real(SINR))

        print("v=", v, " ,greedy-method: best_j:", mat_j, ",final utility:", utility_j)
        utility_all_time.append(min(utility_j))

    print("----------\n[HEURISTIC]greedy-method-time-averaged:", np.average(np.array(utility_all_time)))

    utility_all_time = []
    for v in range(360):
        sim = SatelliteSimulation(mat_u_all[0], perms[0], v)
        utility_j = []
        n = np.random.randint(0, 720)
        mat_j = perms[n]
        sim.mat_js = np.array(mat_j)
        user_sat_matching_greedy_wf(sim)
        sim._water_filling_original()
        SINR = sim.calculate_sinr_optimized()
        utility_j = np.sum(sim._compute_spectral_efficiency_real(SINR))

        best_j = utility_j.index(min(utility_j))
        print("v=", v, " ,greedy-method: best_j:", mat_j, ",final utility:", utility_j)
        utility_all_time.append(min(utility_j))

    print("----------\n[RANDOM]greedy-method-time-averaged:", np.average(np.array(utility_all_time)))

    utility_all_time = []
    for v in range(360):
        sim = SatelliteSimulation(mat_u_all[0], perms[0], v)
        utility_j = []
        for i in range(len(perms)):
            mat_j = perms[i]
            sim.mat_js = np.array(mat_j)
            user_sat_matching_greedy_wf(sim)
            sim._water_filling_original()
            SINR = sim.calculate_sinr_optimized()
            utility_j.append(np.sum(sim._compute_spectral_efficiency_real(SINR)))

        best_j = utility_j.index(min(utility_j))
        print("v=", v, " ,greedy-method: best_j:", perms[best_j], ",final utility:", min(utility_j))
        utility_all_time.append(min(utility_j))

    print("----------\n[SMART]greedy-method-time-averaged:", np.average(np.array(utility_all_time)))