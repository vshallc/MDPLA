import matplotlib.pyplot as plt
import datetime
from scipy.stats import *
from dis_mdp.mdp import *
import time

t2n = lambda hh, mm: hh * 60 + mm


def assign_pdf_abs(states, sid, sid_out, tid, time_span_left, time_span_middle, time_span_right, pdf_scale):
    outcome_list = list()
    r = 1.0
    for j in range(time_span_left, time_span_right, 10):
        p = norm.pdf(j, loc=time_span_middle, scale=pdf_scale)
        outcome_list.append((p, states[sid_out][int(j / 10) - 42]))
        r -= p
    outcome_list.append((r, states[sid_out][int(time_span_right / 10) - 42]))
    return outcome_list


def assign_pdf_rel(states, sid, sid_out, tid, time_span_left, time_span_middle, time_span_right, pdf_scale):
    outcome_list = list()
    r = 1.0
    for j in range(time_span_left, time_span_right, 10):
        new_tid = int(j / 10) + tid
        p = norm.pdf(j, loc=time_span_middle, scale=pdf_scale)
        if new_tid < 43:
            s = states[sid_out][new_tid]
        else:
            s = states[sid_out][42]  # just link to the last state
        outcome_list.append((p, s))
        r -= p
    new_tid = int(time_span_right / 10) + tid
    if new_tid < 43:
        s = states[sid_out][new_tid]
    else:
        s = states[sid_out][42]  # just link to the last state
    outcome_list.append((r, s))
    # return p_list, s_list
    return outcome_list


def init_states():
    states = list()
    # 0 ~ 42 Home
    # 43 ~ 85 x2
    # 86 ~ 128 Work
    label_list = ['Home', ' x2 ', 'Work']
    for i in range(3):
        states.append([])
        for hh in range(7, 14):
            for mm in range(0, 60, 10):
                time_str = str(hh).zfill(2) + ':' + str(mm).zfill(2)
                states[i].append(State(label_list[i], time_str))
        states[i].append(State(label_list[i], '14:00'))
    for i in range(0, 42):
        states[0][i].add_action('dawdling', [(1.0, states[0][i + 1])])
        states[1][i].add_action('dawdling', [(1.0, states[1][i + 1])])
    time_span_left = t2n(9, 10)
    time_span_middle = t2n(9, 45)
    time_span_right = t2n(10, 20)
    for i in range(0, 6):  # 07:00(0) ~ 07:50(5)
        outcome_list = assign_pdf_abs(states, 0, 2, i, time_span_left, time_span_middle, time_span_right, 12.5)
        states[0][i].add_action('taking the 8am train', outcome_list)  # arriving at 09:10 ~ 10:20
    time_span_left = t2n(0, 30)
    time_span_middle = t2n(1, 30)
    time_span_right = t2n(2, 30)
    for i in range(0, 3):  # 07:00(0) ~ 07:20(2)
        # off peak
        outcome_list = assign_pdf_rel(states, 0, 1, i, time_span_left, time_span_middle, time_span_right, 25.5)
        states[0][i].add_action('driving to work via highway', outcome_list)  # time duration 0h30m ~ 2h30m
    for i in range(3, 6):  # 07:30(3) ~ 07:50(5)
        # prob of rush increasing from 07:20 (state 2) with prob 0.0 to 08:00 (state 6) with prob 1.0
        # the prob of rush hour/off peak for state 3,4,5 are 0.25/0.75,0.50/0.50, 0.75/0.25
        outcome_list = list()
        # rush hour
        time_span_left = t2n(0, 30)
        time_span_middle = t2n(2, 20)
        time_span_right = t2n(6, 00)
        outcome_list_tmp = assign_pdf_rel(states, 0, 1, i, time_span_left, time_span_middle, time_span_right, 25.5)
        outcome_list.extend([(p * 0.25 * (i - 2), s) for (p, s) in outcome_list_tmp])
        # off peak
        time_span_left = t2n(0, 30)
        time_span_middle = t2n(1, 30)
        time_span_right = t2n(2, 30)
        outcome_list_tmp = assign_pdf_rel(states, 0, 1, i, time_span_left, time_span_middle, time_span_right, 25.5)
        outcome_list.extend([(p * (1 - 0.25 * (i - 2)), s) for (p, s) in outcome_list_tmp])
        states[0][i].add_action('driving to work via highway', outcome_list)
    time_span_left = t2n(0, 30)
    time_span_middle = t2n(2, 20)
    time_span_right = t2n(6, 00)
    for i in range(6, 16):  # 08:00(6) ~ 09:30(15)
        # rush hour
        outcome_list = assign_pdf_rel(states, 0, 1, i, time_span_left, time_span_middle, time_span_right, 25.5)
        states[0][i].add_action('driving to work via highway', outcome_list)  # time duration 0h30m ~ 2h30m
    for i in range(16, 20):  # 09:40(16) ~ 10:10(19)
        outcome_list = list()
        # rush hour
        time_span_left = t2n(0, 30)
        time_span_middle = t2n(2, 20)
        time_span_right = t2n(6, 00)
        outcome_list_tmp = assign_pdf_rel(states, 0, 1, i, time_span_left, time_span_middle, time_span_right, 25.5)
        outcome_list.extend([(p * (1 - 0.25 * (i - 2)), s) for (p, s) in outcome_list_tmp])
        # off peak
        time_span_left = t2n(0, 30)
        time_span_middle = t2n(1, 30)
        time_span_right = t2n(2, 30)
        outcome_list_tmp = assign_pdf_rel(states, 0, 1, i, time_span_left, time_span_middle, time_span_right, 25.5)
        outcome_list.extend([(p * 0.25 * (i - 2), s) for (p, s) in outcome_list_tmp])
        states[0][i].add_action('driving to work via highway', outcome_list)
    time_span_left = t2n(0, 30)
    time_span_middle = t2n(1, 30)
    time_span_right = t2n(2, 30)
    for i in range(20, 43):  # 10:20(20) ~ 14:00(42)
        # off peak
        outcome_list = assign_pdf_rel(states, 0, 1, i, time_span_left, time_span_middle, time_span_right, 25.5)
        states[0][i].add_action('driving to work via highway', outcome_list)  # time duration 0h30m ~ 2h30m
    for i in range(0, 43):
        new_tid = i + int(t2n(1, 00) / 10)
        if new_tid < 43:
            states[1][i].add_action('driving on backroad', [(1.0, states[2][new_tid])])
        else:
            states[1][i].add_action('driving on backroad', [(1.0, states[2][42])])
    return states


def init_rewards(states):
    rewards = dict()
    for s_from in states:
        rewards[s_from] = dict()
        for s_to in states:
            if s_to.label == 'Work' and s_from.label != 'Work':
                time = t2n(*tuple([int(e) for e in s_to.time.split(':')]))
                if time < t2n(11, 00):
                    rewards[s_from][s_to] = 1.0  # +1 for arriving at work before 11:00
                elif time < t2n(12, 00):
                    rewards[s_from][s_to] = (t2n(12, 00) - time) / t2n(1, 00)  # falls linearly to zero (11:00 ~ 12:00)
                else:
                    rewards[s_from][s_to] = 0.0
            else:
                rewards[s_from][s_to] = 0.0
    return rewards


def main():
    # print('Initializing states...')
    start_time=time.time()
    for i in range(1):
        states = init_states()
        states = [s for sub in states for s in sub]
        # print('Initializing rewards...')
        rewards = init_rewards(states)
        # print('Initialization done')
        # for i in range(len(states)):
        #     print(str(i) + ' ' + str(states[i]))
        # states[48].print_detail()
        mdp = MDP(states, rewards, states[0], set(states[86:129]))
        u = value_iteration(mdp)
    print("--- %s seconds ---" % (time.time() - start_time))
    # for i in range(3 * 43):
    #     print(states[i], u[states[i]])
    # v = [u[s] for s in states[0:43]]
    # t = [s.time for s in states[0:43:6]]
    # # t = [datetime.time(hh, mm) for hh in range(7,14) for mm in range(0,60,10)]
    # # t.append(datetime.time(14, 00))
    # # print(v)
    # # print(t)
    # plt.plot(range(len(v)), v, 'ro')
    # plt.xticks(range(0, len(v), 6), t, size='small')
    # plt.show()

if __name__ == "__main__":
    main()