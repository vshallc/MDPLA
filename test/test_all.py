import matplotlib.pyplot as plt
import timeit
import exact_mdp
import exact_mdp.mdp
import exact_mdp.route_planning
import dis_mdp
import dis_mdp.mdp
import dis_mdp.route_planning

def test_1():
    dis_states = dis_mdp.route_planning.init_states()
    dis_states = [s for sub in dis_states for s in sub]
    dis_rewards = dis_mdp.route_planning.init_rewards(dis_states)
    disc_mdp = dis_mdp.mdp.MDP(dis_states, dis_rewards, dis_states[0], set(dis_states[86:129]))
    dis_u = dis_mdp.mdp.value_iteration(disc_mdp)

def test_2():
    ex_mdp = exact_mdp.route_planning.init_mdp()
    ex_u = exact_mdp.mdp.value_iteration(ex_mdp)

def main():
    dis_states = dis_mdp.route_planning.init_states()
    dis_states = [s for sub in dis_states for s in sub]
    dis_rewards = dis_mdp.route_planning.init_rewards(dis_states)
    disc_mdp = dis_mdp.mdp.MDP(dis_states, dis_rewards, dis_states[0], set(dis_states[86:129]))
    dis_u = dis_mdp.mdp.value_iteration(disc_mdp)
    dis_v = [dis_u[s] for s in dis_states[0:43]]
    dis_t = [s.time for s in dis_states[0:43:6]]


    ex_mdp = exact_mdp.route_planning.init_mdp()
    ex_u = exact_mdp.mdp.value_iteration(ex_mdp)
    # ex_t = ex_u[ex_mdp.states[0]].bounds
    ex_t = [x * 0.1 for x in range(70, 140, 2)]
    ex_v = [ex_u[ex_mdp.states[0]](tt) for tt in ex_t]

    # plt.plot(ex_t, ex_v)
    # plt.plot(range(len(dis_v)), dis_v, 'r.')
    # plt.xticks(range(0, len(dis_v), 6), dis_t, size='small')
    # plt.show()

    print('timeit')
    print(timeit.timeit('test_1()', setup='from __main__ import test_1', number=10))
    print(timeit.timeit('test_2()', setup='from __main__ import test_2', number=10))


if __name__ == '__main__':
    main()