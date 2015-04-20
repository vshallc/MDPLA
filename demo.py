import vrp.vrp as vv
v=vv.random_vrp(2,2,1)
m=vv.vrp2mdp(v)
m.value_iteration()