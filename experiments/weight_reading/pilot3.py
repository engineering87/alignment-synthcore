import copy, torch, gw, sys
from multiprocessing import Pool
def run(args):
    torch.set_num_threads(1); k, s = args
    a, c = gw.train("goal", s, updates=k, return_critic=True)
    init = (copy.deepcopy(a.state_dict()), copy.deepcopy(c.state_dict()))
    out = [k, round(gw.rollout(a, gw.goal_ood, 1024, seed=7)["p_goal"], 2)]
    for m in ["goal", "proxy"]:
        f = gw.train(m, s + 1, updates=150, init=init)
        out += [m, round(gw.rollout(f, gw.SAMPLERS[m], 1024, 1)["p_goal"], 2),
                round(gw.rollout(f, gw.goal_corner, 512, 1)["p_goal"], 2),
                gw.rollout(f, gw.goal_ood, 1024, seed=7)]
    return out
if __name__ == "__main__":
    with Pool(2) as p:
        for r in p.imap(run, [(40, 11), (80, 12), (120, 13), (120, 14)]): print(r, flush=True)
