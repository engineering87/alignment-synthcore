import torch, gw4, time
from multiprocessing import Pool
def run(args):
    torch.set_num_threads(1); p, q, s = args; t = time.time()
    a = gw4.train(p, q, s, updates=300)
    return (p, q, round(time.time() - t), gw4.rollout(a, gw4.sample_shared, 1024, 1), gw4.rollout(a, gw4.sample_ood, 2048, 7))
if __name__ == "__main__":
    cfg = [(1.0, 0.2, 1), (1.0, 0.5, 2), (1.0, 1.0, 3), (0.9, 0.5, 4), (0.7, 0.8, 5), (0.3, 1.0, 6)]
    with Pool(2) as p:
        for r in p.imap(run, cfg): print(r, flush=True)
