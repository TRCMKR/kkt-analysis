from importlib import reload
from collections import defaultdict
import funcs
reload(funcs)
from funcs import *

def check_sh_rm_codes(m, tau):
    res = []

    max_r = m
    for i in range(1, max_r + 1):
        for j in range(1, max_r + 1):
            if i + j > m - 1:
                continue
            rm1 = shorten_og_code(codes.BinaryReedMullerCode(i, m).generator_matrix(), tau)
            rm2 = shorten_og_code(codes.BinaryReedMullerCode(j, m).generator_matrix(), tau)

            if len(list(rm1)) == 0 or len(list(rm2)) == 0:
                continue
            exp_res = codes.LinearCode(shorten_og_code(codes.BinaryReedMullerCode(i + j, m), tau))
            act_res = codes.LinearCode(matr_hadamard_prod(rm1, rm2))

            if exp_res == act_res:
                res.append([i, j, exp_res, act_res])
            else:
                print(i, j, exp_res, act_res)

    return res

def check_sh_rm_codes_parallel(ms, deltass, expc, cores = maxcores):
    RMs = {}
    successes = defaultdict(lambda: 0)
    cnt = defaultdict(lambda: 0)

    def rm_key(r, m):
        return f"{r},{m}"

    def rm_key_delta(r1, r2, m, delta):
        return f"{r1},{r2},{m},{delta}"

    def check_prev_deltas(r1, r2, m, delta):
        k1 = rm_key_delta(r1, r2, m, delta - 1)
        k2 = rm_key_delta(r1, r2, m, delta - 2)
        k3 = rm_key_delta(r1, r2, m, delta - 3)
        # print(k1 in cnt.items(), k2 in cnt.items(), k3 in cnt.items(), successes.items(), cnt.items())
        return (k1 in cnt.items() and k2 in cnt.items() and k3 in cnt.items() and
                successes[k1] / cnt[k1] == 0.0 and successes[k2] / cnt[k2] == 0.0 and successes[k3] / cnt[k3] == 0.0)

    def generate_test_sets(ms, deltass):
        res = []
        inc = 0
        for i in range(len(ms)):
            for r1 in range(2, ms[i]):
                for r2 in range(r1, ms[i]):
                    if r1 + r2 > ms[i] - 1:
                        continue

                    for delta in deltass[i]:
                        key = rm_key(r1, ms[i])
                        if key not in RMs:
                            RMs[key] = codes.BinaryReedMullerCode(r1, m).generator_matrix()

                        key = rm_key(r2, ms[i])
                        if key not in RMs:
                            RMs[key] = codes.BinaryReedMullerCode(r2, m).generator_matrix()

                        key = rm_key(r1 + r2, ms[i])
                        if key not in RMs:
                            RMs[key] = codes.BinaryReedMullerCode(r1 + r2, m).generator_matrix()

                        res.append((inc, r1, r2, ms[i], delta))
                        inc += 1

        return res

    coordss = {}
    for m in ms:
        coordss[m] = list(range(0, 2 ** m))

    @parallel(ncpus=cores)
    def func(i, r1, r2, m, delta):
        coords = coordss[m]
        taus = [tuple(random.sample(coords, delta)) for _ in range(expc)]

        res = []

        for tau in taus:
            if check_prev_deltas(r1, r2, m, delta):
                continue

            rm1 = shorten_og_code(RMs[rm_key(r1, m)], tau)
            rm2 = shorten_og_code(RMs[rm_key(r2, m)], tau)

            exp_res = codes.LinearCode(shorten_og_code(RMs[rm_key(r1 + r2, m)], tau))
            act_res = codes.LinearCode(matr_hadamard_prod(rm1, rm2))

            is_suc = exp_res == act_res
            key = rm_key_delta(r1, r2, m, delta)
            if is_suc:
                successes[key] += 1

            cnt[key] += 1

            res.append([exp_res == act_res, r1, r2, m, tau, rm1, rm2, exp_res, act_res])

        return res

    inps = generate_test_sets(ms, deltass)

    res = {}
    for inp, out in tqdm(func(inps), total=len(inps), desc="Overall Progress"):
        res[inp[0][0]] = out

    return dict(sorted(res.items()))
