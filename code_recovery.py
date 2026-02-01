import random
from importlib import reload
import funcs
reload(funcs)
from funcs import *

def generate_cases(r, m, delta_min, delta_max, cnt):
    rm_base = codes.BinaryReedMullerCode(r, m).generator_matrix()
    coords = list(range(0, 2 ** m))

    cases = []
    for _ in range(cnt):
        perm = random.sample(range(rm_base.ncols()), rm_base.ncols())
        rm_perm = rm_base.matrix_from_columns(perm)

        tau = tuple(random.sample(coords, random.randint(delta_min, delta_max)))
        perm_tau = sorted([perm[t] for t in tau])

        cases.append((puncture_matr(rm_perm, perm_tau), rm_perm, tau, perm, perm_tau))

    return cases

def rm_get_perm(rm):
    perm = [0] * rm.ncols()
    for i in range(rm.ncols()):
        cnt = sum(int(rm[j][i]) * (2 ** (j - 1)) for j in range(1, rm.nrows()))
        perm[i] = cnt

    return perm

def matr_add_columns_at(matr, cols, tau):
    matr_list = [matr.column(i) for i in range(matr.ncols())]

    tau = sorted(tau)
    for idx, col in zip(tau, cols):
        matr_list.insert(idx, vector(GF(2), col))

    rm = Matrix(GF(2), matr.nrows(), len(matr_list))
    for i, col in enumerate(matr_list):
        rm.set_column(i, col)

    return rm

def get_inv_perm(perm):
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i

    return inv_perm

def matr_permute(matr, perm):
    og_rm = Matrix(GF(2), matr.nrows(), matr.ncols())

    for i in range(matr.ncols()):
        og_rm.set_column(i, matr.column(perm[i]))

    return og_rm

def matr_unpermute(matr, perm):
    og_rm = Matrix(GF(2), matr.nrows(), matr.ncols())

    for i in range(matr.ncols()):
        og_rm.set_column(perm[i], matr.column(i))

    return og_rm