from sage.all import *
from sage.modules.all import *
import random
import multiprocessing
import ast
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from collections import Counter
import plotly.graph_objects as go
import time
import plotly.offline as pyo  # Import plotly.offline for controlling output

import csv
import json
import numpy as np
from tqdm import tqdm
from math import comb
from itertools import combinations
import plotly.colors as pc

maxcores = multiprocessing.cpu_count()

def reed_muller_k(r, m):
    return sum(comb(m, i) for i in range(r + 1))

Parallelism().set(nproc=maxcores)

def generate_matrix(n, m, k):
    while True:
        A = random_matrix(GF(2), n, k)
        if A.rank() == k:
            break

    matrix = Matrix(GF(2), n, m)
    nonzero_columns = random.sample(range(m), k)

    for j, col in enumerate(nonzero_columns):
        for i in range(n):
            matrix[i, col] = A[i, j]

    return matrix, nonzero_columns

def apply_matrix_noise(matr, n):
    """Applies random noise to n columns

    Args:
        matr (_type_): Matrix to noise
        n (int): How many columns

    Returns:
        tuple: Resulting matrix, noised coords
    """    
    noise_matrix, noisy_columns = generate_matrix(matr.nrows(), matr.ncols(), n)

    res = copy(matr)
    res += noise_matrix
    
    return Matrix(GF(2), res), set(noisy_columns)

def matr_hadamard_prod(matr1, matr2):
    res = list()

    for i in range(0, matr1.nrows()):
        for j in range(0, matr2.nrows()):
            res.append([matr1[i, ind] * matr2[j, ind] for ind in range(matr1.ncols())])

    return Matrix(GF(2), res)
    
def matrix_hadamard_sqr(matr):
    """Calculates Hadamard square of matrix

    Args:
        matr (_type_): Input matrix

    Returns:
        Matrix: Squared input matrix
    """
    res = list()
    for i in range(0, matr.nrows()):
        for j in range(0, matr.nrows()):
            res.append([matr[i, ind] * matr[j, ind] for ind in range(matr.ncols()) ])
    
    return Matrix(GF(2), res)

def analyze_prmc_dims(r, m, ln, exp = 10, cores = maxcores):
    """Analyzes dimensions for squared pRM codes and squared noised pRM

    Args:
        r (int): Stands for Reed
        m (int): Stands for Muller
        ln (iterable): Contains how many coords will be noised and how many random would be punctured
        exp (int, optional): Number of experiments. Defaults to 10.
        cores (int, optional): How many cores to use. Defaults to 8.

    Returns:
        list: returns list of pairs (input, output for that input)
    """
    c1 = codes.BinaryReedMullerCode(r, m)
    c2 = codes.BinaryReedMullerCode(r * 2, m)

    @parallel(ncpus=cores)
    def func(n):
        (genmatr, noisedcoords) = apply_matrix_noise(c1.generator_matrix(), n)
        genmatr = matrix_hadamard_sqr(genmatr)
        
        res = list()
        for x in range(exp):
            coords = set(random.sample(range(2**m), n))
            
            cp = codes.PuncturedCode(c2, coords)
            noisecp = codes.PuncturedCode(codes.LinearCode(Matrix(GF(2), genmatr)), coords)
            k = c1.dimension()

            res.append( [cp.dimension(), noisecp.dimension(), len(noisedcoords.intersection(coords))] )
            
        return res
    
    tasks = [ x for x in ln ]
    res = []
    for inp, out in func(tasks): res.append( (inp[0], out) )
        
    return res

def code_hull(C):
    """Calculates hull of the code

    Args:
        x (code): Input code

    Returns:
        code: Hull of that code
    """
    if 'matrix' in str(type(C)): C = codes.LinearCode(C)
    
    Cdual = C.dual_code()
    
    matr = Matrix(C.basis()).stack(Matrix(Cdual.basis()))
    hull = codes.LinearCode(matr).dual_code()
    
    return hull

def gen_n_noised_coords(noised, size, n, c=0):
    """Generates c noised coords and n coords up to size

    Args:
        noised (iterable): Noised coords
        size (int): How many columns in your matrix
        n (int): How many coords to generate
        count (int, optional): How many coords from n should be noised. Defaults to 0.

    Returns:
        set: Set of c noised coords and n - c not noised
    """
    coords = set(random.sample(sorted(noised), c))
        
    while (len(coords) != n): 
        num = randint(0, size)
        if num in noised: continue
        coords.add(num)
    
    return coords

def analyze_noised_hull(r, m, lnoisedcount, exp = 10, cores = 8):
    """Analyzes hull of RM code for different numbers of noised coords

    Args:
        r (int): Stands for Reed
        m (int): Stands for Muller
        lnoisedcount (iterable): Numbers of noised coords
        exp (int, optional): Number of experiments. Defaults to 10.
        cores (int, optional): How many coords to use. Defaults to 8.

    Returns:
        list: List of pairs (input, [Dimension of pRM, Dimension of clean pRM hull,\n
        Dimension of noised pRM hull, How many noised coords deleted])
    """
    c = codes.BinaryReedMullerCode(r, m)
    
    @parallel(ncpus=cores)
    def func(noisedcount, deletednoised):
        (genmatr, noisedcoords) = apply_matrix_noise(c.generator_matrix(), noisedcount)
        # genmatr = hadamardsqr(genmatr)
        res = list()
        
        for x in range(exp):
            coords = gen_n_noised_coords(noisedcoords, 2**m - 1, noisedcount, deletednoised)
            
            cp = codes.PuncturedCode(c, coords)
            dims = code_hull(Matrix(cp.basis())).dimension()
            noisecp = codes.PuncturedCode(codes.LinearCode(Matrix(GF(2), genmatr)), coords)
            dimsnoised = code_hull(Matrix(noisecp.basis())).dimension()
            
            res.append( [cp.dimension(), dims, dimsnoised, len(noisedcoords.intersection(coords))] )
            
        return res
    
    tasks = [ (x, i) for x in lnoisedcount for i in range(x + 1)]
    
    res = []
    for inp, out in func(tasks): res.append( (inp[0], out) )
        
    return res

def jsave_data(data, fname, indent=0):
    """Dump compatible data into json

    Args:
        data (_type_): Data to dump
        fname (string): Where to dump
    """
    with open(fname, 'w') as file:
        if indent > 0:
            json.dump(data, file, indent=indent)
        else:
            json.dump(data, file)

def jload_data(fname):
    """Load data from json

    Args:
        fname (string): Where to get data
        
    Returns:
        list: Loaded data
    """
    with open(fname, 'r') as file:
        data = json.load(file)
        return data
    
def printd(data):
    res_sorted = sorted(data, key=lambda x: x[0])
    for inputs, output in res_sorted:
        print(f"Inputs: {inputs}, Output: {output}")
        
def analyze_noised_hull_n(r, m, lnoisedcount, ln=list(), exp=10, cores=maxcores):
    """Analyzes hull of RM code for different numbers of noised coords\n
    where ln controls how many will be deleted for each of noisedcounts

    Args:
        r (int): Stands for Reed
        m (int): Stands for Muller
        lnoisedcount (iterable): Numbers of noised coords
        ln (iterable): 2D-List for each of noised coords count\n
        how many of them should be deleted
        exp (int, optional): Number of experiments. Defaults to 10.
        cores (int, optional): How many coords to use. Defaults to 8.

    Returns:
        list: List of pairs (input, [Dimension of pRM, Dimension of clean pRM hull,\n
        Dimension of noised pRM hull, How many noised coords deleted])
    """
    c = codes.BinaryReedMullerCode(r, m)
    
    @parallel(ncpus=cores)
    def func(noisedcount, n, deletednoised):
        # cdual = codes.BinaryReedMullerCode(m - r - 1, m)
        
        (genmatr, noisedcoords) = apply_matrix_noise(c.generator_matrix(), noisedcount)
        # genmatr = hadamardsqr(genmatr)
        res = list()
        
        for x in range(exp):
            coords = gen_n_noised_coords(noisedcoords, 2**m - 1, n, deletednoised)
            
            cp = codes.PuncturedCode(c, coords)
            dims = code_hull(Matrix(cp.basis())).dimension()
            noisecp = codes.PuncturedCode(codes.LinearCode(Matrix(GF(2), genmatr)), coords)
            dimsnoised = code_hull(Matrix(noisecp.basis())).dimension()
            
            res.append( [cp.dimension(), dims, dimsnoised, len(noisedcoords.intersection(coords))] )
            
        return res
    
    tasks = []
    if (len(ln) == 0): 
        for x in lnoisedcount: ln.append(list(range(x + 1)))
        
    ldeletednoised = dict()
    for x in ln:
        for elem in x:
            ldeletednoised[elem] = list(range(elem + 1))
    
    inps = []
    ind = 0
    for cnt in lnoisedcount:
        inps.extend( [ (cnt, i, j) for i in ln[ind] for j in ldeletednoised[i]] )
        ind += 1

    res = {}
    for inp, out in tqdm(func(inps), total=len(inps), desc="Overall Progress"):
        key = inp[0][0]
        if key not in res:
            res[key] = []
        res[key].append((inp[0][1:], out))

    return res

def make_pairs(res, i, j, k):
    pairs = []
    # print(results[0][0][0])
    for inputs, outputs in res:
        input_3rd = inputs[i]  # 3rd input value (Z-axis)
        input_4th = inputs[j]  # 4th input value
        for output in outputs:
            output_4th = output[k]  # 4th output value
            pairs.append((output_4th, input_3rd, input_4th))
            
    return pairs
            
def draw_graph(pairs, wkdir="", data_name="", open=True):
    # Step 2: Count occurrences of each (input_4th, output_4th) pair
    triplet_counts = Counter(pairs)
    
    # Separate the values and counts for plotting
    z_values = [triplet[0] for triplet in triplet_counts.keys()]
    x_values = [triplet[1] for triplet in triplet_counts.keys()]
    y_values = [triplet[2] for triplet in triplet_counts.keys()]
    sizes = [count for count in triplet_counts.values()]  # Scale the count for plot size
    
    min_size = 1
    max_size = 20
    max_count = max(sizes) if sizes else 1  # Prevent division by zero
    
    # Scale sizes between min_size and max_size
    normalized_sizes = [min_size + (max_size - min_size) * (count / max_count) for count in sizes]

    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    min_z, max_z = min(z_values), max(z_values)

    # Ensure axis ranges start at 0 or the minimum positive value
    # min_x = max(min_x, 0)
    # min_y = max(min_y, 0)
    # min_z = max(min_z, 0)

    if len(set(sizes)) == 1:
        thermal_colors = pc.get_colorscale('thermal')

        max_color = thermal_colors[-1][1]
        # Все значения одинаковые — фиксированный цвет, убираем шкалу
        marker = dict(
            size=normalized_sizes,
            color=max_color,
            opacity=0.7,
            line=dict(width=0.8, color='black')
        )
    else:
        # Значения разные — нормальная цветовая шкала
        marker = dict(
            size=normalized_sizes,
            color=sizes,
            colorscale='thermal',
            cmin=min(sizes),
            cmax=max(sizes),
            colorbar=dict(title='Frequency'),
            opacity=0.7,
            line=dict(width=0.8, color='black')
        )

    fig = go.Figure(data=[go.Scatter3d(
        x=x_values,
        y=y_values,
        z=z_values,
        mode='markers',
        marker=marker
    )])

    # Step 4: Update layout with more descriptive titles and a cleaner design
    fig.update_layout(
        scene=dict(
            xaxis_title='Noised count',
            yaxis_title='Noised deleted',
            zaxis_title='Hull dimensions',
            xaxis=dict(
                tickmode='linear', 
                tickfont=dict(size=12),
                range=[min_x, max_x]  # Explicitly set the range for the X-axis
            ),
            yaxis=dict(
                tickmode='linear', 
                tickfont=dict(size=12),
                range=[min_y, max_y]  # Explicitly set the range for the Y-axis
            ),
            zaxis=dict(
                tickmode='linear', 
                tickfont=dict(size=12),
                range=[0, max_z]  # Explicitly set the range for the Z-axis
            ),
        ),
        # title='3D Scatter Plot of 3rd & 4th Input vs. 4th Output with Frequency',
        # title_font=dict(size=20, color='darkblue'),
        margin=dict(l=0, r=0, b=0, t=40),  # Adjust margins for a cleaner view
        scene_bgcolor='rgba(255, 255, 255, 0.9)',  # Set the background color of the scene
        paper_bgcolor='rgb(243, 243, 243)',  # Set background color outside the 3D plot
        hoverlabel=dict(
            bgcolor='white',
            font_size=14,
            font_family='Arial'
        ),
    )

    # Step 5: Open the plot in a new browser window
    filename = ""
    if data_name == "":
        filename = f'3d_plot.html'
    else:
        filename = f'3d_{data_name}_plot.html'

    if wkdir != "":
        filename = f'{wkdir}/{filename}'

    pyo.plot(fig, filename=filename, auto_open=open)  # This will open the plot in a new tab in the browser
    
def show_stats(res, wkdir="", data_name="", auto_open=True):
    # pairs = make_pairs(res, 2, 3)
    # pairs = make_pairs(res, 1, 2, 2)
    pairs = make_pairs(res, 0, 1, 2)

    draw_graph(pairs, wkdir, data_name, auto_open)

def show_stats2(res, wkdir="", data_name="", auto_open=True):
    # pairs = make_pairs(res, 3, 4)
    pairs = make_pairs(res, 2, 3, 3)

    draw_graph(pairs, wkdir, data_name, auto_open)
    
@parallel(ncpus=maxcores)
def find_coords_sqr(r, m, noisedcount, count, exp):
    c = codes.BinaryReedMullerCode(r, m)
    
    (genmatr, noisedcoords) = apply_matrix_noise(c.generator_matrix(), noisedcount)
    genmatr = matrix_hadamard_sqr(genmatr)
    cp = codes.PuncturedCode(c, noisedcoords)
    dims = genmatr.rank()
    
    res = list()
    
    for x in range(exp):
        (genmatr, noisedcoords) = apply_matrix_noise(c.generator_matrix(), noisedcount)
        code = codes.LinearCode(Matrix(GF(2), genmatr))
        
        coords = set()
        temp = 0
        
        for i in range(2**m):
            noisecp = codes.PuncturedCode(code, i)
            noisecp = matrix_hadamard_sqr(Matrix(noisecp.basis()))
            dimscp = noisecp.rank()
            
            if (dimscp < dims): 
                coords.add(i)
                temp = dimscp
            if (len(coords) >= count): break
        
        res.append( [coords, noisedcoords] )
        print(coords, " | ", noisedcoords, " = ", coords & noisedcoords)
        # print(genmatr)
        # print("---")
        print(c.generator_matrix(), "\n")
    
    return res
    
def find_coords(r, m, noisedcounts, exp = 10, cores = maxcores):
    c = codes.BinaryReedMullerCode(r, m)
    
    @parallel(ncpus=cores)
    def func(noisedcount, exp):     
        res = list()
        
        (genmatr, noisedcoords) = apply_matrix_noise(c.generator_matrix(), noisedcount)
        code = codes.LinearCode(Matrix(GF(2), genmatr))
        dims = code_hull(Matrix(GF(2), genmatr)).dimension()
        
        coords = set()
        for i in range(2**m):
            noisecp = codes.PuncturedCode(code, i)
            # noisecp = hadamardsqr(noisecp)
            hull = code_hull(Matrix(noisecp.basis()))
            dimscp = hull.dimension()
            if (dimscp > dims): 
                coords.add(i)
        
        res.append( [code, coords, noisedcoords] )
        
        return res
    
    inps = [(j, i) for j in noisedcounts for i in range(exp)]
    res = list()
    for inp, out in tqdm(func(inps), total=len(inps), desc="Overall Progress"):
        res.append((inp[0], out))
    
    return res

def get_noised(mycodes, errcounts, coordss, cores=maxcores):
    @parallel(ncpus=cores)
    def func(i, code, errcount, coords):
        if len(coords) <= errcount:
            return coords
        init_len = errcount

        coordslist = sorted(coords)
        temp = set(coordslist[0: init_len])
        res = temp.copy()
        tempcode = codes.PuncturedCode(code, temp)

        maxdim = code_hull(Matrix(tempcode.basis())).dimension()

        for j in range(init_len, len(coordslist)):
            newres = res.copy()
            for x in res:
                temp = newres.copy()
                temp.remove(x)
                temp.add(coordslist[j])
                tempcode = codes.PuncturedCode(code, temp)
                dims = code_hull(Matrix(tempcode.basis())).dimension()
                if (dims > maxdim):
                    newres.remove(x)
                    newres.add(coordslist[j])
                    maxdim = dims
                    break
            res = newres.copy()

        return res

    inps = [(i, mycodes[i], errcounts[i], coordss[i]) for i in range(len(mycodes))]

    res = {}
    for inp, out in tqdm(func(inps), total=len(inps), desc="Overall Progress"):
        res[inp[0][0]] = out

    return res

def get_noised2(rights, mycodes, errcounts, coordss, cores=maxcores):
    
    @parallel(ncpus=cores)
    def func(i, right, code, errcount, coords):
        if len(coords) <= errcount:
            return coords
        init_len = errcount
        # elif 2 * init_len < errcount:
        #     init_len *= 2
        #
        coordslist = sorted(coords)
        # res = coordslist[:init_len]
        temp = set(coordslist[0: init_len])
        res = temp.copy()
        tempcode = codes.PuncturedCode(code, temp)
        maxdim = code_hull(Matrix(tempcode.basis())).dimension()
        #
        # ind = 0
        # for i in range(init_len, len(coordslist)):
        #     temp.add(coordslist[i])
        #
        #     tempcode = codes.PuncturedCode(code, temp)
        #     dims = code_hull(Matrix(tempcode.basis())).dimension()
        #
        #     if (dims > maxdim):
        #         res.add(coordslist[i])
        #         maxdim = dims
        #     else:
        #         temp.remove(coordslist[i])
        #
        #     if len(res) == errcount:
        #         # maxdim = dim
        #         ind = i
        #         break
        #
        # ind += 1
        for j in range(init_len, len(coordslist)):
            newres = res.copy()
            for x in res:
                temp = newres.copy()
                temp.remove(x)
                temp.add(coordslist[j])
                tempcode = codes.PuncturedCode(code, temp)
                dims = code_hull(Matrix(tempcode.basis())).dimension()
                if (dims > maxdim):
                    newres.remove(x)
                    newres.add(coordslist[j])
                    maxdim = dims
                    break
            res = newres.copy()

        return res

    inps = [(i, rights[i], mycodes[i], errcounts[i], coordss[i]) for i in range(len(mycodes))]

    res = {}
    for inp, out in tqdm(func(inps), total=len(inps), desc="Overall Progress"):
        res[inp[0][0]] = out

    return res

# from sage.all import *
# from sage.modules.all import *
# import random
#
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
# from collections import Counter
# import plotly.graph_objects as go
# import plotly.offline as pyo  # Import plotly.offline for controlling output
# from collections import defaultdict
# import json
#
# import csv
# import json
# import numpy as np
#
# import funcs
# from importlib import reload
#
# reload(funcs)
# from funcs import *
#
# Parallelism().set(nproc=8)
#
# r = 2
# m = 8
# n = 8
# left = 4
# right = 15
# step = 1
# expc = 30
#
# len_data = (right - left) * expc
#
# find_coords_res = []
#
# with open("find_coords_res_dump.json", "r") as f:
#     find_coords_res_dump = json.load(f)
#
# for i in range(len(find_coords_res_dump)):
#     find_coords_res.append([find_coords_res_dump[i][0], load(f"matrices2/{find_coords_res_dump[i][1]}.sobj"),
#                             set(find_coords_res_dump[i][2]), set(find_coords_res_dump[i][3])])
#
# print(find_coords_res)
#
# freq_map = defaultdict(lambda: defaultdict(int))
#
# for x in find_coords_res:
#     freq_map[x[0]][len(x[2] & x[3]) / max(len(x[3]), 1)] += 1
#
# find_coords_freq = {k: dict(v) for k, v in freq_map.items()}
#
# with open("find_coords_freq.json", "w") as f:
#     json.dump(find_coords_freq, f)
#
# noisedcodes = [load(f'matrices2/{j}.sobj') for j in range(len_data)]
#
# err_counts = [find_coords_res[j][0] for j in range(len_data)]
# coordss = [find_coords_res[j][2] for j in range(len_data)]
#
#
# find_coords_res = [find_coords_res[i][3] for i in range(len(find_coords_res))]
# offset_cnt = 0
# offset = (offset_cnt) * expc
# print(offset)
# print(len_data)
# results = get_noised2(find_coords_res[offset:], noisedcodes[offset:], err_counts[offset:], coordss[offset:])

def find_noised_candidates(r, m, noisedcounts, exp=10, cores=maxcores):
    c = codes.BinaryReedMullerCode(r, m)
    k = reed_muller_k(r, m)

    @parallel(ncpus=cores)
    def func(noisedcount, exp):
        res = list()

        (genmatr, noisedcoords) = apply_matrix_noise(c.generator_matrix(), noisedcount)
        code = codes.LinearCode(Matrix(GF(2), genmatr))
        bound = k - 2 * noisedcount

        coords = set()
        for i in range(2**m):
            noisecp = codes.PuncturedCode(code, i)
            # noisecp = hadamardsqr(noisecp)
            hull = code_hull(Matrix(noisecp.basis()))
            dimscp = hull.dimension()
            if (dimscp > bound): # было >=
                coords.add(i)

        res.append([code, coords, noisedcoords])

        return res

    inps = [(j, i) for j in noisedcounts for i in range(exp)]
    res = list()
    for inp, out in tqdm(func(inps), total=len(inps), desc="Overall Progress"):
        res.append((inp[0], out))

    return res

def distillation(mycodes, errcounts, coordss, tau = 3, T = 2, cores = maxcores):
    def generate_test_sets(tau_prime, g, w, N):
        base_set = list(set(tau_prime) - {g})
        comb_iter = combinations(base_set, w)

        result = []
        for i, combo in enumerate(comb_iter):
            if i < N:
                result.append(combo)
            else:
                break

        return result

    @parallel(ncpus=cores)
    def func(i, code, errcount, coords):
        coordslist = sorted(coords)
        excess = set()

        if len(coords) == errcount:
            return coordslist

        for j in range(0, len(coordslist)):
            x = coordslist[j]
            tests = generate_test_sets(coords, x, errcount - 1, errcount)
            v = 0
            for test in tests:
                wi = codes.PuncturedCode(code, list(test))
                wig = codes.PuncturedCode(code, list(set(test) | {x}))
                if (code_hull(Matrix(wi.basis())).dimension() - code_hull(Matrix(wig.basis())).dimension()) == 1:
                    v += 1

            if v >= len(tests) / 2 + 1:
                excess.add(x)

        return set(coordslist) - excess

    inps = [(i, mycodes[i], errcounts[i], coordss[i]) for i in range(len(mycodes))]

    res = {}
    for inp, out in tqdm(func(inps), total=len(inps), desc="Overall Progress"):
        res[inp[0][0]] = out

    return res

def distillation2(rights, mycodes, errcounts, coordss, tau = 3, T = 2, cores = maxcores):
    def generate_test_sets(tau_prime, g, w, N):
        base_set = list(set(tau_prime) - {g})
        comb_iter = combinations(base_set, w)

        result = []
        for i, combo in enumerate(comb_iter):
            if i < N:
                result.append(combo)
            else:
                break

        return result

    @parallel(ncpus=cores)
    def func(i, right, code, errcount, coords):
        coordslist = sorted(coords)
        excess = set()

        if len(coords) == errcount:
            return coordslist

        for j in range(0, len(coordslist)):
            x = coordslist[j]
            tests = generate_test_sets(coords, x, errcount - 1, errcount)
            v = 0
            for test in tests:
                wi = codes.PuncturedCode(code, list(test))
                wig = codes.PuncturedCode(code, list(set(test) | {x}))
                if (code_hull(Matrix(wi.basis())).dimension() - code_hull(Matrix(wig.basis())).dimension()) == 1:
                    v += 1

            if v >= len(tests) / 2 + 1:
                excess.add(x)

        return set(coordslist) - excess

    inps = [(i, rights[i], mycodes[i], errcounts[i], coordss[i]) for i in range(len(mycodes))]

    res = {}
    for inp, out in tqdm(func(inps), total=len(inps), desc="Overall Progress"):
        res[inp[0][0]] = out

    return res

# from sage.all import *
# from sage.modules.all import *
# import random
#
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
# from collections import Counter
# import plotly.graph_objects as go
# import plotly.offline as pyo  # Import plotly.offline for controlling output
# from collections import defaultdict
# import json
#
# import csv
# import json
# import numpy as np
#
# import funcs
# from importlib import reload
#
# reload(funcs)
# from funcs import *
#
# Parallelism().set(nproc=8)
#
# def convert(o):
#     if isinstance(o, (Integer,)):
#         return int(o)
#     if isinstance(o, (np.floating,)):
#         return float(o)
#     if isinstance(o, (np.ndarray,)):
#         return o.tolist()
#     if isinstance(o, (set,)):
#         return list(o)
#     raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
#
# r = 2
# m = 7
# n = 8
# left = 4
# right = 15
# step = 1
# expc = 10
#
# len_data = (right - left) * expc
#
# find_noised_candidates_res = []
# with open("find_noised_candidates_res_dump.json", "r") as f:
#     find_noised_candidates_res_dump = json.load(f)
#
# for i in range(len(find_noised_candidates_res_dump)):
#     find_noised_candidates_res.append([int(find_noised_candidates_res_dump[i][0]), load(f"matrices4/{find_noised_candidates_res_dump[i][1]}.sobj"), set(find_noised_candidates_res_dump[i][2]), set(find_noised_candidates_res_dump[i][3])])
#
#
# rights = [find_noised_candidates_res[i][3] for i in range(len_data)]
# noisedcodes = [load(f'matrices4/{j}.sobj') for j in range(len_data)]
# err_counts = [find_noised_candidates_res[j][0] for j in range(len_data)]
# coordss = [find_noised_candidates_res[j][2] for j in range(len_data)]
#
# offset_cnt = 9
# offset = (offset_cnt) * expc
#
# get_noised_res = get_noised2(rights[offset:], noisedcodes[offset:], err_counts[offset:], coordss[offset:])
#
# get_noised_res_dump = [list( (k, list(get_noised_res[k])) ) for k in get_noised_res]
#
# with open("get_noised_res.json", "w") as f:
#     json.dump(get_noised_res_dump, f, default=convert)
#
# freq_map = defaultdict(lambda: defaultdict(int))
#
# for i in range(len(get_noised_res)):
#     freq_map[int(i / expc + left)][len(get_noised_res[i] & find_noised_candidates_res[i][3]) / max(len(find_noised_candidates_res[i][3]), 1)] += 1
#
# get_noised_freq = {k: dict(v) for k, v in freq_map.items()}
#
# for key in get_noised_freq:
#     print(key, ":", get_noised_freq[key], end='\n')

def grCn(s, title, rm):
    with open(s, "r", encoding="utf-8") as f:
        raw_text = f.read()

    data = {}

    for line in raw_text.strip().split('\n'):
        key_str, val_str = line.split(':', 1)
        key = int(key_str.strip())
        val_dict = ast.literal_eval(val_str.strip())
        data[key] = val_dict

    x = sorted(data.keys())
    res = []
    total_counts = []
    avg_values = []
    count_1s = []
    delta_s = []

    for k in x:
        inner = data[k]
        total = sum([k * inner_k * inner[inner_k] for inner_k in inner])

        exp_c = sum(inner.values())
        res.append(k * exp_c)

        weighted_avg = sum([(k * inner_k * inner[inner_k]) for inner_k in inner]) / (exp_c)

        count_1 = inner.get(1.0, 0) * k

        delta_s.append(k)
        total_counts.append(total)
        avg_values.append(weighted_avg)
        count_1s.append(count_1)

    plt.plot(x, total_counts, label="Найдено кандидатов", color="blue")
    plt.plot(x, count_1s, label=r"Количество столбцов, когда найдено ровно $\delta$", color="red")
    plt.plot(x, res, label="Всего грязных", color="green")
    plt.xlabel(r"$\delta$", fontsize=14)
    plt.ylabel("Число координат")
    plt.xticks(range(min(x), max(x) + 1))
    plt.legend()
    plt.grid(True)
    plt.show()

def grCl(s, title, rm):
    with open(s, "r", encoding="utf-8") as f:
        raw_text = f.read()

    data = {}

    for line in raw_text.strip().split('\n'):
        key_str, val_str = line.split(':', 1)
        key = int(key_str.strip())
        val_dict = ast.literal_eval(val_str.strip())
        data[key] = val_dict

    # Подготовка
    x = sorted(data.keys())
    res = []
    total_counts = []
    avg_values = []
    count_1s = []
    delta_s = []

    for k in x:
        inner = data[k]
        total = sum([k * inner_k * inner[inner_k] for inner_k in inner]) + 0.0

        exp_c = sum(inner.values())
        res.append(k * exp_c)

        weighted_avg = sum([(k * inner_k * inner[inner_k]) for inner_k in inner]) / (exp_c)

        count_1 = inner.get(1.0, 0) * k

        delta_s.append(k)
        total_counts.append(total)
        avg_values.append(weighted_avg)
        count_1s.append(count_1)

    # Левый график
    plt.plot(x, total_counts, label="Грязных после очищения", color="blue")
    plt.plot(x, count_1s, label="Количество успешно очищенных", color="red")
    plt.plot(x, res, label="Всего грязных", color="green")
    plt.xlabel(r"$\delta$", fontsize=14)
    plt.ylabel("Число координат")
    plt.xticks(range(min(x), max(x) + 1))
    plt.legend()
    plt.grid(True)
    plt.show()

def gr(s, title, rm):
    grCn(s + "/candidates_freq.txt", "Алгоритм 1", "")
    grCl(s + "/distillation_freq.txt", "Алгоритм 2", "")
    grCl(s + "/clearing_freq.txt", "Алгоритм 3", "")

# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------

def puncture_matr(matr, tau):
    cols_to_keep = [i for i in range(matr.ncols()) if i not in tau]

    return matrix(matr.base_ring(), [[row[j] for j in cols_to_keep] for row in matr.rows()])

def shorten_code(gen_matr, tau):
    pct_gen_matr = puncture_matr(gen_matr, tau)

    return codes.LinearCode(pct_gen_matr).dual_code().generator_matrix()

def shorten_og_code(gen_matr, tau):
    dual_gen_matr = codes.LinearCode(gen_matr).dual_code().generator_matrix()

    return shorten_code(dual_gen_matr, tau)
