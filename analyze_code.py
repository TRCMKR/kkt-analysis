from funcs import *
import multiprocessing
import os
from collections import defaultdict

print("PART 0/4")

def convert(o):
    if isinstance(o, (Integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (set,)):
        return list(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

Parallelism().set(nproc=8)

args = sys.argv[1:]

if len(args) < 5:
    raise Exception('Должно быть минимум 5 аргументов: r m left right exp_count')
elif len(args) != 7:
    raise Exception('Должно быть 7 аргументов: r m left right exp_count part(1, 2, 3, 4) folder_to_update')

r = int(args[0])
if r < 0:
    raise Exception('Параметр r не должен быть отрицательным')

m = int(args[1])
if m < 0:
    raise Exception('Параметр m не должен быть отрицательным')
if m <= r:
    raise Exception('Параметр m должен быть больше r')

left = int(args[2])
if left < 0:
    raise Exception('Параметр left не должен быть отрицательным')

right = int(args[3])
if right < 0:
    raise Exception('Параметр right не должен быть отрицательным')
if right <= left:
    raise Exception('Параметр right должен быть больше left')

expc = int(args[4])
if expc < 0:
    raise Exception('Параметр expc не должен быть отрицательным')

part = 0
filename = ""
if len(args) > 5:
    part = int(args[5])
    filename = args[6]

step = 1
len_data = (right + 1 - left) * expc

def make_unique_dir(base_name):
    path = base_name
    counter = 1
    while os.path.exists(path):
        counter += 1
        path = f"{base_name}_N{counter}"
    os.makedirs(path)
    return path

if part <= 0:
    folder = make_unique_dir(f"RM({r}_{m})_L{left}_R{right}_E{expc}")
    print(f"Создана папка: {folder}")

    matrices_folder = make_unique_dir(f"{folder}/matrices")
    print(f"Создана папка для матриц: {matrices_folder}")
else:
    folder = filename
    matrices_folder = f"{folder}/matrices"

right += 1

print()
# -----------------------------

if part <= 1:
    print("PART 1/4")
    lnoisedcount = range(left, right, step)
    ln = []
    for cnt in lnoisedcount:
        tmp = range(0, cnt + 1)
        ln.append(tmp)

    print("Начинается подсчёт размеров халла")
    hulls = analyze_noised_hull_n(r, m, lnoisedcount, ln, expc)
    print("Успешно")

    print("Сохранение результатов")
    results = copy(hulls)

    jsave_data(results, folder+"/hulls.json")

    hull_graphs_folder = make_unique_dir(f"{folder}/hull_graphs")
    print(f"Создана папка для матриц: {hull_graphs_folder}")

    for key in results:
        tmp = results[key]
        show_stats(tmp, hull_graphs_folder, f"hulls_{key}", False)

    print("Результаты подсчёта халлов сохранены")

    print()
# -----------------------------
if part <= 2:
    print("PART 2/4")
    print("Начинается поиск кандидатов")
    candidates = find_noised_candidates(r, m, range(left, right, step), expc)
    print("Успешно")

    print("Сохранение результатов поиска кандидатов")
    candidates_dump = []
    i = 0
    for inp, out in candidates:
        candidates_dump.append([inp[0], i, list(out[0][1]), list(out[0][2])])
        save(out[0][0], matrices_folder + f"/{i}.sobj")
        i += 1

    with open(folder + "/candidates.json", "w") as f:
        json.dump(candidates_dump, f)

    results = []
    for inp, out in candidates:
        results.append([inp[0], out[0][0], out[0][1], out[0][2]])
    print("Результаты сохранены")

    print("Сохранение представления результатов")
    freq_map = defaultdict(lambda: defaultdict(int))

    for x in results:
        freq_map[x[0]][len(x[2]) / max(len(x[3]), 1)] += 1

    find_noised_candidates_res_freq = {k: dict(v) for k, v in freq_map.items()}

    with open(folder + "/candidates_freq.txt", "w") as f:
        for key in find_noised_candidates_res_freq:
            print(key, ":", find_noised_candidates_res_freq[key], end='\n', file=f)
    print("Представление сохранено")

    print()
# -----------------------------

tmp = []
with open(folder + "/candidates.json", "r") as f:
    tmp = json.load(f)

noisedcodes = [load(f'{matrices_folder}/{j}.sobj') for j in range(len_data)]
err_counts = [tmp[j][0] for j in range(len_data)]
coordss = [tmp[j][2] for j in range(len_data)]

if part <= 3:
    print("PART 3/4")

    print("Начинается очищение")
    get_noised_res = get_noised(noisedcodes, err_counts, coordss)
    print("Успешно")

    print("Сохранение результатов очищения")
    get_noised_res_dump = [list( (k, list(get_noised_res[k])) ) for k in get_noised_res]

    with open(folder + "/clearing.json", "w") as f:
        json.dump(get_noised_res_dump, f, default=convert)
    print("Результаты очищения сохранены")

    print("Сохранение представления очищения")
    freq_map = defaultdict(lambda: defaultdict(int))

    for i in range(len(get_noised_res)):
        freq_map[int(i / expc + left)][len(get_noised_res[i] & results[i][3]) / max(len(results[i][3]), 1)] += 1

    get_noised_freq = {k: dict(v) for k, v in freq_map.items()}

    with open(folder + "/clearing_freq.txt", "w") as f:
        for key in get_noised_freq:
            print(key, ":", get_noised_freq[key], end='\n', file=f)
    print("Представление сохранено")

    print()
# -----------------------------

print("PART 4/4")

print("Начинается дистилляция")
distillation_res = distillation(noisedcodes, err_counts, coordss)
print("Успешно")

print("Сохранение результатов дистилляции")
distillation_res_dump = [list( (k, list(distillation_res[k])) ) for k in distillation_res]

with open(folder + "/distillation.json", "w") as f:
    json.dump(distillation_res, f, default=convert)
print("Результаты дистилляции сохранены")

print("Сохранение представления дистилляции")
freq_map = defaultdict(lambda: defaultdict(int))

for i in range(len(distillation_res)):
    freq_map[int(i / expc + left)][len(set(distillation_res[i]) & results[i][3]) / max(len(results[i][3]), 1)] += 1

distillation_freq = {k: dict(v) for k, v in freq_map.items()}

with open(folder + "/distillation_freq.txt", "w") as f:
    for key in distillation_freq:
        print(key, ":", distillation_freq[key], end='\n', file=f)
print("Представление сохранено")

print()

print("Расчёты окончены!")