from funcs import *



# res = analyze_noised_hull(3, 7, range(10))

# show_stats(res)

# res = analyze_prmc_dims(3, 8, range(40), 10)

# printd(res)


# t = matrix(GF(2), [[1, 1, 1, 0],
#                    [0, 1, 0, 0]])

# print(type(t))
# print(t)

# print(code_hull(t).dual_code().dimension())


res = jload_data("rm3_7_30_30_30_100.json")

show_stats2(res)