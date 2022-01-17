import hmm

def powerset(ls):
    if len(ls) == 0:
        return [[]]
    else:
        subsets = powerset(ls[1:])
        return [[ls[0]] + subset for subset in subsets] + subsets

def cartesian_product(ls1, ls2):
    ordered_pairs = []
    for elem1 in ls1:
        for elem2 in ls2:
            ordered_pairs.append((elem1, elem2))
    return ordered_pairs

# for obs_types, cov_types in cartesian_product(powerset(['turning_angle', 'speed', 'depth']), powerset(['temperature', 'depth'])):
#     # if obs_types == [] or ('depth' in obs_types and 'depth' in cov_types):
#     #     continue
#     if obs_types == []:
#         continue
#     print(obs_types, cov_types)
#     hmm.main('../data/2020-21/2020-21-477-long-15-min-run-temps.csv', '../models', 2, observation_types=obs_types, covariate_types=cov_types, overwrite=False)
