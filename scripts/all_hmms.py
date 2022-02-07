import hmm
import os

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

ids = [
    '2020-04',
    '2020-10',
    '2020-12',
    '2020-13',
    '2020-15',
    '2020-16',
    '2020-17',
    '2020-19',
    '2020-20',
    '2020-21',
    '2020-22',
    '2020-31',
    '2020-32',
    '2020-33',
    '2020-34',
    '2020-35',
    '2020-35_2',
    '2020-36',
    '2020-37',
    '2020-40',
    '2020-41',
    '2020-42'
]

for id in ids:
    runs_dir = '../data/{}/runs'.format(id, id)
    if os.path.isdir(runs_dir):
        for run in os.listdir(runs_dir):
            hmm.main(os.path.join(runs_dir, run), '../models', 2, observation_types=['turning_angle', 'speed'], covariate_types=['temperature'], overwrite=False)
    else:
        print('No runs found for {}'.format(id))