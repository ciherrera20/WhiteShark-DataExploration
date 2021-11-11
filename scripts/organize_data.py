import os
import re

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

path = '../data/'

for id in ids:
    if not os.path.isdir(os.path.join(path, id)):
        os.mkdir(os.path.join(path, id))

for filename in os.listdir(path):
    if os.path.isfile(os.path.join(path, filename)):
        for id in ids:
            pattern = re.compile('{}{}'.format(id, r'(?!_)'))
            if pattern.match(filename):
                os.rename(os.path.join(path, filename), os.path.join(path, id, filename))