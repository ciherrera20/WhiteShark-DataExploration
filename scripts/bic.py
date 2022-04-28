import hmm
import os
import json

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
    models_dir = '../models/{}'.format(id)
    if os.path.isdir(models_dir):
        for model in os.listdir(models_dir):
            if os.path.isdir(model):
                data_obj = json.load(open(os.path.join(model, 'hmm-data.json'), 'r'))
                
    else:
        print('No models found for {}'.format(id))