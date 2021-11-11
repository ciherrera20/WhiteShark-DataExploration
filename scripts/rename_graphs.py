import os
import re

path = '../graphs'

pattern = re.compile(r'^(\d)s-2o-(.*)$')

for subdir in os.listdir(path):
    abs_dir = os.path.join(path, subdir)
    if os.path.isdir(abs_dir):
        for filename in os.listdir(abs_dir):
            filepath = os.path.join(abs_dir, filename)
            m = pattern.match(filename)
            if m is not None:
                g = m.groups()
                print(filepath, '->', os.path.join(abs_dir, '{}s-ts-{}'.format(g[0], g[1])))
                os.rename(filepath, os.path.join(abs_dir, '{}s-ts-{}'.format(g[0], g[1])))