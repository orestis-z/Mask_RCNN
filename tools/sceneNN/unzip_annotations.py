import os
import sys

parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import tools


def unzip_all(path, dest):
    for file in os.listdir(path):
        if file[-4:] == '.zip' and file[-5] != 'b':
            print(file)
            file_path = os.path.join(path, file)
            tools.unzip(file_path, dest)
            os.remove(file_path)


if __name__ == '__main__':
    args = sys.argv[1:3]
    unzip_all(*args)
