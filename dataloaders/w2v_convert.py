import sys
import struct
import numpy as np

matrix = []
raw_path = '/Users/kuroneko/Downloads/glove.840B.300d.txt'
output_path = '/Users/kuroneko/Downloads/glove.840B.300d.dat'
np_output = '/Users/kuroneko/Downloads/glove.840B.300d'

with open(raw_path, 'r') as inf:
    with open(output_path, 'wb') as ouf:
        counter = 0
        for line in inf:
            try:
                row = [float(x) for x in line.split()[1:]]
                assert len(row) == 300
                ouf.write(struct.pack('i', len(row)))
                ouf.write(struct.pack('%sf' % len(row), *row))
                counter += 1
                matrix.append(np.array(row, dtype=np.float32))
            except:
                print('guagua')
                continue
            if counter % 10000 == 0:
                sys.stdout.write('%d points processed...\n' % counter)
np.save(np_output, np.array(matrix))
