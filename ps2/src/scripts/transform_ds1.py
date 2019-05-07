import numpy as np

def load_data(filename):
    Y = D[:, 0]
    X = D[:, 1:]
    return X, Y

def transform(source_path, dest_path):
    D = np.loadtxt(source_path)
    num_x_values = D.shape[1] - 1
    header = ','.join(['y'] + ['x{}'.format(i) for i in range(num_x_values)])
    np.savetxt(dest_path, D, delimiter=',', header=header, comments='')

if __name__ == '__main__':
    transform('./scripts/ds1_a.txt', '../data/ds1_a.csv')
    transform('./scripts/ds1_b.txt', '../data/ds1_b.csv')