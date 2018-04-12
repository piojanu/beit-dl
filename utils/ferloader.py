import numpy as np

def load_fer(path):
    """Loads and parses FER+ dataset
    
    Arguments:
      path (string): Path to FER+ dataset .csv or .npz file.

    Return:
      (numpy.array): Array with data in shape <samples number> x <img size>. 
      (numpy.array): Array with target emotion number for each sample.
    """

    if path.endswith('.npz'):
        # Load from numpy archive...
        dataset = np.load(path)
        return dataset['data'], dataset['target']
    else:
        # Parse .csv file...
        with open(path, 'r') as f:
            dataset = f.read()

        targets = []
        samples = []
        for line in dataset.split('\n')[1:-1]:
            label, pixels, _ = line.split(',')
            targets.append(int(label))
            samples.append([int(pixel) for pixel in pixels.split(' ')])

        return np.array(samples), np.array(targets)
