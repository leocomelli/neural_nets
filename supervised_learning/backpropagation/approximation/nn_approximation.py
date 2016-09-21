from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer

def load_dataset(filename):

    with open(filename) as f:
        lines = f.read().splitlines()

    ds = SupervisedDataSet(14, 1)

    for line in lines:
        if len(line) == 0 or line.startswith('%'):
            continue

        # normalizing values
        cols = [float(c) for c in line.split(' ')]
        max_value = max(cols)
        cols = [c / max_value for c in cols]

        array = []
        for col in cols[:-1]:
            array.append(float(col))
        else:
            clazz = float(cols[-1])

        ds.appendLinked(array, clazz)

    return ds

if __name__ == '__main__':

    ds = load_dataset('teste1.txt')

    tstdata, trndata = ds.splitWithProportion(0.25)

    print "Number of training patterns: ", len(trndata)
    print "Input and output dimensions: ", trndata.indim, trndata.outdim
    print "Number of testing patterns: ", len(tstdata)
    print "Input and output dimensions: ", tstdata.indim, tstdata.outdim

    n = buildNetwork(trndata.indim, 300, trndata.outdim)

    t = BackpropTrainer(n, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

    t.trainEpochs(10)

    print "epoch: %4d" % t.totalepochs, \
          "  mean square error (mse): %s" % t.testOnData(tstdata)
    print "final weights: %s" % n.params