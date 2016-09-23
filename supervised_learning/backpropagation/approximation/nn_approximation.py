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

def compute():

    ds = load_dataset('teste1.txt')

    tstdata, trndata = ds.splitWithProportion(0.15)

    print "Number of training patterns: ", len(trndata)
    print "Input and output dimensions: ", trndata.indim, trndata.outdim
    print "Number of testing patterns: ", len(tstdata)
    print "Input and output dimensions: ", tstdata.indim, tstdata.outdim

    n = buildNetwork(trndata.indim, 500, trndata.outdim)

    t = BackpropTrainer(n, dataset=trndata, momentum=0.1, learningrate=0.01, weightdecay=0.01) #verbose=True

    t.trainEpochs(50)

    mse = t.testOnData(tstdata)
    #n.params

    return t.totalepochs, mse

if __name__ == '__main__':

    count = 0
    mse_t = 0
    while count < 10:
        epochs, mse = compute()

        print "epoch: %4d" % epochs, \
              "  mean square error (mse): %s" % mse

        mse_t += mse
        count += 1

    print mse_t / 10