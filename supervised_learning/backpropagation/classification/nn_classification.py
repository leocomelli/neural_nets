from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
from pybrain.supervised import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer

def load_dataset(filename):

    with open(filename) as f:
        lines = f.read().splitlines()

    ds = ClassificationDataSet(7, 1, class_labels=['1', '2', '3'])

    for line in lines:
        array = []
        cols = line.split('\t')
        for col in cols[:-1]:
            if col != '' and col != ' ': #\s\t or #\t\t
                array.append(float(col))
        else:
            clazz = int(cols[-1]) -1

        print "class: %s, array = %s" % (clazz, array)
        ds.addSample(array, clazz)
        #ds.appendLinked(array, clazz)

    return ds

def calculate_statistics_dataset(ds):
    counter = [0, 0, 0]
    for d in ds:
        d = d[1].tolist() # convert numpy.array to list
        for i, x in enumerate(d):
            counter[i] += x

    return counter

if __name__ == '__main__':

    ds = load_dataset('seeds.txt')

    # Produce two new datasets, the first one containing the fraction given
    # by `proportion` of the samples.
    tstdata, trndata = ds.splitWithProportion(0.25)

    counter = calculate_statistics_dataset(trndata)
    print "Number of training patterns: ", len(trndata)
    counter = calculate_statistics_dataset(trndata)
    print "Number of training patterns by class: 1 => %s, 2 => %s, 3 => %s" % (counter[0], counter[1], counter[2])
    print "Input and output dimensions: ", trndata.indim, trndata.outdim
    print "Number of testing patterns: ", len(tstdata)
    counter = calculate_statistics_dataset(tstdata)
    print "Number of testing patterns by class: 1 => %s, 2 => %s, 3 => %s" % (counter[0], counter[1], counter[2])
    print "Input and output dimensions: ", tstdata.indim, tstdata.outdim

    # Converts the target classes to a 1-of-k representation, retaining the old targets as a field `class`
    # For neural network classification, it is highly advisable to encode classes with one output
    # neuron per class. Note that this operation duplicates the original targets and stores them
    # in an (integer) field named class.
    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()

    # a softmax function because we are doing classification. There are more options to explore here,
    # e.g. try changing the hidden layer transfer function to linear instead of (the default) sigmoid
    n = buildNetwork(trndata.indim, 300, trndata.outdim, outclass=SoftmaxLayer)

    # momentum: is the ratio by which the gradient of the last timestep is used
    # weightdecay: corresponds to the weightdecay rate, where 0 is no weight decay at all.
    t = BackpropTrainer(n, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

    #for i in range(200):
    t.trainEpochs(10)

    # percentError: return percentage of mismatch between out and target values
    # testOnClassData: Return winner-takes-all classification output on a given dataset
    #                  Winner-take-all is a computational principle applied in computational
    #                  models of neural networks by which neurons in a layer compete with each
    #                  other for activation
    trnresult = percentError(t.testOnClassData(), trndata['class'])
    tstresult = percentError(t.testOnClassData(dataset=tstdata), tstdata['class'])

    #t.activateOnDataset(ds)

    print "epoch: %4d" % t.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult, \
