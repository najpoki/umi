#! /usr/bin/env python3

import csv
import datetime
import pathlib
import pickle
import random
import sys

import numpy
import pybrain
import pybrain.optimization
import pybrain.rl.environments.fitnessevaluator
import pybrain.tools.customxml


def create_network(na, nb):
    n = pybrain.structure.networks.RecurrentNetwork()
    inLayer = pybrain.structure.LinearLayer(4)
    aLayer = pybrain.structure.ReluLayer(na)
    bLayer = pybrain.structure.GateLayer(nb)
    outLayer = pybrain.structure.TanhLayer(1)
    biasUnit = pybrain.structure.BiasUnit()
    n.addInputModule(inLayer)
    n.addModule(aLayer)
    n.addModule(bLayer)
    n.addOutputModule(outLayer)
    n.addModule(biasUnit)
    in_to_a = pybrain.structure.FullConnection(inLayer, aLayer)
    a_to_b = pybrain.structure.FullConnection(aLayer, bLayer)
    b_to_out = pybrain.structure.FullConnection(bLayer, outLayer)
    bias_to_a = pybrain.structure.FullConnection(biasUnit, aLayer)
    bias_to_b = pybrain.structure.FullConnection(biasUnit, bLayer)
    n.addConnection(in_to_a)
    n.addConnection(a_to_b)
    n.addConnection(b_to_out)
    n.addConnection(bias_to_a)
    n.addConnection(bias_to_b)
    b_to_b = pybrain.structure.FullConnection(bLayer, bLayer)
    n.addRecurrentConnection(b_to_b)
    n.sortModules()
    return n


def read_data(currency, ex):
    data = []
    trans = str.maketrans("/ :", "   ")
    data_path = pathlib.Path(sys.argv[0]).resolve().parent / "data/"
    name = "{0}{1}_[0-9]*.csv".format(currency, "_EX" if ex else "")
    for path in data_path.rglob(name):
        with path.open(mode="r", encoding="shift_jis", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == "日時":
                    continue
                t = datetime.datetime(*map(int, row[0].translate(trans).split()))
                d = tuple(map(float, row[1:]))
                d = (t,) + d
                data.append(d)
    data.sort(key=lambda x: x[0])
    return data


class MyEvaluator(pybrain.rl.environments.fitnessevaluator.FitnessEvaluator):
    def __init__(self, data, total_min, element_min):
        self.data = data
        self.total_min = total_min
        self.element_min = element_min
        self.num = 15
        self._hoge()

    def _hoge(self):
        self.index = []
        for _ in range(self.num):
            self.index.append(random.choice(list(range(len(self.data)-3*self.total_min+1))))

    def _f(self, x, index):
        x.reset()
        prev, now = None, []
        for d in self.data[index:index+2*self.total_min]:
            now.append(d)
            if len(now) < self.element_min:
                continue
            now = [now[0][0], now[0][1],
                   max([i[2] for i in now]),
                   min([i[3] for i in now]),
                   now[-1][4]]
            if prev is not None:
                _prev = prev[4]
                _d = [i / _prev for i in now[1:5]]
                x.activate(_d)
            prev, now = now, []
        buy, sell = 0.0, 0.0
        position = 0
        for d in self.data[index+2*self.total_min:index+3*self.total_min]:
            now.append(d)
            if len(now) < self.element_min:
                continue
            now = [now[0][0], now[0][1],
                   max([i[2] for i in now]),
                   min([i[3] for i in now]),
                   now[-1][4]]
            if prev is not None:
                _prev = prev[4]
                _d = [i / _prev for i in now[1:5]]
                v = x.activate(_d)[0]
                new_position = int(numpy.nan_to_num(v)*10)
                if new_position - position > 0:
                    buy += now[4] * (new_position - position)
                elif new_position - position < 0:
                    sell += now[4] * -1 * (new_position - position)
                position = new_position
            prev, now = now, []
        if position > 0:
            sell += prev[4] * position
        elif position < 0:
            buy += prev[4] * -1 * position
        position = 0
        point = buy - sell
        return point

    def f(self, x):
        point = 0.0
        for i in range(self.num):
            point += self._f(x, self.index[i])
        point /= self.num
        print(point)
        return point


def main():
    na, nb = 2, 2
    currency = "USDJPY"
    ex = False
    nfile = "nfile"
    # initialPopulation = None
    with open(nfile + ".pop", "rb") as f:
        initialPopulation = pickle.load(f)
    # n = create_network(na, nb)
    n = pybrain.tools.customxml.NetworkReader.readFrom(nfile + ".net")
    data = read_data(currency, ex)
    myEvaluator = MyEvaluator(data, 24 * 60, 5)
    ga = pybrain.optimization.GA(evaluator=myEvaluator, initEvaluable=n,
                                 populationSize=100,
                                 topProportion=0.5, initialPopulation=initialPopulation)
    for _ in range(10):
        ga.bestEvaluation = None
        best = ga.learn(additionalLearningSteps=1)
        print(best)
        myEvaluator._hoge()
    with open(nfile + ".pop", "wb") as f:
        pickle.dump(ga.currentpop, f)
    pybrain.tools.customxml.NetworkWriter.writeToFile(n, nfile + ".net")

if __name__ == "__main__":
    main()
