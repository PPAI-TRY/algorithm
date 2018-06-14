#!/bin/sh

import os
import json
import sys


class Generator(object):
    def __init__(self, output, test_data_path="../data/test.csv", prediction_path="output/prediction"):
        self.output = output
        self.test_data_path = test_data_path
        self.prediction_path = prediction_path
        self.predictions = {}  # (q1, q2) => prediction dict
        self.test_data = []  # p1

    def run(self):
        self.load_prediction()

        with open(self.test_data_path) as f:
            line_no = 0
            for line in f:
                line_no += 1
                if line_no == 1:
                    continue
                tmp = line.split(",")
                assert(len(tmp) == 2)
                q1 = int(tmp[0][1:])
                q2 = int(tmp[1][1:])
                prediction = self.predictions[(q1, q2)]
                self.test_data.append(prediction["p1"])

        with open(self.output, "w") as f:
            f.write("y_pre\n")
            for p1 in self.test_data:
                f.write("%s\n" % p1)

    def load_prediction(self):
        file_names = os.listdir(self.prediction_path)
        for name in file_names:
            print name, name[-5:]
            if name[-5:] != ".json":
                continue
            full_path = os.path.join(self.prediction_path, name)
            with open(full_path) as f:
                for line in f:
                    data = json.loads(line)
                    self.predictions[(data["q1"], data["q2"])] = data

        print "predictions count %d" % len(self.predictions)


if __name__ == "__main__":
    output = "submission.csv"
    if len(sys.argv) > 1:
        output = sys.argv[1]

    generator = Generator(output)
    generator.run()
