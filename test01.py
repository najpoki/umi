#! /usr/bin/env python3

import csv
import datetime
import sys


def load_data(path):
    trans = str.maketrans("/ :", "   ")
    data = []
    with open(path, "r", newline="") as f:
        parser = csv.reader(f)
        for row in parser:
            assert len(row) == 10
            pair = row[0]
            date = [int(i) for i in row[1].translate(trans).split()]
            assert len(date) == 6
            date = datetime.datetime(*date)
            values = [float(i) for i in row[2:10]]
            row = [pair, date]
            row.extend(values)
            data.append(tuple(row))
    return data


def iter_data(data, pair, m):
    def _merge(r1, r2):
        l, r1 = r1
        if l == 0:
            return (1, tuple(r2))
        return (l + 1,
                (r1[0], r1[1],
                 r1[2], max(r1[3], r2[3]), min(r1[4], r2[4]), r2[5],
                 r1[6], max(r1[7], r2[7]), min(r1[8], r2[8]), r2[9]))

    assert m >= 1
    data = [i for i in data if i[0] == pair]
    data.sort(key=lambda x: x[1])
    _row = (0, None)
    for row in data:
        _row = _merge(_row, row)
        if _row[0] == m:
            yield _row[1]
            _row = (0, None)


def main():
    path = sys.argv[1]
    data = load_data(path)
    usdjpy = list(iter_data(data, "USDJPY", 5))
    print(len(usdjpy))
    for i in usdjpy[:10]:
        print(i)

if __name__ == "__main__":
    main()
