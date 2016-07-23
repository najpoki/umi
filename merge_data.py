#! /usr/bin/env python3

import os.path
import re
import sys


def main():
    header = ("日時,始値(BID),高値(BID),安値(BID),終値(BID),"
              "始値(ASK),高値(ASK),安値(ASK),終値(ASK)")
    for line_in in sys.stdin:
        path = line_in.strip()
        pair = re.match(r"(.+)_[0-9]+\.csv", os.path.basename(path)).group(1)
        with open(path, "r", encoding="shift_jis") as f:
            for row in f:
                row = row.strip()
                if row == header:
                    continue
                print(",".join([pair, row]))

if __name__ == "__main__":
    main()
