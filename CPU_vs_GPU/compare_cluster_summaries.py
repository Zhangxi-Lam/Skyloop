#!/usr/bin/env python

"""
Compare the skyloop result summary printed for each cluster, between
two versions of the output.
"""

import re

# Regex for the values in a line:
values_findall = re.compile("\s*(.*?)\s*=\s*(\S+)").findall

def get_values(line):
    """
    Return a mapping between variable names and values found around
    the equal signs in the given line.

    The values are returned as floats.
    """
    
    return {variable: float(value)
            for (variable, value) in values_findall(line)}
    
if __name__ == "__main__":

    import argparse
    import itertools
    import sys
    
    parser = argparse.ArgumentParser(
        description="""
        Compare the skyloop result summary printed for each cluster,
        between two versions of the output. The lines are compared one
        by one and must therefore normally (mostly) match. The values
        following the equal signs are matched together: the format is
        '... = <value>' (followed by a space or newline). They should
        therefore typically correspond to variables printed in the
        same order. The program checks that the two files have the
        same number of lines and issues a warning otherwise.""")

    parser.add_argument("path0", help="first output file")
    parser.add_argument("path1", help="second output file")
    args = parser.parse_args()

    with open(args.path0) as file0, open(args.path1) as file1:
        for (line0, line1) in itertools.izip_longest(file0, file1):
            try:
                values0, values1 = get_values(line0), get_values(line1)
            except TypeError:  # Happens if a file is longer: None was parsed
                print >> sys.stderr, "Warning: one file has more lines"
                break
            
            print values0

