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
    import collections

    from matplotlib import pyplot
    
    parser = argparse.ArgumentParser(
        description="""
        Compare the skyloop result summary printed for each cluster,
        between two versions of the output. The lines are compared one
        by one and must therefore normally (mostly) match. The
        variables and values following the equal signs are matched
        together (by variable name). The variables in a line therefore
        do not have to be in the same order, and different lines can
        contain different variables. The program checks that variable
        names must match line by line between the two files,
        though. The program also checks that the two files have the
        same number of lines and issues a warning otherwise.""")

    parser.add_argument("path0", help="first output file")
    parser.add_argument("path1", help="second output file")
    args = parser.parse_args()

    # Main results of this comparison program: mapping between
    # variables and observed relative errors, and counts of identical
    # values. Relative errors are of the form value1 - value0, divided
    # by value0 if it is not 0, or else by value1.
    relative_errors = collections.defaultdict(list)
    identical_value_counts = collections.Counter()
    
    with open(args.path0) as file0, open(args.path1) as file1:
        
        for (line_num, (line0, line1)) in enumerate(
            itertools.izip_longest(file0, file1), 1):

            # Variables and values reading:
            try:
                values0, values1 = get_values(line0), get_values(line1)
            except TypeError:  # Happens if a file is longer: None was parsed
                print >> sys.stderr, "WARNING: one file has more lines."
                line_num -= 1  # One more line read than exists
                break
            
            # Basic check: do the two lines contain the same variables?
            if values0.viewkeys() ^ values1.viewkeys():
                sys.exit("Error: Lines #{} must have the same variables."
                         .format(line_num))

            for variable in values0.iterkeys():
                
                value0, value1 = values0[variable], values1[variable]

                if value0 == value1:
                    identical_value_counts[variable] += 1
                else:
                    ref_value = value0 if value0 else value1  # Never 0
                    relative_errors[variable].append((value1-value0)/ref_value)
                    
    # Report:
    
    print "Number of identical values (increasing numbers, sorted names):"
    
    for (variable, count) in sorted(
        identical_value_counts.viewitems(),
        key=lambda (variable, count): (count, variable)):
        
        print "- {}: {}/{}".format(variable, count, line_num)

    for (variable, errors) in relative_errors.iteritems():
        pyplot.figure(variable)
        pyplot.hist(errors)
        pyplot.xlabel("Error")
        pyplot.ylabel("Count")
        pyplot.grid()
        pyplot.title("Relative error histogram for {}".format(variable))
        
    print "Quit by closing all the windows or doing Ctrl-Z kill %%..."
    pyplot.show()
