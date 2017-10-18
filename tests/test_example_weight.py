# coding: utf-8
"""
# columnspec: weight, weight_train, weight_metric

[example_weight_ignored]
$ vwoptimize.py -d simple_w.csv --columnspec y,drop,text,text --metric acc 2>&1 | egrep 'weighted|acc'
weighted example sum = 3.000000
weighted label sum = 4.000000
acc = 0.666667

[example_weight]
$ vwoptimize.py -d simple_w.csv --columnspec y,weight,text,text 2>&1 --metric acc | egrep 'weighted|acc'
weighted example sum = 2.100000
weighted label sum = 3.100000
acc = 0.52381

[example_weight_metric]
$ vwoptimize.py -d simple_w.csv --columnspec y,weight_metric,text,text 2>&1 --metric acc | egrep 'weighted|acc'
weighted example sum = 3.000000
weighted label sum = 4.000000
acc = 0.52381

[example_weight_train]
$ vwoptimize.py -d simple_w.csv --columnspec y,weight_train,text,text 2>&1 --metric acc | egrep 'weighted|acc'
weighted example sum = 2.100000
weighted label sum = 3.100000
acc = 0.666667

[example_weight_tune]
$ vwoptimize.py -d simple_w.csv --columnspec y,weight,text,text -b 1/2/3? --quiet 2>&1 --metric acc
Result vw --quiet -b 1... acc=0.04762
Result vw --quiet -b 2... acc=0.52381*
Result vw --quiet -b 3... acc=0.52381
Best vw options = --quiet -b 2
Best acc = 0.52381
acc = 0.52381

[cleanup]
$ ls .vwoptimize
<BLANKLINE>
"""

import sys
__doc__ = __doc__.replace('vwoptimize.py', '%s ../vwoptimize.py' % sys.executable)
