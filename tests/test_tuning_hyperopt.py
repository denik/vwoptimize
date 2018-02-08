"""
[tuning1]
$ HYPEROPT_FMIN_SEED=20 vwoptimize.py -d small_ag_news.csv --oaa 4 -b 10..20? --learning_rate 0.1..10? --l1 1e-11..1e-2?? --quiet --hyperopt 10
Result vw --oaa 4 --quiet --l1 3e-10 --learning_rate 0.9 -b 14 : vw_average_loss=0.62*
Result vw --oaa 4 --quiet --learning_rate 5.3 -b 16 : vw_average_loss=0.62
Result vw --oaa 4 --quiet --learning_rate 2.1 -b 18 : vw_average_loss=0.62
Result vw --oaa 4 --quiet --l1 2e-06 --learning_rate 7.4 -b 19 : vw_average_loss=0.62
Result vw --oaa 4 --quiet --learning_rate 0.6 -b 20 : vw_average_loss=0.6*
Result vw --oaa 4 --quiet --learning_rate 8.4 -b 18 : vw_average_loss=0.62
Result vw --oaa 4 --quiet --learning_rate 2.9 -b 19 : vw_average_loss=0.62
Result vw --oaa 4 --quiet --learning_rate 9.4 -b 17 : vw_average_loss=0.64
Result vw --oaa 4 --quiet --l1 2e-11 --learning_rate 4.5 -b 11 : vw_average_loss=0.7
Result vw --oaa 4 --quiet --l1 1e-03 --learning_rate 8.5 -b 14 : vw_average_loss=0.66
Best vw options = --oaa 4 --quiet --learning_rate 0.6 -b 20
Best vw_average_loss = 0.6

$ HYPEROPT_FMIN_SEED=20 vwoptimize.py -d small_ag_news.csv --oaa 4 -b 10..20? --learning_rate 0.1..10? --l1 1e-11..1e-2?? --quiet --hyperopt 5 --hyperopt_alg rand --workers 1
Result vw --oaa 4 --quiet --l1 3e-10 --learning_rate 0.9 -b 14 : vw_average_loss=0.62*
Result vw --oaa 4 --quiet --learning_rate 5.3 -b 16 : vw_average_loss=0.62
Result vw --oaa 4 --quiet --learning_rate 2.1 -b 18 : vw_average_loss=0.62
Result vw --oaa 4 --quiet --l1 2e-06 --learning_rate 7.4 -b 19 : vw_average_loss=0.62
Result vw --oaa 4 --quiet --learning_rate 0.6 -b 20 : vw_average_loss=0.6*
Best vw options = --oaa 4 --quiet --learning_rate 0.6 -b 20
Best vw_average_loss = 0.6

$ HYPEROPT_FMIN_SEED=20 vwoptimize.py -d small_ag_news.csv --oaa 4 -b 10..20? --learning_rate 0.1..10? --l1 1e-11..1e-2?? --quiet --hyperopt 5 --hyperopt_alg anneal
Result vw --oaa 4 --quiet --l1 3e-10 --learning_rate 0.9 -b 14 : vw_average_loss=0.62*
Result vw --oaa 4 --quiet --l1 7e-08 --learning_rate 8.2 -b 16 : vw_average_loss=0.62
Result vw --oaa 4 --quiet --l1 6e-05 --learning_rate 3.9 -b 13 : vw_average_loss=0.64
Result vw --oaa 4 --quiet --l1 5e-06 --learning_rate 3.0 -b 10 : vw_average_loss=0.76
Result vw --oaa 4 --quiet --l1 3e-11 --learning_rate 5.7 -b 16 : vw_average_loss=0.62
Best vw options = --oaa 4 --quiet --l1 3e-10 --learning_rate 0.9 -b 14
Best vw_average_loss = 0.62

$ HYPEROPT_FMIN_SEED=20 vwoptimize.py -d small_ag_news.csv --oaa 4 -b 10..20? --learning_rate 0.1..10? --l1 1e-11..1e-2?? --quiet --hyperopt 5 --hyperopt_alg bla
(1)
Traceback (most recent call last):
 ...
ImportError: No module named bla

$ HYPEROPT_FMIN_SEED=20 vwoptimize.py --adaptive? -d iris.vw --hyperopt 2 --quiet
Result vw --quiet --adaptive : vw_average_loss=1.440247*
Result vw --quiet --adaptive : vw_average_loss=1.440247
Best vw options = --quiet --adaptive
Best vw_average_loss = 1.440247

[cleanup]
$ ls .vwoptimize
<BLANKLINE>
"""

import sys
__doc__ = __doc__.replace('vwoptimize.py', '%s ../vwoptimize.py' % sys.executable)
