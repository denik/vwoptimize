# flake8: noqa
"""
[validation_same_file]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 -b 18/20? --validation small_ag_news.csv --quiet
Result vw --oaa 4 --quiet -b 18 : vw_average_loss=0*
Result vw --oaa 4 --quiet -b 20 : vw_average_loss=0
Best vw options = --oaa 4 --quiet -b 18
Best vw_average_loss = 0

[validation_other_file]
$ vwoptimize.py -d smaller_ag_news.csv --oaa 4 --lowercase? -b 18/20? --learning_rate 0.50? --validation small_ag_news.csv --quiet
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.50 : vw_average_loss=0.38**
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.53 : vw_average_loss=0.38
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.47 : vw_average_loss=0.38
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.51 : vw_average_loss=0.38
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.49 : vw_average_loss=0.38
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.50 : vw_average_loss=0.38+
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.53 : vw_average_loss=0.38
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.47 : vw_average_loss=0.38
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.51 : vw_average_loss=0.38
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.49 : vw_average_loss=0.38
Best vw_average_loss with 'no preprocessing' = 0.38*
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.50 : vw_average_loss=0.36**
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.53 : vw_average_loss=0.34**
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.55 : vw_average_loss=0.34
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.54 : vw_average_loss=0.34
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.51 : vw_average_loss=0.36
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.52 : vw_average_loss=0.36
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.53 : vw_average_loss=0.34+
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.55 : vw_average_loss=0.36
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.50 : vw_average_loss=0.36
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.54 : vw_average_loss=0.34
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.51 : vw_average_loss=0.36
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.52 : vw_average_loss=0.34
Best options with --lowercase = --oaa 4 --quiet -b 18 --learning_rate 0.53
Best vw_average_loss with '--lowercase' = 0.34*
Best preprocessor options = --lowercase
Best vw options = --oaa 4 --quiet -b 18 --learning_rate 0.53
Best vw_average_loss = 0.34

[cleanup]
$ ls .vwoptimize
<BLANKLINE>
"""

import sys
__doc__ = __doc__.replace('vwoptimize.py', '%s ../vwoptimize.py' % sys.executable)
