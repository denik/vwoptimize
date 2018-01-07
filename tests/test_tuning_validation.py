"""
[validation_same_file]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 -b 18/20? --validation small_ag_news.csv --quiet
Result vw --oaa 4 --quiet -b 18... vw_average_loss=0.48*
Result vw --oaa 4 --quiet -b 20... vw_average_loss=0.5
Best vw options = --oaa 4 --quiet -b 18
Best vw_average_loss = 0.48

[validation_other_file]
$ vwoptimize.py -d smaller_ag_news.csv --oaa 4 --lowercase? -b 18/20? --learning_rate 0.50? --validation small_ag_news.csv --quiet
Result vw --oaa 4 --quiet -b 18... vw_average_loss=0.48*
Result vw --oaa 4 --quiet -b 20... vw_average_loss=0.5
Best vw options = --oaa 4 --quiet -b 18
Best vw_average_loss = 0.48

[cleanup]
$ ls .vwoptimize
<BLANKLINE>
"""

import sys
__doc__ = __doc__.replace('vwoptimize.py', '%s ../vwoptimize.py' % sys.executable)
