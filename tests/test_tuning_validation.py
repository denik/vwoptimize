# flake8: noqa
"""
[validation_same_file]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 -b 18/20? --validation small_ag_news.csv --quiet
Result vw --oaa 4 --quiet -b 18... vw_average_loss=0*
Result vw --oaa 4 --quiet -b 20... vw_average_loss=0
Best vw options = --oaa 4 --quiet -b 18
Best vw_average_loss = 0

[validation_other_file]
$ vwoptimize.py -d smaller_ag_news.csv --oaa 4 --lowercase? -b 18/20? --learning_rate 0.50? --validation small_ag_news.csv --quiet
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.50... vw_average_loss=0.38**
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.53... vw_average_loss=0.38
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.47... vw_average_loss=0.38
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.51... vw_average_loss=0.38
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.49... vw_average_loss=0.38
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.50... vw_average_loss=0.38+
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.53... vw_average_loss=0.38
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.47... vw_average_loss=0.38
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.51... vw_average_loss=0.38
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.49... vw_average_loss=0.38
Best vw_average_loss with 'no preprocessing' = 0.38*
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.50... vw_average_loss=0.36**
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.53... vw_average_loss=0.34**
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.55... vw_average_loss=0.34
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.54... vw_average_loss=0.34
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.51... vw_average_loss=0.36
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.52... vw_average_loss=0.36
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.53... vw_average_loss=0.34+
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.55... vw_average_loss=0.36
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.50... vw_average_loss=0.36
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.54... vw_average_loss=0.34
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.51... vw_average_loss=0.36
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.52... vw_average_loss=0.34
Best options with --lowercase = --oaa 4 --quiet -b 18 --learning_rate 0.53
Best vw_average_loss with '--lowercase' = 0.34*
Best preprocessor options = --lowercase
Best vw options = --oaa 4 --quiet -b 18 --learning_rate 0.53
Best vw_average_loss = 0.34

[validation_intermediate]
$ vwoptimize.py -d smaller_ag_news.csv --oaa 4 --lowercase? -b 18/20? --validation small_ag_news.csv --quiet --intermediate_results results1/
Result vw --oaa 4 --quiet -b 18... vw_average_loss=0.38**
Results saved to results1/1.json
Result vw --oaa 4 --quiet -b 20... vw_average_loss=0.38
Best vw_average_loss with 'no preprocessing' = 0.38*
Result vw --oaa 4 --quiet -b 18... vw_average_loss=0.36**
Results saved to results1/2.json
Result vw --oaa 4 --quiet -b 20... vw_average_loss=0.36
Best options with --lowercase = --oaa 4 --quiet -b 18
Best vw_average_loss with '--lowercase' = 0.36*
Best preprocessor options = --lowercase
Best vw options = --oaa 4 --quiet -b 18
Best vw_average_loss = 0.36

$ grep result results1/*
results1/1.json:    "result": "0.38", 
results1/2.json:    "result": "0.36",

[validation_intermediate]
$ vwoptimize.py -d smaller_ag_news.csv --oaa 4 --lowercase? -b 18/20? --validation small_ag_news.csv --quiet --intermediate_results results2/ --learning_rate 0.50? --lesslogs
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.50... vw_average_loss=0.38**
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.50... vw_average_loss=0.38+
Best vw_average_loss with 'no preprocessing' = 0.38*
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.50... vw_average_loss=0.36**
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.53... vw_average_loss=0.34**
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.53... vw_average_loss=0.34+
Best options with --lowercase = --oaa 4 --quiet -b 18 --learning_rate 0.53
Best vw_average_loss with '--lowercase' = 0.34*
Best preprocessor options = --lowercase
Best vw options = --oaa 4 --quiet -b 18 --learning_rate 0.53
Best vw_average_loss = 0.34

$ grep result results2/*
results2/1.json:    "result": "0.38", 
results2/2.json:    "result": "0.38", 
results2/3.json:    "result": "0.34", 
results2/4.json:    "result": "0.34",

$ rm -fr results1 results2
<BLANKLINE>

[cleanup]
$ ls .vwoptimize
<BLANKLINE>
"""

import sys
__doc__ = __doc__.replace('vwoptimize.py', '%s ../vwoptimize.py' % sys.executable)
