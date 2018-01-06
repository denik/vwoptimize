# coding: utf-8
"""
[vw_train_options_saved]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --quiet -f tmp.model --writeconfig tmp.config && grep vw_train_options tmp.config
"vw_train_options": "--oaa 4"

[vw_train_options_saved_cv]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 -b 15 --quiet -f tmp.model --writeconfig tmp.config --kfold 10 && grep vw_train_options tmp.config
10-fold vw_average_loss = 0.52
"vw_train_options": "--oaa 4 -b 15"

[vw_train_options_saved_tuning]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 -b 10/16? --quiet -f tmp.model --writeconfig tmp.config && grep vw_train_options tmp.config
Result vw --oaa 4 --quiet -b 10... vw_average_loss=0.74000
Result vw --oaa 4 --quiet -b 16... vw_average_loss=0.62000*
Best vw options = --oaa 4 --quiet -b 16
Best vw_average_loss = 0.62000
"vw_train_options": "--oaa 4 -b 16"

[vw_train_options_saved_tuning_kfold]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 -b 10/16? --quiet -f tmp.model --writeconfig tmp.config --kfold 10 && grep vw_train_options tmp.config
Result vw --oaa 4 --quiet -b 10... vw_average_loss=0.64000
Result vw --oaa 4 --quiet -b 16... vw_average_loss=0.52000*
Best vw options = --oaa 4 --quiet -b 16
Best vw_average_loss = 0.52000
"vw_train_options": "--oaa 4 -b 16"

[load_regressor_test]
$ vwoptimize.py -d small_ag_news.csv --readconfig tmp.config -t --metric acc 2>&1 | egrep 'test|loss =|acc ='
only testing
average loss = 0.000000
acc = 1

[load_regressor_notest]
$ vwoptimize.py -d small_ag_news.csv --readconfig tmp.config --metric acc 2>&1 | egrep 'test|loss =|acc ='
average loss = 0.000000
acc = 1

[load_train_options]
$ vwoptimize.py -d small_ag_news.csv --readconfig tmp.config --metric acc --initial_regressor '' --tmpid x 2>&1 | egrep '\+|test|loss =|acc ='
+ vw -d .vwoptimize/x.1.vw -p .vwoptimize/x.3.pred --oaa 4 -b 16
average loss = 0.620000
acc = 0.56

[cleanup]
$ ls .vwoptimize
<BLANKLINE>
"""

import sys
__doc__ = __doc__.replace('vwoptimize.py', '%s ../vwoptimize.py' % sys.executable)
