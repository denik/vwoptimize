"""
[tuning1]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 -b 18/20?   # by default vw_average_loss is being tuned
(0)Result vw --oaa 4 -b 18... vw_average_loss=0.4800
Result vw --oaa 4 -b 20... vw_average_loss=0.5000
Best vw_average_loss with 'no preprocessing' = 0.4800*
Best preprocessor options = <none>
Best vw options = --oaa 4 -b 18
Best vw_average_loss = 0.4800

[tuning1__nfolds1]
$ vwoptimize.py -d small_ag_news.csv --nfolds 1 --oaa 4 -b 18/20?   # by default vw_train_average_loss is being tuned
(0)Result vw --oaa 4 -b 18... vw_train_average_loss=0.6200
Result vw --oaa 4 -b 20... vw_train_average_loss=0.6000*
Best vw_train_average_loss with 'no preprocessing' = 0.6000*
Best preprocessor options = <none>
Best vw options = --oaa 4 -b 20
Best vw_train_average_loss = 0.6000
cv vw_train_average_loss = 0.6

[tuning2]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --metric vw_average_loss -b 18/20?   # same as above
(0)Result vw --oaa 4 -b 18... vw_average_loss=0.4800
Result vw --oaa 4 -b 20... vw_average_loss=0.5000
Best vw_average_loss with 'no preprocessing' = 0.4800*
Best preprocessor options = <none>
Best vw options = --oaa 4 -b 18
Best vw_average_loss = 0.4800

[tuning3]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --metric acc -b 18/20?   # same result, since acc = 1-vw_average_loss in this case
(0)Result vw --oaa 4 -b 18... acc=0.5200
Result vw --oaa 4 -b 20... acc=0.5000
Best acc with 'no preprocessing' = 0.5200*
Best preprocessor options = <none>
Best vw options = --oaa 4 -b 18
Best acc = 0.5200

[tuning4]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --metric acc,vw_average_loss -b 18/20?   # same, but capture vw output as well
(0)Result vw --oaa 4 -b 18... acc=0.5200   vw_average_loss=0.4800
Result vw --oaa 4 -b 20... acc=0.5000   vw_average_loss=0.5000
Best acc with 'no preprocessing' = 0.5200*
Best preprocessor options = <none>
Best vw options = --oaa 4 -b 18
Best acc = 0.5200

[tuning5_error]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --metric acc,vw_average_loss,num_features,vw_train_passes_used --passes 2/4?
(1)Result vw --oaa 4 --passes 2... error: vw failed: Error: need a cache file for multiple passes : try using --cache_file
Result vw --oaa 4 --passes 4... error: vw failed: Error: need a cache file for multiple passes : try using --cache_file
Best acc with 'no preprocessing' = None
Best preprocessor options = <none>
Best vw options = None
Best acc = None
tuning failed

[tuning_bad_metrics]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --metric vw_average_loss,brier,auc,acc,num_features -b 18/20?
(0)Result vw --oaa 4 -b 18... vw_average_loss=0.4800   brier=ValueError auc=ValueError acc=0.5200 num_features=4103.2
Result vw --oaa 4 -b 20... vw_average_loss=0.5000   brier=ValueError auc=ValueError acc=0.5000 num_features=4138.8
Best vw_average_loss with 'no preprocessing' = 0.4800*
Best preprocessor options = <none>
Best vw options = --oaa 4 -b 18
Best vw_average_loss = 0.4800

[tuning_with_model]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 -b 18/20? -f tmp_model1 --writeconfig tmp_config1 --quiet
(0)Result vw --oaa 4 --quiet -b 18... vw_average_loss=0.4800
Result vw --oaa 4 --quiet -b 20... vw_average_loss=0.5000
Best vw_average_loss with 'no preprocessing' = 0.4800*
Best preprocessor options = <none>
Best vw options = --oaa 4 --quiet -b 18
Best vw_average_loss = 0.4800

[using_model1]
$ vwoptimize.py -d small_ag_news.csv --readconfig tmp_config1 -t 2>&1 | grep 'loss ='
average loss = 0.000000

[using_model1_quiet]
$ vwoptimize.py -d small_ag_news.csv --readconfig tmp_config1 -t --quiet 2>&1
<BLANKLINE>

[cv_with_model]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 -b 20 -f tmp_model2 --writeconfig tmp_config2 --quiet
<BLANKLINE>

[using_model2]
$ vwoptimize.py -d small_ag_news.csv --readconfig tmp_config2 -t 2>&1 | grep 'loss ='
average loss = 0.000000

[cleanup]
$ ls .vwoptimize
<BLANKLINE>
"""

import sys
__doc__ = __doc__.replace('vwoptimize.py', '%s ../vwoptimize.py' % sys.executable)
