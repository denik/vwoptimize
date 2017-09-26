"""
[tuning1__kfold10]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 -b 18/20? --kfold 10 --quiet   # by default vw_average_loss is being tuned
(0)Result vw --oaa 4 --quiet -b 18... vw_average_loss=0.4800
Result vw --oaa 4 --quiet -b 20... vw_average_loss=0.5000
Best vw_average_loss with 'no preprocessing' = 0.4800*
Best preprocessor options = <none>
Best vw options = --oaa 4 --quiet -b 18
Best vw_average_loss = 0.4800

[tuning_acc_kfold10]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --metric acc -b 18/20? --kfold 10 --quiet
(0)Result vw --oaa 4 --quiet -b 18... acc=0.5200
Result vw --oaa 4 --quiet -b 20... acc=0.5000
Best acc with 'no preprocessing' = 0.5200*
Best preprocessor options = <none>
Best vw options = --oaa 4 --quiet -b 18
Best acc = 0.5200
acc = 0.38

[tuning1__progressive]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 -b 18/20? --quiet   #  in this case vw_average_loss is progressive validation loss
(0)Result vw --oaa 4 --quiet -b 18... vw_average_loss=0.6200
Result vw --oaa 4 --quiet -b 20... vw_average_loss=0.6000*
Best vw_average_loss with 'no preprocessing' = 0.6000*
Best preprocessor options = <none>
Best vw options = --oaa 4 --quiet -b 20
Best vw_average_loss = 0.6000

[tuning_acc]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --metric acc -b 18/20? --quiet   # same result, since acc = 1-vw_average_loss in this case
(0)Result vw --oaa 4 --quiet -b 18... acc=0.3800
Result vw --oaa 4 --quiet -b 20... acc=0.4000*
Best acc with 'no preprocessing' = 0.4000*
Best preprocessor options = <none>
Best vw options = --oaa 4 --quiet -b 20
Best acc = 0.4000
acc = 0.4

[tuning_acc_and_print_loss]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --metric acc,vw_average_loss -b 18/20? --quiet
(0)Result vw --oaa 4 --quiet -b 18... acc=0.3800   vw_average_loss=0.6200
Result vw --oaa 4 --quiet -b 20... acc=0.4000*  vw_average_loss=0.6000
Best acc with 'no preprocessing' = 0.4000*
Best preprocessor options = <none>
Best vw options = --oaa 4 --quiet -b 20
Best acc = 0.4000
acc = 0.4

[tuning5_error]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --metric acc,vw_average_loss,num_features,vw_passes_used --passes 2/4?
(1)Result vw --oaa 4 --passes 2... error: vw failed: Error: need a cache file for multiple passes : try using --cache_file
Result vw --oaa 4 --passes 4... error: vw failed: Error: need a cache file for multiple passes : try using --cache_file
Best acc with 'no preprocessing' = None
Best preprocessor options = <none>
Best vw options = None
Best acc = None
tuning failed

[tuning_bad_metrics]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --metric vw_average_loss,brier,auc,acc,num_features -b 18/20? --kfold 10 --quiet
(0)Result vw --oaa 4 --quiet -b 18... vw_average_loss=0.4800   brier=ValueError auc=ValueError acc=0.5200 num_features=4103.2
Result vw --oaa 4 --quiet -b 20... vw_average_loss=0.5000   brier=ValueError auc=ValueError acc=0.5000 num_features=4138.8
Best vw_average_loss with 'no preprocessing' = 0.4800*
Best preprocessor options = <none>
Best vw options = --oaa 4 --quiet -b 18
Best vw_average_loss = 0.4800
brier = ValueError: Cannot calculate on multiclass
auc = ValueError: multiclass format is not supported
acc = 0.38
num_features = 4484

[tuning_with_model__kfold10]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 -b 18/20? -f tmp_model1 --writeconfig tmp_config1 --quiet --kfold 10
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
$ vwoptimize.py -d small_ag_news.csv --oaa 4 -b 20 -f tmp_model2 --writeconfig tmp_config2 --quiet --kfold 10
(0)10-fold vw_average_loss = 0.5

[using_model2]
$ vwoptimize.py -d small_ag_news.csv --readconfig tmp_config2 -t 2>&1 | grep 'loss ='
average loss = 0.000000

[tune_preprocessor__max_words]
$ vwoptimize.py -d small_ag_news.csv --max_words /2/4? --quiet
(0)Result vw --quiet... vw_average_loss=2.3963
Best vw_average_loss with 'no preprocessing' = 2.3963*
Result vw --quiet... vw_average_loss=2.8069
Best options with --max_words 2 = --quiet
Best vw_average_loss with '--max_words 2' = 2.8069
Result vw --quiet... vw_average_loss=3.0816
Best options with --max_words 4 = --quiet
Best vw_average_loss with '--max_words 4' = 3.0816
Best preprocessor options = <none>
Best vw options = --quiet
Best vw_average_loss = 2.3963

[tune_preprocessor__strip_punct]
$ vwoptimize.py -d small_ag_news.csv --strip_punct? --quiet
(0)Result vw --quiet... vw_average_loss=2.3963
Best vw_average_loss with 'no preprocessing' = 2.3963*
Result vw --quiet... vw_average_loss=2.4007
Best options with --strip_punct = --quiet
Best vw_average_loss with '--strip_punct' = 2.4007
Best preprocessor options = <none>
Best vw options = --quiet
Best vw_average_loss = 2.3963

[tune_preprocessor__both]
$ vwoptimize.py -d small_ag_news.csv --max_words /2/5? --strip_punct? --quiet
(0)Result vw --quiet... vw_average_loss=2.3963
Best vw_average_loss with 'no preprocessing' = 2.3963*
Result vw --quiet... vw_average_loss=2.8069
Best options with --max_words 2 = --quiet
Best vw_average_loss with '--max_words 2' = 2.8069
Result vw --quiet... vw_average_loss=3.1641
Best options with --max_words 5 = --quiet
Best vw_average_loss with '--max_words 5' = 3.1641
Result vw --quiet... vw_average_loss=2.4007
Best options with --strip_punct = --quiet
Best vw_average_loss with '--strip_punct' = 2.4007
Result vw --quiet... vw_average_loss=2.7978
Best options with --strip_punct --max_words 2 = --quiet
Best vw_average_loss with '--strip_punct --max_words 2' = 2.7978
Result vw --quiet... vw_average_loss=3.3755
Best options with --strip_punct --max_words 5 = --quiet
Best vw_average_loss with '--strip_punct --max_words 5' = 3.3755
Best preprocessor options = <none>
Best vw options = --quiet
Best vw_average_loss = 2.3963

[cleanup]
$ ls .vwoptimize
<BLANKLINE>
"""

import sys
__doc__ = __doc__.replace('vwoptimize.py', '%s ../vwoptimize.py' % sys.executable)
