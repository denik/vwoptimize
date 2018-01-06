# coding: utf-8
"""
[normal_run]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 2>&1 | grep average
average  since         example        example  current  current  current
average loss = 0.620000

[normal_run_with_acc]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --metric acc 2>&1 | egrep 'average|acc'
average  since         example        example  current  current  current
average loss = 0.620000
acc = 0.38

[metricformat_raw]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --metric acc --quiet --metricformat raw
acc = 0.38

[metricformat_raw_cv]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --metricformat raw --kfold 10
10-fold vw_average_loss = [0.2, 0.4, 0.6, 0.4, 0.4, 0.4, 0.6, 0.6, 0.8, 0.4]

[normal_run_write_model]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 -b 20 --writeconfig tmp_config -f tmp_model --quiet --morelogs --tmpid hello
write config = tmp_config
+ vw -d .vwoptimize/hello.1.vw -f tmp_model.tmp --oaa 4 -b 20 --quiet

[parseaudit1]
$ vwoptimize.py --quiet -i tmp_model -t -a -d small_ag_news.csv | vwoptimize.py --parseaudit | head -n 2
Unique non-zero features: 4532
0.0901711 his 5
0.0761782 agreed 2

[normal_run_read_model]
$ vwoptimize.py -d small_ag_news.csv --readconfig tmp_config 2>&1 | egrep 'bits|loss|only'
Num weight bits = 20
loss     last          counter         weight    label  predict features
average loss = 0.000000

[normal_run_read_model_t]
$ vwoptimize.py -d small_ag_news.csv --readconfig tmp_config -t 2>&1 | egrep 'bits|loss|only'
only testing
Num weight bits = 20
loss     last          counter         weight    label  predict features
average loss = 0.000000

[cv]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --kfold 10
10-fold vw_average_loss = 0.48

[cv_same]
$ vwoptimize.py -d small_ag_news.csv --kfold 10 --oaa 4 --metric vw_average_loss  # this is default
10-fold vw_average_loss = 0.48

[cv_run_write_model]
$ vwoptimize.py -d small_ag_news.csv --kfold 10 --ect 4 -b 21 --writeconfig tmp_config1 -f tmp_model1 --quiet --morelogs --tmpid hello
10-fold vw_average_loss = 0.46
write config = tmp_config1
+ vw -d .vwoptimize/hello.1.vw -f tmp_model1.tmp --ect 4 -b 21 --quiet

[cv_run_read_model]
$ vwoptimize.py -d small_ag_news.csv --readconfig tmp_config1 2>&1 | egrep 'bits|loss|only'
Num weight bits = 21
loss     last          counter         weight    label  predict features
average loss = 0.000000

[cv_run_read_model_t]
$ vwoptimize.py -d small_ag_news.csv --readconfig tmp_config1 -t 2>&1 | egrep 'bits|loss|only'
only testing
Num weight bits = 21
loss     last          counter         weight    label  predict features
average loss = 0.000000

[cv_ect_b20]
$ vwoptimize.py -d small_ag_news.csv --kfold 10 --oaa 4 -b 20
10-fold vw_average_loss = 0.5

[cv_metric_acc]
$ vwoptimize.py -d small_ag_news.csv --kfold 10 --oaa 4 --metric acc
10-fold acc = 0.52

[cv_metric_acc_num_features]
$ vwoptimize.py -d small_ag_news.csv --kfold 10 --oaa 4 --metric acc,num_features
10-fold acc = 0.52
10-fold num_features = 4103.2

[cv_metric_acc_num_features_vw_average_loss]
$ vwoptimize.py -d small_ag_news.csv  --kfold 10 --oaa 4 --metric acc,num_features,vw_average_loss
10-fold acc = 0.52
10-fold num_features = 4103.2
10-fold vw_average_loss = 0.48

[cv_with_passes_and_cache_agnews_csv]
$ vwoptimize.py -d small_ag_news.csv  --kfold 10 --oaa 4 --metric acc,num_features,vw_average_loss,vw_train_passes_used --passes 10 -c -k
10-fold acc = 0.52
10-fold num_features = 3791.2
10-fold vw_average_loss = 0.48
10-fold vw_train_passes_used = 4.2

[cv_with_passes_and_cache_iris_vw]
$ vwoptimize.py -d iris.vw --kfold 10 --oaa 3 --metric acc,num_features,vw_average_loss,vw_train_passes_used --passes 10 -c -k
10-fold acc = 0.68
10-fold num_features = 15
10-fold vw_average_loss = 0.32
10-fold vw_train_passes_used = 7.4

[cv_bad_metrics_multiclass_mse]
$ vwoptimize.py -d small_ag_news.csv --kfold 10 --oaa 4 --metric mse
10-fold mse = 1.22

[cv_bad_metrics]
$ vwoptimize.py -d small_ag_news.csv --kfold 10 --oaa 4 --metric brier,auc,acc,vw_average_loss
10-fold brier = ValueError: Cannot calculate on multiclass
10-fold auc = ValueError: multiclass format is not supported
10-fold acc = 0.52
10-fold vw_average_loss = 0.48

[cache_file_err1]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --passes 2 --kfold 10 --metric vw_train_passes_used
(1)
vw failed: Error: need a cache file for multiple passes : try using --cache_file

[cache_file_err2]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --passes 2 --kfold 10 --metric vw_train_passes_used --cache_file tmp.cache
(1)
Dont provide --cache_file, one will be added automatically.

A few tests for --report:

[prepare_tmp_y_true]
$ printf '0\\n1\\n0\\n1\\n' > tmp_y_true
<BLANKLINE>

[prepare_tmp_y_pred]
$ printf '0\\n0\\n0\\n1\\n' > tmp_y_pred
<BLANKLINE>

[report1]
$ vwoptimize.py -d tmp_y_true --format csv --report -p tmp_y_pred --metric acc
acc = 0.75

[report2]
$ vwoptimize.py -d tmp_y_true --format csv --report -p tmp_y_pred --metric acc --weight 0:0
acc = 0.5

[report3]
$ vwoptimize.py -d tmp_y_true --format csv --report -p tmp_y_pred --metric acc --weight 0:0.22
acc = 0.590164

[report4]
$ cat tmp_y_true | vwoptimize.py --format csv --report -p tmp_y_pred --metric acc
acc = 0.75

[report5]
$ cat tmp_y_pred | vwoptimize.py --format csv --report -p /dev/stdin -d tmp_y_true --metric acc
acc = 0.75

[kfold25]
$ vwoptimize.py -d iris.vw --metric vw_train_weighted_example_sum,vw_average_loss --oaa 3 --kfold 25
25-fold vw_train_weighted_example_sum = 144
25-fold vw_average_loss = 0.333333

[kfold2]
$ vwoptimize.py -d iris.vw --metric vw_train_weighted_example_sum,vw_average_loss --oaa 3 --kfold 2
2-fold vw_train_weighted_example_sum = 75
2-fold vw_average_loss = 0.32

[kfold_extreme]
$ head -n 50 iris.vw | vwoptimize.py -d - --metric vw_train_weighted_example_sum,vw_average_loss --oaa 3 --kfold 50
50-fold vw_train_weighted_example_sum = 49
50-fold vw_average_loss = 0.24

[kfold_extreme2]
$ head -n 50 iris.vw | vwoptimize.py -d - --metric vw_train_weighted_example_sum,vw_average_loss --oaa 3 --kfold 49
49-fold vw_train_weighted_example_sum = 48.979592
49-fold vw_average_loss = 0.234694

[kfold_too_many]
$ head -n 50 iris.vw | vwoptimize.py -d - --metric vw_train_weighted_example_sum,vw_average_loss --oaa 3 --kfold 51
51-fold vw_train_weighted_example_sum = 49.019608
51-fold vw_average_loss = nan

[cv_predictions_stdout]
$ head -n 10 iris.vw | vwoptimize.py -d - --kfold 10 --oaa 3 -p /dev/stdout
10-fold vw_average_loss = 0.6
3
1
1
3
3
1
2
3
3
2

[cv_predictions1]
$ head -n 10 iris.vw | vwoptimize.py -d - --kfold 10 --oaa 3 -p tmp_iris_vw_10_predictions -r tmp_iris_vw_10_raw -f tmp_model_after_cv --readable_model tmp_readable_model_after_cv --quiet
10-fold vw_average_loss = 0.6

[cv_predictions2]
$ cat tmp_iris_vw_10_predictions
3
1
1
3
3
1
2
3
3
2

[cv_predictions3]
$ cat tmp_iris_vw_10_raw
1:-0.638016 2:-0.472186 3:-0.105228
1:-0.176385 2:-0.44021 3:-0.258789
1:-0.164775 2:-0.437558 3:-0.259951
1:-0.553075 2:-0.577699 3:-0.057384
1:-0.415519 2:-0.53863 3:-0.118955
1:-0.185424 2:-0.340418 3:-0.255382
1:-0.516176 2:-0.30486 3:-0.339061
1:-0.381771 2:-0.661139 3:-0.0247422
1:-0.301372 2:-0.31029 3:-0.208859
1:-0.46615 2:-0.158856 3:-0.565657

[cv_predictions3_vw]
$ head -n 10 iris.vw | vw -f tmp_vw_model --readable_model tmp_readable_model --quiet --oaa 3
<BLANKLINE>

[cv_predictions4_model]
$ diff tmp_vw_model tmp_model_after_cv
<BLANKLINE>

[cv_predictions5_readable_model]
$ diff -u tmp_readable_model tmp_readable_model_after_cv
<BLANKLINE>

[parseaudit2]
$ vwoptimize.py --quiet -i tmp_vw_model -t -a -d iris.vw | vwoptimize.py --parseaudit | head -n 1
Unique non-zero features: 15
0.0811443 3 150

[progressive_validation_vw]
$ printf '1 | hello\\n-1 | world' | vw 2>&1 | grep 'average loss'
average loss = 1.285535

[progressive_validation]
$ printf '1 | hello\\n-1 | world' | vwoptimize.py -d - -p /dev/stdout --quiet
0
0.253423

[progressive_validation_raw_mse]
$ printf '1 | hello\\n-1 | world' | vwoptimize.py -d - -r /dev/stdout --metric mse --quiet
mse = 1.285535
0
0.253423

[kfold2]
$ printf '1 | hello\\n-1 | world' | vwoptimize.py -d - --kfold 2 -p /dev/stdout
2-fold vw_average_loss = 1.571069
-0.253423
0.253423

[kfold2_raw]
$ printf '1 | hello\\n-1 | world' | vwoptimize.py -d - --kfold 2 -r /dev/stdout --metric mse
2-fold mse = 1.571069
-0.253423
0.253423

[kfold3]
$ printf '1 | hello\\n-1 | world' | vwoptimize.py -d - --kfold 3
3-fold vw_average_loss = nan

[kfold2_with_holdout]
$ vwoptimize.py -d iris.vw --kfold 2 --passes 2 -c -k --metric vw_train_average_loss,vw_average_loss
2-fold vw_train_average_loss = 0.165788 h
2-fold vw_average_loss = 0.069007

[kfold2_without_holdout]
$ vwoptimize.py -d iris.vw --kfold 2 --passes 2 -c -k --metric vw_train_average_loss,vw_average_loss --holdout_off
2-fold vw_train_average_loss = 0.192494
2-fold vw_average_loss = 0.067851

[tags]
$ vwoptimize.py -d tags.vw -p /dev/stdout --quiet
0 tag1
0.253423 tag2

[tags_kfold2]
$ vwoptimize.py -d tags.vw --kfold 2 -p /dev/stdout
2-fold vw_average_loss = 1.571069
-0.253423 tag1
0.253423 tag2

[kfold1_error]
$ vwoptimize.py -d tags.vw -p /dev/stdout --kfold 1
(1)
kfold parameter must > 1

[cleanup]
$ ls .vwoptimize
<BLANKLINE>
"""

import sys
__doc__ = __doc__.replace('vwoptimize.py', '%s ../vwoptimize.py' % sys.executable)
