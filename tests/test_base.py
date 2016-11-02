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
(0)acc = 0.38

[metricformat_raw_cv]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --metricformat raw --cv
(0)cv vw_average_loss = [0.2, 0.4, 0.6, 0.4, 0.4, 0.4, 0.6, 0.6, 0.8, 0.4]

[normal_run_write_model]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 -b 20 --writeconfig tmp_config -f tmp_model --quiet --morelogs --tmpid hello
(0)write config = tmp_config
+ vw -d .vwoptimize/hello.1.vw -f tmp_model.tmp --oaa 4 -b 20 --quiet

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

[feature_mask_retrain_write_model]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 -b 15 --writeconfig tmp_config -f tmp_model --quiet --morelogs --tmpid hello --feature_mask_retrain
(0)write config = tmp_config
+ [1] vw -d .vwoptimize/hello.1.vw -f tmp_model.tmp.feature_mask --oaa 4 -b 15 --quiet
+ [2] vw -d .vwoptimize/hello.1.vw --quiet -f tmp_model.tmp --feature_mask tmp_model.tmp.feature_mask -i tmp_model.tmp.feature_mask

[feature_mask_retrain_read_model_t]
$ vwoptimize.py -d small_ag_news.csv --readconfig tmp_config --tmpid hey -t 2>&1 | egrep '\+|bits|loss|only'
+ vw -d .vwoptimize/hey.1.vw -i tmp_model -t
only testing
Num weight bits = 15
loss     last          counter         weight    label  predict features
average loss = 0.000000

[cv]
$ vwoptimize.py -d small_ag_news.csv --cv --oaa 4
(0)cv vw_average_loss = 0.48

[cv_same]
$ vwoptimize.py -d small_ag_news.csv --cv --oaa 4 --metric vw_average_loss  # this is default
(0)cv vw_average_loss = 0.48

[cv_run_write_model]
$ vwoptimize.py -d small_ag_news.csv --cv --ect 4 -b 21 --writeconfig tmp_config1 -f tmp_model1 --quiet --morelogs --tmpid hello
(0)cv vw_average_loss = 0.46
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
$ vwoptimize.py -d small_ag_news.csv  --cv --oaa 4 -b 20
(0)cv vw_average_loss = 0.5

[cv_metric_acc]
$ vwoptimize.py -d small_ag_news.csv  --cv --oaa 4 --metric acc
(0)cv acc = 0.52

[cv_metric_acc_num_features]
$ vwoptimize.py -d small_ag_news.csv  --cv --oaa 4 --metric acc,num_features
(0)cv acc = 0.52
cv num_features = 4103.2

[cv_metric_acc_num_features_vw_average_loss]
$ vwoptimize.py -d small_ag_news.csv  --cv --oaa 4 --metric acc,num_features,vw_average_loss
(0)cv acc = 0.52
cv vw_average_loss = 0.48
cv num_features = 4103.2

[cv_with_passes_and_cache_agnews_csv]
$ vwoptimize.py -d small_ag_news.csv  --cv --oaa 4 --metric acc,num_features,vw_average_loss,vw_train_passes_used --passes 10 -c -k
(0)cv acc = 0.52
cv vw_average_loss = 0.48
cv vw_train_passes_used = 4.2
cv num_features = 3791.2

[cv_with_passes_and_cache_iris_vw]
$ vwoptimize.py -d iris.vw --cv --oaa 3 --metric acc,num_features,vw_average_loss,vw_train_passes_used --passes 10 -c -k
(0)cv acc = 0.68
cv vw_average_loss = 0.32
cv vw_train_passes_used = 7.4
cv num_features = 15

[cv_bad_metrics_multiclass_mse]
$ vwoptimize.py -d small_ag_news.csv --cv --oaa 4 --metric mse
(0)cv mse = 1.22

[cv_bad_metrics]
$ vwoptimize.py -d small_ag_news.csv --cv --oaa 4 --metric brier,auc,acc,vw_average_loss
(0)cv brier = ValueError: Cannot calculate on multiclass
cv auc = ValueError: multiclass format is not supported
cv acc = 0.52
cv vw_average_loss = 0.48

[cache_file_err1]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --passes 2 --cv --metric vw_train_passes_used
(1)vw failed: Error: need a cache file for multiple passes : try using --cache_file

[cache_file_err2]
$ vwoptimize.py -d small_ag_news.csv --oaa 4 --passes 2 --cv --metric vw_train_passes_used --cache_file tmp.cache
(1)Dont provide --cache_file, one will be added automatically.

A few tests for --report:

[prepare_tmp_y_true]
$ printf '0\\n1\\n0\\n1\\n' > tmp_y_true
<BLANKLINE>

[prepare_tmp_y_pred]
$ printf '0\\n0\\n0\\n1\\n' > tmp_y_pred
<BLANKLINE>

[report1]
$ vwoptimize.py -d tmp_y_true --format csv --report -p tmp_y_pred --metric acc,acc_w
(0)acc = 0.75
acc_w = 0.75

[report2]
$ vwoptimize.py -d tmp_y_true --format csv --report -p tmp_y_pred --metric acc_w,acc --weight 0:0
(0)acc_w = 0.5
acc = 0.75

[report3]
$ vwoptimize.py -d tmp_y_true --format csv --report -p tmp_y_pred --metric acc,acc_w --weight 0:0.22
(0)acc = 0.75
acc_w = 0.590164

[report4]
$ cat tmp_y_true | vwoptimize.py --format csv --report -p tmp_y_pred --metric acc
(0)acc = 0.75

[report5]
$ cat tmp_y_pred | vwoptimize.py --format csv --report -p /dev/stdin -d tmp_y_true --metric acc
(0)acc = 0.75

[tovw]
$ vwoptimize.py -d simple.csv --tovw /dev/stdout
1 | Hello World! first class
2 | Goodbye World. second class
1 | hello first class again

[tovw__weight]
$ vwoptimize.py -d simple.csv --tovw /dev/stdout --weight 1:0.5
1 0.5 | Hello World! first class
2 | Goodbye World. second class
1 0.5 | hello first class again

[tovw__weight_train]
$ vwoptimize.py -d simple.csv --tovw /dev/stdout --weight_train 1:0.5
1 0.5 | Hello World! first class
2 | Goodbye World. second class
1 0.5 | hello first class again

[tovw__weight_metric]
$ vwoptimize.py -d simple.csv --tovw /dev/stdout --weight_metric 1:0.5
1 | Hello World! first class
2 | Goodbye World. second class
1 | hello first class again

[named_labels1]
$ vwoptimize.py -d text_labels.csv --tovw /dev/stdout --named_labels welcome,bye --weight welcome:0.5
welcome 0.5 | Hello World!
bye | Goodbye World.
welcome 0.5 | hello

[named_labels2]
$ vwoptimize.py -d text_labels.csv --tovw /dev/stdout --named_labels welcome,bye
welcome | Hello World!
bye | Goodbye World.
welcome | hello

[nfolds_25]
$ vwoptimize.py -d iris.vw --metric vw_train_weighted_example_sum,vw_average_loss --oaa 3 --cv --nfolds 25
(0)cv vw_train_weighted_example_sum = 144
cv vw_average_loss = 0.333333

[nfolds_2]
$ vwoptimize.py -d iris.vw --metric vw_train_weighted_example_sum,vw_average_loss --oaa 3 --cv --nfolds 2
(0)cv vw_train_weighted_example_sum = 75
cv vw_average_loss = 0.32

[nfolds_extreme]
$ head -n 50 iris.vw | vwoptimize.py --metric vw_train_weighted_example_sum,vw_average_loss --oaa 3 --cv --nfolds 50
(0)cv vw_train_weighted_example_sum = 49
cv vw_average_loss = 0.24

[nfolds_extreme2]
$ head -n 50 iris.vw | vwoptimize.py --metric vw_train_weighted_example_sum,vw_average_loss --oaa 3 --cv --nfolds 49
(0)cv vw_train_weighted_example_sum = 48.9796
cv vw_average_loss = 0.234694

[nfolds_too_many]
$ head -n 50 iris.vw | vwoptimize.py --metric vw_train_weighted_example_sum,vw_average_loss --oaa 3 --cv --nfolds 51
(0)cv vw_train_weighted_example_sum = 49.0196
cv vw_average_loss = nan

[cv_predictions_stdout]
$ head -n 10 iris.vw | vwoptimize.py --cv --oaa 3 -p /dev/stdout
(0)cv vw_average_loss = 0.6
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
$ head -n 10 iris.vw | vwoptimize.py --cv --oaa 3 -p tmp_iris_vw_10_predictions -r tmp_iris_vw_10_raw -f tmp_model_after_cv --readable_model tmp_readable_model_after_cv --quiet
(0)cv vw_average_loss = 0.6

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

[cv_predictions_vw]
$ head -n 10 iris.vw | vw -f tmp_vw_model --readable_model tmp_readable_model --quiet --oaa 3
<BLANKLINE>

[cv_predictions_model]
$ diff tmp_vw_model tmp_model_after_cv
<BLANKLINE>

[cv_predictions_readable_model]
$ diff -u tmp_readable_model tmp_readable_model_after_cv
<BLANKLINE>

[progressive_validation_vw]
$ printf '1 | hello\\n-1 | world' | vw 2>&1 | grep 'average loss'
average loss = 1.285535

[nfolds1]
$ printf '1 | hello\\n-1 | world' | vwoptimize.py --nfolds 1 -p /dev/stdout
(0)cv vw_train_average_loss = 1.28554
0
0.253423

[nfolds1_raw]
$ printf '1 | hello\\n-1 | world' | vwoptimize.py --nfolds 1 -r /dev/stdout --metric mse
(0)cv mse = 1.28553
0
0.253423

[nfolds2]
$ printf '1 | hello\\n-1 | world' | vwoptimize.py --nfolds 2 -p /dev/stdout
(0)cv vw_average_loss = 1.57107
-0.253423
0.253423

[nfolds2_raw]
$ printf '1 | hello\\n-1 | world' | vwoptimize.py --nfolds 2 -r /dev/stdout --metric mse
(0)cv mse = 1.57107
-0.253423
0.253423

[nfolds3]
$ printf '1 | hello\\n-1 | world' | vwoptimize.py --nfolds 3
(0)cv vw_average_loss = nan

[holdout_validation_nfolds2]
$ vwoptimize.py -d iris.vw --nfolds 2 --passes 2 -c -k --metric vw_train_average_loss,vw_average_loss
(0)cv vw_train_average_loss = 0.165788 h
cv vw_average_loss = 0.0690075

[holdout_validation_nfolds1]
$ vwoptimize.py -d iris.vw --nfolds 1 --passes 2 -c -k
(0)cv vw_train_average_loss = 0.116409 h

[nfolds1_tags]
$ vwoptimize.py -d tags.vw --nfolds 1 -p /dev/stdout
(0)cv vw_train_average_loss = 1.28554
0 tag1
0.253423 tag2

[nfolds2_tags]
$ vwoptimize.py -d tags.vw --nfolds 2 -p /dev/stdout
(0)cv vw_average_loss = 1.57107
-0.253423 tag1
0.253423 tag2

[cleanup]
$ ls .vwoptimize
<BLANKLINE>
"""

import sys
__doc__ = __doc__.replace('vwoptimize.py', '%s ../vwoptimize.py' % sys.executable)
