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

[validation_intermediate]
$ vwoptimize.py -d smaller_ag_news.csv --oaa 4 --lowercase? -b 18/20? --validation small_ag_news.csv --quiet --intermediate_results results1/
Result vw --oaa 4 --quiet -b 18 : vw_average_loss=0.38**
Results saved to results1/1.json
Result vw --oaa 4 --quiet -b 20 : vw_average_loss=0.38
Best vw_average_loss with 'no preprocessing' = 0.38*
Result vw --oaa 4 --quiet -b 18 : vw_average_loss=0.36**
Results saved to results1/2.json
Result vw --oaa 4 --quiet -b 20 : vw_average_loss=0.36
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
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.50 : vw_average_loss=0.38**
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.50 : vw_average_loss=0.38+
Best vw_average_loss with 'no preprocessing' = 0.38*
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.50 : vw_average_loss=0.36**
Result vw --oaa 4 --quiet -b 18 --learning_rate 0.53 : vw_average_loss=0.34**
Result vw --oaa 4 --quiet -b 20 --learning_rate 0.53 : vw_average_loss=0.34+
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

[with_test]
$ vwoptimize.py -d simple.vw  --learning_rate 0.50? --validation simple_validation.vw  --test simple_test.vw --intermediate_results simple_results/
Result vw --learning_rate 0.50 : vw_average_loss=0.355994*
Result vw --learning_rate 0.53 : vw_average_loss=0.352473*
Result vw --learning_rate 0.55 : vw_average_loss=0.350462*
Result vw --learning_rate 0.58 : vw_average_loss=0.347893*
Result vw --learning_rate 0.63 : vw_average_loss=0.344646*
Result vw --learning_rate 0.68 : vw_average_loss=0.342467*
Result vw --learning_rate 0.78 : vw_average_loss=0.340499*
Result vw --learning_rate 0.88 : vw_average_loss=0.34071
Result vw --learning_rate 0.83 : vw_average_loss=0.340394*
Result vw --learning_rate 0.80 : vw_average_loss=0.3404
Result vw --learning_rate 0.85 : vw_average_loss=0.340476
Result vw --learning_rate 0.81 : vw_average_loss=0.340379*
Result vw --learning_rate 0.82 : vw_average_loss=0.340378*
Results saved to simple_results/1.json
<BLANKLINE>
Best vw options = --learning_rate 0.82
Best vw_average_loss = 0.340378
+ vw -d simple.vw --learning_rate 0.82
Num weight bits = 18
learning rate = 0.82
initial_t = 0
power_t = 0.5
using no cache
Reading datafile = simple.vw
num sources = 1
average  since         example        example  current  current  current
loss     last          counter         weight    label  predict features
0.000000 0.000000            1            1.0   0.0000   0.0000        5
0.666667 1.000000            2            3.0   1.0000   0.0000        5
<BLANKLINE>
finished run
number of examples per pass = 3
passes used = 1
weighted example sum = 4.000000
weighted label sum = 2.000000
average loss = 0.750000
best constant = 0.500000
best constant's loss = 0.250000
total feature number = 15

$ cat simple_results/*
{
    "args": [
        "", 
        "--learning_rate 0.82"
    ], 
    "preprocessor_opts": "", 
    "result": "0.340378", 
    "y_pred_text": "0.107407\\n0.289508 second_house\\n", 
    "y_pred_text_test": "0.455976 third_house\\n"
}

$ rm -fr results1 results2 simple_results
<BLANKLINE>

[cleanup]
$ ls .vwoptimize
<BLANKLINE>
"""

import sys
__doc__ = __doc__.replace('vwoptimize.py', '%s ../vwoptimize.py' % sys.executable)
