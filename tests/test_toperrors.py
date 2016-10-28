# coding: utf-8
"""
[toperrors_cv_float]
$ vwoptimize.py -d small_ag_news_binary.csv --strip_punct --lowercase --quiet --toperrors 1.3 --cv --metric acc,vw_average_loss,vw_train_average_loss
(0)cv acc = 0.72
cv vw_average_loss = 0.778847
cv vw_train_average_loss = 0.951884
-0.40328,1,Clearing Out Fannie #39;s  #39;Phantoms #39;,Score one for the lion tamer. The federal regulator that oversees home-mortgage giant Fannie Mae finally persuaded that government-sponsored enterprise to agree to clean up its questionable accounting practices.

[toperrors_cv_float_0.0]
$ vwoptimize.py -d small_ag_news_binary.csv --strip_punct --lowercase --quiet --toperrors 0.0 --cv | wc -l
(0)cv vw_average_loss = 0.778847
50

[toperrors_cv_float_1.0]
$ vwoptimize.py -d small_ag_news_binary.csv --strip_punct --lowercase --quiet --toperrors 1.0 --cv | wc -l
(0)cv vw_average_loss = 0.778847
14

[toperrors_cv_int]
$ vwoptimize.py -d small_ag_news_binary.csv --strip_punct --lowercase --quiet --toperrors 1 --cv --metric acc,vw_average_loss,vw_train_average_loss
(0)cv acc = 0.72
cv vw_average_loss = 0.778847
cv vw_train_average_loss = 0.951884
-0.40328,1,Clearing Out Fannie #39;s  #39;Phantoms #39;,Score one for the lion tamer. The federal regulator that oversees home-mortgage giant Fannie Mae finally persuaded that government-sponsored enterprise to agree to clean up its questionable accounting practices.

[toperrors_cv_int_2]
$ vwoptimize.py -d small_ag_news_binary.csv --strip_punct --lowercase --quiet --toperrors 2 --cv | wc -l
(0)cv vw_average_loss = 0.778847
2

[toperrors_cv_int_50]
$ vwoptimize.py -d small_ag_news_binary.csv --strip_punct --lowercase --quiet --toperrors 50 --cv | wc -l
(0)cv vw_average_loss = 0.778847
50

[toperrors_cv_int_150]
$ vwoptimize.py -d small_ag_news_binary.csv --strip_punct --lowercase --quiet --toperrors 150 --cv | wc -l
(0)cv vw_average_loss = 0.778847
50

"""

import sys
import os
__doc__ = __doc__.replace('vwoptimize.py', '%s ../vwoptimize.py' % sys.executable)

output = open('small_ag_news_binary.csv', 'w')
for line in open('small_ag_news.csv'):
    klass, rest = line.split(',', 1)
    if klass.strip('"') in "12":
        klass = '-1,'
    else:
        klass = '1,'
    output.write(klass + rest)
output.flush()
os.fsync(output)
output.close()
