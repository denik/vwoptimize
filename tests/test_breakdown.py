# coding: utf-8
"""
[breakdown_nogroups]
$ vwoptimize.py -d simple.vw --metric acc,count --breakdown '[a-z]+_house' --quiet
(0)acc = 0.333333
count = 3
breakdown second_house acc = 0
breakdown second_house count = 1
breakdown  third_house acc = 0
breakdown  third_house count = 1
breakdown      nomatch acc = 1
breakdown      nomatch count = 1

[breakdown_nogroups_top1]
$ vwoptimize.py -d simple.vw --metric acc,count --breakdown '[a-z]+_house' --quiet --breakdown_top 1
(0)acc = 0.333333
count = 3
breakdown second_house acc = 0
breakdown second_house count = 1
breakdown rest acc = 0.5
breakdown rest count = 2

[breakdown_groups]
$ vwoptimize.py -d simple.vw --metric acc,count --breakdown '([a-z]+)_house' --quiet
(0)acc = 0.333333
count = 3
breakdown  second acc = 0
breakdown  second count = 1
breakdown   third acc = 0
breakdown   third count = 1
breakdown nomatch acc = 1
breakdown nomatch count = 1

[breakdown_simple]
$ vwoptimize.py -d simple.vw --metric acc,count --breakdown 'house' --quiet
(0)acc = 0.333333
count = 3
breakdown   house acc = 0
breakdown   house count = 2
breakdown nomatch acc = 1
breakdown nomatch count = 1

[train]
$ vwoptimize.py -d simple.vw --quiet -f tmp.model
<BLANKLINE>

[breakdown_test]
$ vwoptimize.py -d simple.vw --metric acc,count --breakdown 'house' -t --quiet -i tmp.model
(0)acc = 0.333333
count = 3
breakdown   house acc = 0
breakdown   house count = 2
breakdown nomatch acc = 1
breakdown nomatch count = 1

[breakdown_kfold]
$ vwoptimize.py -d simple.vw --metric acc,count --breakdown 'house' --quiet --kfold 3
(0)3-fold acc = 0.333333
3-fold count = 3
3-fold breakdown   house acc = 0
3-fold breakdown   house count = 2
3-fold breakdown nomatch acc = 1
3-fold breakdown nomatch count = 1

[breakdown_top_only1]
$ vwoptimize.py -d simple.vw --metric acc,count --quiet --breakdown_top 1
(0)acc = 0.333333
count = 3
breakdown second_house acc = 0
breakdown second_house count = 1
breakdown rest acc = 0.5
breakdown rest count = 2

[breakdown_top_only2]
$ vwoptimize.py -d simple.vw --metric acc,count --quiet --breakdown_top 2
(0)acc = 0.333333
count = 3
breakdown second_house acc = 0
breakdown second_house count = 1
breakdown  third_house acc = 0
breakdown  third_house count = 1
breakdown rest acc = 1
breakdown rest count = 1

[breakdown_top_only3]
$ vwoptimize.py -d simple.vw --metric acc,count --quiet --breakdown_top 3
(0)acc = 0.333333
count = 3
breakdown second_house acc = 0
breakdown second_house count = 1
breakdown  third_house acc = 0
breakdown  third_house count = 1
breakdown      nomatch acc = 1
breakdown      nomatch count = 1

[breakdown_top_only4]
$ vwoptimize.py -d simple.vw --metric acc,count --quiet --breakdown_top 4
(0)acc = 0.333333
count = 3
breakdown second_house acc = 0
breakdown second_house count = 1
breakdown  third_house acc = 0
breakdown  third_house count = 1
breakdown      nomatch acc = 1
breakdown      nomatch count = 1
"""

import sys
__doc__ = __doc__.replace('vwoptimize.py', '%s ../vwoptimize.py' % sys.executable)
