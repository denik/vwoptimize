# coding: utf-8
"""
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

[columnspec1]
$ echo 'Hello:1,H:2,444:44 55,Hello:|,World:||,!:|,5' | vwoptimize.py --columnspec vw,vw,vw_namespace,text_namespace,text_a,text,y --tovw /dev/stdout --lowercase --quiet
5 | Hello:1 H:2 |namespace 444:44 55 hello |a world | !

[columnspec2]
$ echo '|hello:25 world AA |t a b c,hey' | vwoptimize.py --columnspec vw_a,vw_a --tovw /dev/stdout --lowercase --quiet
|hello:25 world AA |t a b c |a hey

[columnspec3]
$ echo '|hello:25 world AA |t a b c,hey' | vwoptimize.py --columnspec vw,vw --tovw /dev/stdout --lowercase --quiet
|hello:25 world AA |t a b c | hey

[preprocessor_vw]
$ echo '|Hello:25\\n' | vwoptimize.py --tovw /dev/stdout --lowercase --quiet
|hello:25

[weight_vw]
$ echo '1 | Hello:25\\n' | vwoptimize.py --tovw /dev/stdout --quiet --weight 1:10
1 10 | Hello:25

[weight_vw_run]
$ echo '1 | Hello:25' | vwoptimize.py --weight 1:10 2>&1 | grep weighted
weighted example sum = 10.000000
weighted label sum = 10.000000

[weight_csv_run]
$ echo '1,hello' | vwoptimize.py --weight 1:10 --format csv 2>&1 | grep weighted
weighted example sum = 10.000000
weighted label sum = 10.000000

[cleanup]
$ ls .vwoptimize
<BLANKLINE>
"""

import sys
__doc__ = __doc__.replace('vwoptimize.py', '%s ../vwoptimize.py' % sys.executable)
