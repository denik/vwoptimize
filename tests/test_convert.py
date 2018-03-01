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

[fromvw_tovw]
$ vwoptimize.py -d simple.vw --tovw /dev/stdout
0 | price:.23 sqft:.25 age:.05 2006
1 2 'second_house | price:.18 sqft:.15 age:.35 1976
0 1 0.5 'third_house | price:.53 sqft:.32 age:.87 1924

[fromvw_tovw__weight1]
$ vwoptimize.py -d simple.vw --tovw /dev/stdout --weight 1:1.5
0 | price:.23 sqft:.25 age:.05 2006
1 3.0 'second_house | price:.18 sqft:.15 age:.35 1976
0 1 0.5 'third_house | price:.53 sqft:.32 age:.87 1924

[fromvw_tovw__weight2]
$ vwoptimize.py -d simple.vw --tovw /dev/stdout --weight 1:2,0:0.1
0 0.1 | price:.23 sqft:.25 age:.05 2006
1 4.0 'second_house | price:.18 sqft:.15 age:.35 1976
0 0.1 0.5 'third_house | price:.53 sqft:.32 age:.87 1924

[fromvw_tovw__preprocessor]
$ vwoptimize.py -d simple.vw --tovw /dev/stdout --weight 1:2,0:0.1 --lowercase
preprocessor = --lowercase
0 0.1 | price:.23 sqft:.25 age:.05 2006
1 4.0 'second_house | price:.18 sqft:.15 age:.35 1976
0 0.1 0.5 'third_house | price:.53 sqft:.32 age:.87 1924

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

[remap_label_vw]
$ echo '2 | hello' | vwoptimize.py --remap_label 2:-1 --tovw /dev/stdout
-1 | hello

[remap_label_csv]
$ echo '2,hello' | vwoptimize.py --remap_label 2:-1 --tovw /dev/stdout --format csv
-1 | hello

[preprocessor_vw]
$ echo '| Hello:25\\n' | vwoptimize.py --tovw /dev/stdout --lowercase --quiet
| hello:25

[preprocessor_vw_namespace]
$ echo '|X Hello:25\\n' | vwoptimize.py --tovw /dev/stdout --lowercase --quiet
|X hello:25

[weight_vw]
$ echo '1 | Hello:25\\n' | vwoptimize.py --tovw /dev/stdout --quiet --weight 1:10
1 10.0 | Hello:25

[weight_vw_run]
$ echo '1 | Hello:25' | vwoptimize.py --weight 1:10 2>&1 | grep weighted
weighted example sum = 10.000000
weighted label sum = 10.000000

[weight_csv_run]
$ echo '1,hello' | vwoptimize.py --weight 1:10 --format csv 2>&1 | grep weighted
weighted example sum = 10.000000
weighted label sum = 10.000000

[weight_vw_run_file]
$ vwoptimize.py -d iris.vw --weight 1:10 2>&1 | grep weighted
weighted example sum = 600.000000
weighted label sum = 750.000000

[weight_csv_run_file]
$ vwoptimize.py -d simple.csv --weight 2:2 2>&1 | grep weighted
weighted example sum = 4.000000
weighted label sum = 6.000000

[weight_vw_run_opt]
$ vwoptimize.py -d iris.vw --weight 1:10 -b 10/20? 2>&1 | grep weighted
weighted example sum = 600.000000
weighted label sum = 750.000000

[weight_csv_run_opt]
$ vwoptimize.py -d simple.csv --weight 2:2 -b 10/20? 2>&1 | grep weighted
weighted example sum = 4.000000
weighted label sum = 6.000000

[max_words_0]
$ vwoptimize.py -d simple.csv --max_words 0 --tovw_simple /dev/stdout | sed 's/ //g'
1|
2|
1|

[max_words_1]
$ vwoptimize.py -d simple.csv --max_words 1 --tovw_simple /dev/stdout
1 | Hello first
2 | Goodbye second
1 | hello first

[max_words_2]
$ vwoptimize.py -d simple.csv --max_words 2 --tovw_simple /dev/stdout
1 | Hello World! first class
2 | Goodbye World. second class
1 | hello first class

[max_words_-1]
$ vwoptimize.py -d simple.csv --max_words -1 --tovw_simple /dev/stdout
1 | Hello first
2 | Goodbye second
1 |  first class

[max_words_size_4]
$ vwoptimize.py -d simple.csv --max_word_size 4 --tovw_simple /dev/stdout
1 | Hell Worl firs clas
2 | Good Worl seco clas
1 | hell firs clas agai

[max_words_size_0]
$ vwoptimize.py -d simple.csv --max_word_size 0 --tovw_simple /dev/stdout | sed 's/ //g'
1|
2|
1|

[max_words_size_-1]
$ vwoptimize.py -d simple.csv --max_word_size -1 --tovw_simple /dev/stdout
1 | Hell World firs clas
2 | Goodby World secon clas
1 | hell firs clas agai

[max_length_5]
$ vwoptimize.py -d simple.csv --max_length 5 --tovw_simple /dev/stdout
1 | Hello first
2 | Goodb secon
1 | hello first

[max_length_0]
$ vwoptimize.py -d simple.csv --max_length 0 --tovw_simple /dev/stdout | sed 's/ //g'
1|
2|
1|

[max_length_-2]
$ vwoptimize.py -d simple.csv --max_length -2 --tovw_simple /dev/stdout
1 | Hello Worl first cla
2 | Goodbye Worl second cla
1 | hel first class aga

[max_length_offset_3]
$ vwoptimize.py -d simple.csv --max_length_offset 3 --tovw_simple /dev/stdout
1 | lo World! st class
2 | dbye World. ond class
1 | lo st class again

[remove_duplicate_words]
$ printf '1 | one two one three\\n' | vwoptimize.py -d - --tovw /dev/stdout --remove_duplicate_words
preprocessor = --remove_duplicate_words
1 | one two three

[cleanup]
$ ls .vwoptimize
<BLANKLINE>
"""

import sys
__doc__ = __doc__.replace('vwoptimize.py', '%s ../vwoptimize.py' % sys.executable)
