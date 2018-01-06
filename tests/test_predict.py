"""
$ vwoptimize.py -d simpler.csv --quiet -p /dev/stdout
0
0.193097
0.395649

$ cat simpler.csv | vwoptimize.py -d - --format csv --quiet -p /dev/stdout
0
0.193097
0.395649

$ vwoptimize.py -d multiline.csv --quiet -p /dev/stdout --oaa 2
1
1
1

$ vwoptimize.py -d text_labels.csv --named_labels welcome,bye --quiet -p /dev/stdout --oaa 2
welcome
welcome
welcome

$ vwoptimize.py -d small_ag_news.csv --quiet --oaa 4 -f tmp_ag_news.model --lowercase
<BLANKLINE>

$ echo '| gulf states' | vw -i tmp_ag_news.model -t -p /dev/stdout --quiet
3

$ echo '| gulf states' | vw -i tmp_ag_news.model -t -r /dev/stdout --quiet
1:-0.122422 2:-0.111875 3:0.0379335 4:-0.124088

$ echo '| gulf states' | vw -i tmp_ag_news.model -t -p /dev/stdout -r /dev/stdout --quiet
1:-0.122422 2:-0.111875 3:0.0379335 4:-0.124088
3

$ echo '| gulf states' | vwoptimize.py -d - -i tmp_ag_news.model -t -p /dev/stdout --quiet
3

$ echo '| gulf states' | vwoptimize.py -d - -i tmp_ag_news.model -p /dev/stdout --quiet
3

$ echo '| gulf states' | vwoptimize.py -d - -i tmp_ag_news.model -r /dev/stdout --quiet
1:-0.122422 2:-0.111875 3:0.0379335 4:-0.124088

$ echo '| gulf states' | vwoptimize.py -d - -i tmp_ag_news.model -p /dev/stdout -r /dev/stdout --quiet
1:-0.122422 2:-0.111875 3:0.0379335 4:-0.124088
3

$ echo '1 | gulf states' | vwoptimize.py -d - -i tmp_ag_news.model --quiet --metric acc
acc = 0

$ echo '3 | gulf states' | vwoptimize.py -d - -i tmp_ag_news.model -t --metric acc 2>&1 | egrep 'average|acc'
average  since         example        example  current  current  current
average loss = 0.000000
acc = 1

# Same but now read csv

$ echo ',gulf states' | vwoptimize.py -d - --format csv -i tmp_ag_news.model -t -p /dev/stdout --quiet
3

$ echo ',gulf states' | vwoptimize.py -d - --format csv -i tmp_ag_news.model -p /dev/stdout --quiet
3

$ echo ',gulf states' | vwoptimize.py -d - --format csv -i tmp_ag_news.model -r /dev/stdout --quiet
1:-0.122422 2:-0.111875 3:0.0379335 4:-0.124088

$ echo ',gulf states' | vwoptimize.py -d - --format csv -i tmp_ag_news.model -p /dev/stdout -r /dev/stdout --quiet
1:-0.122422 2:-0.111875 3:0.0379335 4:-0.124088
3

$ echo ',gulf states' | vwoptimize.py -d - --format csv -i tmp_ag_news.model -p /dev/stdout --quiet --linemode
3

# write & read config
$ vwoptimize.py -d small_ag_news.csv --quiet --oaa 4 -f tmp_ag_news1.model --writeconfig tmp_ag_news1.config --lowercase
<BLANKLINE>

$ echo ',gulf states' | vwoptimize.py -d - --readconfig tmp_ag_news1.config -t -p /dev/stdout --quiet
3

$ echo ',GULF STATES' | vwoptimize.py -d - --readconfig tmp_ag_news1.config -t -p /dev/stdout --quiet
3

$ echo ',xgulf xstates' | vwoptimize.py -d - --readconfig tmp_ag_news1.config -t -p /dev/stdout --quiet
2

[cleanup]
$ ls .vwoptimize
<BLANKLINE>
"""
import sys
__doc__ = __doc__.replace('vwoptimize.py', '%s ../vwoptimize.py' % sys.executable)
