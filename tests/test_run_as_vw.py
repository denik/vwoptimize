import sys
from util import *


class Test(TestCase):

    def test_vwoptimize_same_output(self):
        options = '-d simple.vw  --passes 10 -c -k --holdout_off'
        vw_output = grab_output('vw %s -p tmp/vw_pred -r tmp/vw_raw 2>&1 ' % options)
        os.rename('tmp/vw_pred', 'tmp/vw_pred_real')
        os.rename('tmp/vw_raw', 'tmp/vw_raw_real')
        vwopt_output = grab_output('%s ../vwoptimize.py %s -p tmp/vw_pred -r tmp/vw_raw 2>&1 ' % (sys.executable, options))
        self.assertMultiLineEqual(open('tmp/vw_pred_real').read(), open('tmp/vw_pred').read())
        self.assertMultiLineEqual(open('tmp/vw_raw_real').read(), open('tmp/vw_raw').read())
        self.assertMultiLineEqual('Found 2 integer classes: 0: 66.67%, 1: 33.33%\n' + vw_output, vwopt_output)


if __name__ == '__main__':
    unittest.main()
