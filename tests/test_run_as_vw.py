import os
import unittest
from util import grab_output, TestCase


class Test(TestCase):

    def _test_vwoptimize_same_output(self, cmd):
        vw_output = grab_output('%s -p tmp/vw_pred -r tmp/vw_raw 2>&1 ' % cmd)
        os.rename('tmp/vw_pred', 'tmp/vw_pred_real')
        os.rename('tmp/vw_raw', 'tmp/vw_raw_real')
        vwoptimize = 'PYTHON ../vwoptimize.py'
        cmd = [vwoptimize if x == 'vw' else x for x in cmd.split()]
        cmd = ' '.join(cmd)
        assert vwoptimize in cmd, (vwoptimize, cmd)
        vwopt_output = grab_output('%s -p tmp/vw_pred -r tmp/vw_raw 2>&1 ' % cmd)
        self.assertMultiLineEqual(open('tmp/vw_pred_real').read(), open('tmp/vw_pred').read())
        self.assertMultiLineEqual(open('tmp/vw_raw_real').read(), open('tmp/vw_raw').read())
        self.assertMultiLineEqual('Found 2 integer classes: 0: 66.67%, 1: 33.33%\n' + vw_output, vwopt_output)

    def test_data(self):
        self._test_vwoptimize_same_output('vw -d simple.vw  --passes 10 -c -k --holdout_off')

    def broken_test_stdin(self):
        self._test_vwoptimize_same_output('cat simple.vw | vw --passes 10 -c -k --holdout_off')


if __name__ == '__main__':
    unittest.main()
