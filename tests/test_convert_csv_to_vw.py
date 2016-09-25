from util import *

MSG = 'Found 2 integer classes: 1: 66.67%, 2: 33.33%\n'


class Test(TestCase):

    def test_simple(self):
        system('%s ../vwoptimize.py -d simple.csv --tovw tmp/tmp.vw --columnspec y,text,text  2>&1 | tee tmp/out' % (sys.executable, ))
        self.assertMultiLineEqual(open('tmp/tmp.vw').read(), '''1 | Hello World! first class
2 | Goodbye World. second class
1 | hello first class again\n''')
        self.assertMultiLineEqual(open('tmp/out').read(), MSG)

    def _test_weight(self, opt_value, expected_out):
        for opt in ('--weight', '--weight_train'):
            out = grab_output('%s ../vwoptimize.py %s %s -d simple.csv --tovw tmp/tmp.vw --columnspec y,text,text 2>&1' % (sys.executable, opt, opt_value))
            self.assertMultiLineEqual(open('tmp/tmp.vw').read(), '''1 0.5 | Hello World! first class
2 | Goodbye World. second class
1 0.5 | hello first class again\n''')

            self.assertMultiLineEqual(out, expected_out)

    def test_weight(self):
        self._test_weight('1:0.5', MSG)

    def test_weight_balanced(self):
        self._test_weight('balanced', MSG + 'Calculated balanced weights: 1: 0.5 2: 1\n')


if __name__ == '__main__':
    unittest.main()
