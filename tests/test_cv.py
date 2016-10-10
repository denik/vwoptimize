import unittest
from util import TestCase, system


class Test(TestCase):

    def test_multiclass_with_cache(self):
        self.assert_command_output('PYTHON ../vwoptimize.py -d iris.vw --ect 0 --cv --passes 2 -c -k --holdout_off --metric acc', '''
Found 3 integer classes: 1: 33.33%, 2: 33.33%, 3: 33.33%
cv acc: 0.893333
'''.lstrip())

    def test_regression(self):
        self.assert_command_output('PYTHON ../vwoptimize.py -d iris.vw --cv --metric mse', '''
Found 3 integer classes: 1: 33.33%, 2: 33.33%, 3: 33.33%
cv mse: 0.085602
'''.lstrip())

    def test_predictions(self):
        # also tests reading .vw from stdin
        cmd = 'PYTHON ../vwoptimize.py --format vw --cv --oaa 3 -p tmp/cv_predictions -r tmp/cv_raw_predictions -f tmp/model.vwoptimize --quiet --lesslogs'
        self.assert_command_output('head -n 10 iris.vw | ' + cmd, '')
        self.assertMultiLineEqual(open('tmp/cv_predictions').read(), open('iris.vw.10.cv_predictions').read())
        self.assertMultiLineEqual(open('tmp/cv_raw_predictions').read(), open('iris.vw.10.cv_raw_predictions').read())
        system('head -n 10 iris.vw | vw --oaa 3 -f tmp/model.vw --quiet')
        self.assertEqual(open('tmp/model.vwoptimize').read() == open('tmp/model.vw').read(), True)

    # def test_cv_predictions_stdout(self):
    #     grab_output('PYTHON ../vwoptimize.py -d iris.vw --ect 0 --cv --passes 2 -c -k --holdout_off --cv_predictions /dev/stdout')


if __name__ == '__main__':
    unittest.main()
