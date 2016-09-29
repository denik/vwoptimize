import unittest
from util import TestCase


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

    # def test_cv_predictions_stdout(self):
    #     grab_output('PYTHON ../vwoptimize.py -d iris.vw --ect 0 --cv --passes 2 -c -k --holdout_off --cv_predictions /dev/stdout')


if __name__ == '__main__':
    unittest.main()
