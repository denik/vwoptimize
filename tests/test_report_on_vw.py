import unittest
from util import TestCase


class Test(TestCase):

    def test_predictions_from_stdin(self):
        self.assert_command_output('vw --quiet iris.vw --ect 3 -p /dev/stdout | PYTHON ../vwoptimize.py -d iris.vw --oaa 3 --report -p /dev/stdin --metric acc', '''
Found 3 integer classes: 1: 33.33%, 2: 33.33%, 3: 33.33%
acc: 0.726667
'''.lstrip())

    def test_weight_metric(self):
        self.assert_command_output('vw --quiet iris.vw --ect 3 -p /dev/stdout | PYTHON ../vwoptimize.py -d iris.vw --oaa 3 --weight_metric 3:0.2,2:0 --report -p /dev/stdin --metric acc', '''
Found 3 integer classes: 1: 33.33%, 2: 33.33%, 3: 33.33%
acc: 0.726667
'''.lstrip())


if __name__ == '__main__':
    unittest.main()
