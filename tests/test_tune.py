from util import *


class Test(TestCase):

    def test_gridsearch_passes(self):
        self.assert_command_output('PYTHON ../vwoptimize.py -d iris.vw --ect 0 --passes 1/3/5? -c -k --holdout_off --metric acc --quiet', '''
Found 3 integer classes: 1: 33.33%, 2: 33.33%, 3: 33.33%
Result vw --ect 3 -c -k --holdout_off --quiet --passes 1... acc=0.9067
Result vw --ect 3 -c -k --holdout_off --quiet --passes 3... acc=0.9133*
Result vw --ect 3 -c -k --holdout_off --quiet --passes 5... acc=0.9467*
Best acc with 'no preprocessing': 0.9467*
Best preprocessor options: <none>
Best vw options: --ect 3 -c -k --holdout_off --quiet --passes 5
Best acc: 0.9467
'''.lstrip())

    def test_finetune_learning_rate(self):
        self.assert_command_output('PYTHON ../vwoptimize.py -d iris.vw --ect 0 --learning_rate 0.50? --metric acc --quiet', '''
Found 3 integer classes: 1: 33.33%, 2: 33.33%, 3: 33.33%
Result vw --ect 3 --quiet... acc=0.9067
Result vw --ect 3 --quiet --learning_rate 0.5... acc=0.9067
Result vw --ect 3 --quiet --learning_rate 0.53... acc=0.9067
Result vw --ect 3 --quiet --learning_rate 0.47... acc=0.9133*
Result vw --ect 3 --quiet --learning_rate 0.45... acc=0.9200*
Result vw --ect 3 --quiet --learning_rate 0.4... acc=0.9267*
Result vw --ect 3 --quiet --learning_rate 0.35... acc=0.9533*
Result vw --ect 3 --quiet --learning_rate 0.25... acc=0.9667*
Result vw --ect 3 --quiet --learning_rate 0.15... acc=0.9133
Result vw --ect 3 --quiet --learning_rate 0.3... acc=0.9600
Result vw --ect 3 --quiet --learning_rate 0.2... acc=0.9600
Result vw --ect 3 --quiet --learning_rate 0.27... acc=0.9667
Result vw --ect 3 --quiet --learning_rate 0.22... acc=0.9667
Result vw --ect 3 --quiet --learning_rate 0.26... acc=0.9667
Result vw --ect 3 --quiet --learning_rate 0.24... acc=0.9733*
Result vw --ect 3 --quiet --learning_rate 0.23... acc=0.9667
Best acc with 'no preprocessing': 0.9733*
Best preprocessor options: <none>
Best vw options: --ect 3 --quiet --learning_rate 0.24
Best acc: 0.9733
'''.lstrip())

    def test_tune_several(self):
        self.assert_command_output("""
            PYTHON ../vwoptimize.py --ect 0 -d iris.vw --loss_function squared/logistic? --learning_rate 0.50? --power_t 0.50? --metric acc --quiet --lesslogs
""".strip(), '''
Result vw --ect 3 --quiet --loss_function squared --learning_rate 0.5 --power_t 0.53... acc=0.9133*
Result vw --ect 3 --quiet --loss_function squared --learning_rate 0.47 --power_t 0.52... acc=0.9200*
Result vw --ect 3 --quiet --loss_function squared --learning_rate 0.45 --power_t 0.54... acc=0.9267*
Result vw --ect 3 --quiet --loss_function squared --learning_rate 0.45 --power_t 0.56... acc=0.9333*
Result vw --ect 3 --quiet --loss_function squared --learning_rate 0.42 --power_t 0.59... acc=0.9533*
Result vw --ect 3 --quiet --loss_function squared --learning_rate 0.37 --power_t 0.61... acc=0.9667*
Result vw --ect 3 --quiet --loss_function squared --learning_rate 0.35 --power_t 0.66... acc=0.9733*
Result vw --ect 3 --quiet --loss_function squared --learning_rate 0.36 --power_t 0.64... acc=0.9800*
Best acc with 'no preprocessing': 0.9800*
Best preprocessor options: <none>
Best vw options: --ect 3 --quiet --loss_function squared --learning_rate 0.36 --power_t 0.64
Best acc: 0.9800
'''.lstrip())


if __name__ == '__main__':
    unittest.main()
