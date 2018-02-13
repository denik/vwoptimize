# vwoptimize

Drop-in wrapper for [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit) that adds hyper-parameter tuning, more performance metrics, text preprocessing, reading from csv/tsv, feature extraction and k-fold cross-validation.

* [Hyper\-parameter tuning](#hyper-parameter-tuning)
  * [Specifying metric to optimize](#specifying-metric-to-optimize)
  * [Cross\-validation](#cross-validation)
* [Using vwoptimize\.py for model evaluation](#using-vwoptimizepy-for-model-evaluation)
* [Preprocessing the input](#preprocessing-the-input)
  * [Handling CSV/TSV inputs](#handling-csvtsv-inputs)
  * [Setting class weights](#setting-class-weights)
  * [Text processing](#text-processing)
* [Saving &amp; loading configuration](#saving--loading-configuration)

# Hyper-parameter tuning

vwoptimize.py can automatically select the best hyper-parameters, by doing grid search for discrete and Nelder-Mead search for continuous parameters or by optimizing all parameters at once using [hyperopt](https://github.com/hyperopt/hyperopt).

## Using grid-search

In order to enable grid search, list possible values separated by a slash and append a question mark. For example,

    $ python vwoptimize.py -d rcv1.train.vw  -b 22/24/26? --ngram 1/2? -f my.model

will try 6 configurations, select the one that gives the lowest progressive validation loss (reported by VW as `average loss`) and save the best model in "my.model" file.

## Using Nelder-Mead

If there is no slash but there is a question mark, the parameter is treated as a float and fine-tuned using Nelder-Mead algorithm from [scipy](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html):

    $ vwoptimize.py -d rcv1.train.vw -b 24 --ngram /2? --learning_rate 0.500? --l1 1e-07?

The number of digits after comma controls the precision of the tuner (if "0.500?" is specified then "0.500" and "0.501" might be tried but not "0.5005"). If the number is written in scientific notation ("1e-07?") then the search is done in log-space.

## Using hyperopt

Adding "--hyperopt N" enables optimization using hyperopt for N rounds. For that to work, one need to provide boundaries for each parameter. For example,

    $ vwoptimize.py -d rcv1.train.vw -b 24 --ngram 1..3? --learning_rate 0.100..5.000? --l1 1e-11..1e-2?

In order to select optimization algorithm, use --hyperopt_alg ALG where ALG can be "tpe" or "rand" or "package_name.module_name.function_name" for custom implementation of hyperopt's "suggest" method.

## Specifying metric to optimize

By default, vwoptimize.py reads the loss reported by Vowpal Wabbit and uses that as an optimization objective. It is also possible to specify custom metrics. For example, this will try different loss functions and select the one that gives the best accuracy:

    $ vwoptimize.py -d data.vw -b 28 --loss_function squared/hinge/logistic? --metric acc

In this case, vwoptimize.py will ask vowpal wabbit to store predictions in a temporary file and then use them together with true values extracted from data.vw to calculate the required metrics.

Some other metrics that can be used there: f1, brier, auc, logloss.

If the value of `--metric` is a comma-separated list, then only the first one is used as optimization objective and others are just printed for information. For example,

    $ vwoptimize.py -d data.vw -b 28 --loss_function squared/hinge/logistic? --metric acc,precision,recall,vw_average_loss

will perform the same optimization as previous but also report extra metrics for each run.

## Cross-validation

When doing multiple passes over data, the metrics reported by `--metric` are no longer suitable for tuning (vwoptimize.py does not automatically switches to using holdout set like VW itself does and thus ends up using predictions over already seen examples). K-fold cross validation avoids that by explicitly separating training and testing sets:

    $ vwoptimize.py -d data.vw -b 28 --loss_function squared/hinge/logistic? --kfold 5 -c -k --passes 1/2/5? --metric acc

This will calculate mean test-set accuracy over 5 different runs, each time using 80% of data for training and 20% for testing.

The `--kfold` option will not shuffle the dataset and will always use the same split.

## Using vwoptimize.py for model evaluation

The --metric option can be used without the optimizer, in a regular run:

    $ vwoptimize.py -d data.vw -i model -t --metric precision,recall,acc,auc

## Preprocessing the input

## Handling CSV/TSV inputs

vwoptimize.py can work with CSV/TSV files. One need to specify `--columnspec` that describes how to interpret each column:

    $ vwoptimize.py -d data.csv --columnspec y,weight,text_s,vw_w --ignoreheader --tovw /dev/stdout --lowercase --strip_punct
    1 10 |s hello |w YEAR:2017

    $ cat data.csv
    label,weight,message
    1,10,Hello!,YEAR:2017

In this example,

  * --ignoreheader tells vwoptimize.py to skip the first line of the input
  * --columnspec y,weight,text_s,vw_w describes how to interpret the 4 columns found in the csv file:
    - `y` means first column is used as a label
    - `weight` means second column is used as example weight
    - `text_s` means the third column is interpreted as text and put into namespace "s".
    - `vw_w` means the fourth column is interpreted as raw vowpal wabbit format and put into namespace "w". The difference between `vw` an `text` is that `text` has
      `:` and `|` removed from it as well as preprocessor options applied (`--lowercase --strip_punct` in this case), while `vw` is copied as is.
  * Finally, `--tovw /dev/stdout` tells to the preprocessed file to stdout and exit (rather than start training or tuning on the preprocessed input).

`vwoptimize.py` can recognize .csv, .tsv, .csv.gz, .tsv.gz and load them accordingly.

Other useful `--columnspec` values:

  * `drop` or empty string will ignore the field
  * `info` means put this message into tag section of .vw format
  * `weight_train` is like `--weight` but it is not taken into account when calculating the metrics with `--metric` (only affects training)
  * `weight_metric` is like `--weight` but it does not affect training (only affects `--metric`)

## Setting class weights

    $ vwoptimize.py -d data.vw --weight 1:0.5,2:0.9

This will multiply the weights of the examples that have label "1" by 0.5 and the examples with label "2" by 0.9. When used like this, it also affects the weights used when calculating the metrics.

The `--weight_train` option only affects the weights passed to VW but does not affect how the metrics are calculated. The `--weight_metric` does the opposite: it has not effect on training but it is used as sample weight when calculating metrics.

UPDATE: In recent VW, --classweight is equivalent to --weight_train.

## Text processing

The following text processing options are available:

  * `--lowercase` Lowercase the text.
  * `--strip_punct`  Strip punctuation.
  * `--stem`  Stem each word. Requires NLTK and pycld2.
  * `--split_chars`  Insert spaces between all characters.
  * `--max_word_size=MAX_WORD_SIZE`  Limit the word size to MAX_WORD_SIZE characters
  * `--max_words=MAX_WORDS`  Only keep the first MAX_WORDS words (applies individually on each column or namespaces)
  * `--max_length=MAX_LENGTH`  Only keep the first MAX_LENGTH characters
  * `--max_length_offset=MAX_LENGTH_OFFSET`  Ignore the first MAX_LENGTH_OFFSET characters
  * `--htmlunescape` Decode HTML-encoded characters (`&amp;` -> `&`)
  * `--NFKC` Normalize unicode characters
  * `--chinese_simplify` Convert Traditional Chinese characters into Simplified Chinese
  * `--split_ideographs` / `--split_hangul` / `--split_hiragana` / `--split_katakana` Insert spaces between the characters of the corresponding unicode script
  * `--split_combined` Combines the four split options above

## Saving & loading configuration

    $ vwoptimize.py -d data.csv --columnspec y,weight,text_s,vw_w --ignoreheader --lowercase --strip_punct --writeconfig my.config -f my.model

This will save the relevant preprocessing options (`--columnspec y,weight,text_s,vw_w --lowercase --strip_punct`) into my.config file, so that they don't have to be specified
when using the model to get the predictions:

    $ vwoptimize.py -d test_data.csv --readconfig my.config -t -p predictions.txt --metric precision,recall

It is especially important if the best preprocessing options are selected automatically. For example:

    $ vwoptimize.py -d data.csv --columnspec y,weight,text_s,vw_w --ignoreheader --lowercase? --strip_punct? --writeconfig my.config -f my.model

will grid search for best configuration of `--lowercase` & `--strip_punct`, use that to train the final model and store those settings in my.config so that the same transformations can be applied during prediction time.
