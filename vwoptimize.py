#!/usr/bin/env python
"""Wrapper for Vowpal Wabbit that does cross-validation and hyper-parameter tuning"""
__version__ = '0.3.1'
import sys
import os
import optparse
import traceback
import math
import csv
import re
import subprocess
import time
import heapq
import json
import pprint
from pipes import quote
import numpy as np


csv.field_size_limit(10000000)
LOG_LEVEL = 1
MAIN_PID = str(os.getpid())
KEEPTMP = False
STDIN_NAMES = ('/dev/stdin', '-')


for path in '.vwoptimize /tmp/vwoptimize'.split():
    if os.path.exists(path):
        TMP_PREFIX = path
        break
    try:
        os.mkdir(path)
    except Exception, ex:
        sys.stderr.write('Failed to create %r: %s\n' % (path, ex))
    else:
        TMP_PREFIX = path
        break


def htmlparser_unescape(text, cache=[]):
    if not cache:
        import HTMLParser
        cache.append(HTMLParser.HTMLParser())
    return cache[0].unescape(text)


def unlink(*filenames):
    if KEEPTMP:
        return
    for filename in filenames:
        if not filename:
            continue
        if not isinstance(filename, basestring):
            sys.exit('unlink() expects list of strings: %r\n' % (filenames, ))
        if not os.path.exists(filename):
            continue
        try:
            os.unlink(filename)
        except Exception:
            sys.stderr.write('Failed to unlink %r\n' % filename)
            traceback.print_exc()


def kill(*jobs, **kwargs):
    verbose = kwargs.pop('verbose', False)
    assert not kwargs, kwargs
    for job in jobs:
        try:
            if job.poll() is None:
                if verbose:
                    log('Killing %s', job.pid)
                job.kill()
        except Exception, ex:
            if 'no such process' not in str(ex):
                sys.stderr.write('Failed to kill %r: %s\n' % (job, ex))


def open_regular_or_compressed(filename):
    if filename is None:
        return sys.stdin

    if hasattr(filename, 'read'):
        fobj = filename
    else:
        f = filename.lower()
        ext = f.rsplit('.', 1)[-1]
        if ext == 'gz':
            import gzip
            fobj = gzip.GzipFile(filename)
        elif ext == 'bz2':
            import bz2
            fobj = bz2.BZ2File(filename)
        elif ext == 'xz':
            import lzma
            fobj = lzma.open(filename)
        else:
            fobj = open(filename)
    return fobj


def get_real_ext(filename):
    filename = filename.rsplit('/', 1)[-1]
    items = filename.rsplit('.', 2)
    if len(items) >= 2 and items[-1] in 'gz bz2 xz'.split():
        return items[-2]
    return items[-1]


def get_temp_filename(suffix, counter=[0]):
    counter[0] += 1
    fname = '%s/%s.%s.%s' % (TMP_PREFIX, MAIN_PID, counter[0], suffix)
    assert not os.path.exists(fname), 'internal error: %s' % fname
    return fname


def log(s, *params, **kwargs):
    log_level = int(kwargs.pop('log_level', None) or 0)
    assert not kwargs, kwargs
    if log_level >= LOG_LEVEL:
        sys.stdout.flush()
        try:
            s = s % params
        except Exception:
            s = '%s %r' % (s, params)
        sys.stderr.write('%s\n' % (s, ))


def get_format_from_filename(filename):
    items = filename.lower().split('.')

    for ext in reversed(items):
        if ext in ['vw', 'csv', 'tsv']:
            return ext


def _read_lines_vw(fobj):
    for orig_line in fobj:
        if not orig_line.strip():
            continue
        items = orig_line.split('|')
        if len(items) <= 1:
            continue
        prefix_items = items[0].strip().split()
        if not prefix_items or not prefix_items or prefix_items[0].startswith("'"):
            yield (None, orig_line)
        else:
            yield (prefix_items[0], orig_line)


def _read_lines_csv(reader):
    expected_columns = None
    errors = 0

    for row in reader:
        if not row:
            continue

        bad_line = False

        if expected_columns is not None and len(row) != expected_columns:
            bad_line = 'Expected %s columns, got %s' % (expected_columns, len(row))

        if bad_line:
            log('Bad line (%s): %s', bad_line, limited_repr(row), log_level=3)
            errors += 1
            if errors >= 10:
                sys.exit('Too many errors while reading %s' % reader)
            continue

        errors = 0
        expected_columns = len(row)
        klass = row[0]
        klass = klass.replace(',', '_')
        yield row


def open_anything(source, format, ignoreheader):
    if format == 'vw':
        return _read_lines_vw(open_regular_or_compressed(source))

    if format == 'tsv':
        reader = csv.reader(open_regular_or_compressed(source), csv.excel_tab)
        if ignoreheader:
            reader.next()
    elif format == 'csv':
        reader = csv.reader(open_regular_or_compressed(source), csv.excel)
        if ignoreheader:
            reader.next()
    else:
        raise ValueError('format not supported: %s' % format)

    return _read_lines_csv(reader)


def limited_repr(obj, limit=80):
    s = repr(obj)
    if len(s) >= limit:
        s = s[:limit - 3] + '...'
    return s


class PassThroughOptionParser(optparse.OptionParser):

    def _process_args(self, largs, rargs, values):
        while rargs:
            try:
                optparse.OptionParser._process_args(self, largs, rargs, values)
            except (optparse.BadOptionError, optparse.AmbiguousOptionError), e:
                largs.append(e.opt_str)


def system(cmd, log_level=0, stdin=None):
    sys.stdout.flush()
    start = time.time()
    log('+ %s' % cmd, log_level=log_level)

    if stdin is not None:
        popen = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE)
        popen.communicate(stdin)
        retcode = popen.wait()
    else:
        retcode = os.system(cmd)

    if retcode:
        log('%s [%.1fs] %s', '-' if retcode == 0 else '!', time.time() - start, cmd, log_level=log_level - 1)
    if retcode:
        sys.exit(1)


def split_file(source, nfolds=None, ignoreheader=False, log_level=0, minfolds=1):
    if nfolds is None:
        nfolds = 10

    if isinstance(source, basestring):
        ext = get_real_ext(source)
    else:
        ext = 'xxx'

    if hasattr(source, 'seek'):
        source.seek(0)

    # XXX already have examples_count
    total_lines = 0
    for line in open_regular_or_compressed(source):
        total_lines += 1

    if hasattr(source, 'seek'):
        source.seek(0)

    source = open_regular_or_compressed(source)

    if ignoreheader:
        source.next()
        total_lines -= 1

    foldsize = int(math.ceil(total_lines / float(nfolds)))
    foldsize = max(foldsize, 1)
    orig_nfolds = nfolds
    nfolds = int(math.ceil(total_lines / float(foldsize)))
    if nfolds != orig_nfolds:
        log('Reduced number of folds from %r to %r', orig_nfolds, nfolds, log_level=1 + log_level)

    if minfolds is not None and nfolds <= minfolds:
        sys.exit('Too few folds: %r' % nfolds)

    folds = []

    current_fold = -1
    count = foldsize
    current_fileobj = None
    total_count = 0
    for line in source:
        if count >= foldsize:
            if current_fileobj is not None:
                current_fileobj.flush()
                os.fsync(current_fileobj.fileno())
                current_fileobj.close()
                current_fileobj = None
            current_fold += 1
            if current_fold >= nfolds:
                break
            fname = get_temp_filename('fold%s.%s' % (current_fold, ext))
            current_fileobj = open(fname, 'w')
            count = 0
            folds.append(fname)
        current_fileobj.write(line)
        count += 1
        total_count += 1

    if current_fileobj is not None:
        current_fileobj.flush()
        os.fsync(current_fileobj.fileno())
        current_fileobj.close()

    if total_count != total_lines:
        sys.exit('internal error: total_count=%r total_lines=%r source=%r' % (total_count, total_lines, source))

    return folds, total_lines


def _workers(workers):
    if workers is not None and workers <= 1:
        return 1
    if workers is None or workers <= 0:
        import multiprocessing
        return multiprocessing.cpu_count()
    return workers


def die_if_parent_dies(signum=9):
    if 'linux' not in sys.platform:
        return
    try:
        import ctypes
        libc = ctypes.CDLL('libc.so.6', use_errno=True)
        PR_SET_PDEATHSIG = 1
        result = libc.prctl(PR_SET_PDEATHSIG, signum)
        if result == 0:
            return True
        else:
            log('prctl failed: %s', os.strerror(ctypes.get_errno()))
    except StandardError, ex:
        sys.stderr.write(str(ex) + '\n')


def run_subprocesses(cmds, workers=None, log_level=None):
    workers = _workers(workers)
    cmds_queue = list(cmds)
    cmds_queue.reverse()
    queue = []
    while queue or cmds_queue:
        if cmds_queue and len(queue) <= workers:
            cmd = cmds_queue.pop()
            log('+ %s', cmd, log_level=log_level)
            popen = subprocess.Popen(cmd, shell=True, preexec_fn=die_if_parent_dies)
            popen._cmd = cmd
            queue.append(popen)
        else:
            popen = queue[0]
            del queue[0]
            retcode = popen.wait()
            if retcode:
                log('failed: %s', popen._cmd, log_level=3)
                kill(*queue, verbose=True)
                return False
            else:
                log('%s %s', '-' if retcode == 0 else '!', popen._cmd, log_level=log_level)

    return True


def vw_cross_validation(folds, vw_args, workers=None, p_fname=None, r_fname=None, audit=False):
    assert len(folds) >= 2, folds
    workers = _workers(workers)
    p_filenames = []
    r_filenames = []
    audit_filenames = []
    training_commands = []
    testing_commands = []
    models = []
    to_cleanup = []

    if '--quiet' not in vw_args:
        vw_args += ' --quiet'

    # verify that the the options are valid and the file format is not completely off
    training_command = 'head -n 10 %s | vw %s' % (folds[0], vw_args)
    if os.system(training_command) != 0:
        sys.exit(1)

    for test_fold in xrange(len(folds)):
        trainset = [fold for (index, fold) in enumerate(folds) if index != test_fold]
        assert trainset and os.path.exists(trainset[0]), trainset
        trainset = ' '.join(trainset)
        testset = folds[test_fold]
        assert testset and os.path.exists(testset), testset

        model_filename = get_temp_filename('model')
        assert not os.path.exists(model_filename), 'internal error: temporary file already exists: %r' % model_filename
        models.append(model_filename)

        p_filename = '%s.predictions' % model_filename
        with_p = '-p %s' % p_filename
        p_filenames.append(p_filename)

        if r_fname:
            r_filename = '%s.raw' % model_filename
            with_r = '-r %s' % r_filename
            r_filenames.append(r_filename)
        else:
            r_filename = None
            with_r = ''

        my_args = vw_args

        cache_file = None
        if '-c' in my_args.split() and '--cache_file' not in my_args:
            my_args = my_args.replace('-c', '')
            cache_file = get_temp_filename('cache')
            my_args += ' --cache_file %s' % cache_file
            to_cleanup.append(cache_file)

        if audit:
            audit_filename = '%s.audit' % model_filename
            audit = '-a > %s' % audit_filename
            audit_filenames.append(audit_filename)
        else:
            audit_filename = ''
            audit = ''

        training_command = 'cat %s | vw -f %s %s' % (trainset, model_filename, my_args)
        testing_command = 'vw --quiet -d %s -t -i %s %s %s %s' % (testset, model_filename, with_p, with_r, audit)
        training_commands.append(training_command)
        testing_commands.append(testing_command)

    try:
        success = run_subprocesses(training_commands, workers=workers, log_level=-1)

        for name in models:
            if not os.path.exists(name):
                sys.exit('vw failed to write a model')

        if success:
            success = run_subprocesses(testing_commands, workers=workers, log_level=-1)

        if not success:
            sys.exit('vw failed')

        predictions = []
        for fname in p_filenames:
            predictions.extend(_load_first_float_from_each_string(fname))

        if p_fname and p_filenames:
            system('cat %s > %s' % (' '.join(p_filenames), p_fname), log_level=-1)

        if r_fname and r_filenames:
            system('cat %s > %s' % (' '.join(r_filenames), r_fname), log_level=-1)

        return np.array(predictions)

    finally:
        unlink(*p_filenames)
        unlink(*r_filenames)
        unlink(*audit_filenames)
        unlink(*to_cleanup)
        unlink(*models)


def _load_first_float_from_each_string(file):
    filename = file
    if hasattr(file, 'read'):
        pass
    elif isinstance(file, basestring):
        file = open(file)
    else:
        raise AssertionError(limited_repr(file))

    result = []

    for line in file:
        try:
            result.append(float(line.split()[0]))
        except:
            sys.stderr.write('Error while parsing %r\nin %r\n' % (line, filename))
            raise

    return result


class BaseParam(object):

    PRINTABLE_KEYS = 'opt init min max values format extra'.split()
    _cast = float

    @classmethod
    def cast(cls, value):
        if value is None:
            return None
        if value == '':
            return None
        return cls._cast(value)

    def pack(self, value):
        if self._pack is None:
            return value
        return self._pack(value)

    def unpack(self, value):
        if self._unpack is None:
            return value
        return self._unpack(value)

    def __init__(self, opt, init=None, min=None, max=None, format=None, pack=None, unpack=None, extra=None):
        self.opt = opt
        self.init = self.cast(init)
        self.min = self.cast(min)
        self.max = self.cast(max)
        self.format = format
        self._pack = pack
        self._unpack = unpack
        self.extra = None

        if self.init is None:
            if self.min is not None and self.max is not None:
                self.init = self.avg(self.min, self.max)
            elif self.min is not None:
                self.init = self.min
            elif self.max is not None:
                self.init = self.max

    def avg(self, a, b):
        result = self.cast(self.unpack((self.pack(self.min) + self.pack(self.max)) / 2.0))
        if self.format:
            result = self.format % result
            result = self.cast(result)
        return result

    def __repr__(self):
        klass = type(self).__name__
        items = []
        for name in self.PRINTABLE_KEYS:
            value = getattr(self, name, None)
            if value is not None:
                items.append('%s=%r' % (name, value))
        return klass + '(' + ', '.join(items) + ')'

    def packed_init(self):
        init = self.init
        init = self.pack(init)
        return init

    def get_extra_args(self, param):
        if param is None or param == '':
            return None
        param = self.unpack(param)
        if self.min is not None and param <= self.min:
            param = self.min
        elif self.max is not None and param >= self.max:
            param = self.max
        format = self.format or '%s'
        extra_arg = format % param
        return self.opt + ' ' + extra_arg + ' '.join(self.extra or [])


class IntegerParam(BaseParam):
    _cast = int


class FloatParam(BaseParam):
    pass


class LogParam(FloatParam):

    def __init__(self, opt, **kwargs):
        FloatParam.__init__(self, opt, pack=np.log, unpack=np.exp, **kwargs)


class ValuesParam(BaseParam):

    def __init__(self, opt, values, **kwargs):
        BaseParam.__init__(self, opt, **kwargs)
        self.values = values

    def enumerate_all(self):
        return [self.get_extra_args(x) for x in self.values]


class BinaryParam(BaseParam):

    def __init__(self, opt, **kwargs):
        BaseParam.__init__(self, opt, **kwargs)

    def enumerate_all(self):
        return ['', self.opt]


PREPROCESSING_BINARY_OPTS = ['--%s' % x for x in 'htmlunescape lowercase strip_punct stem'.split()]


def get_format(value):
    """
    >>> get_format("1e-5")
    '%.0e'

    >>> get_format("1e5")
    '%.0e'

    >>> get_format("0.")
    '%.0g'

    >>> get_format("0.5")
    '%.1g'

    >>> get_format("0.5")
    '%.1g'

    >>> get_format("0.50")
    '%.2g'

    >>> get_format('5')
    """
    value = value.lower()

    if 'e' in value and '.' not in value:
        return '%.0e'

    x = value

    if '.' in x:
        x = x.split('.')[-1]

    if 'e' in x:
        x = x.split('e')[0]
        return '%%.%se' % len(x)

    if '.' in value:
        return '%%.%sg' % len(x)


DEFAULTS = {
    '--ngram': {
        'min': 1
    },
    '--l1': {
        'min': 1e-11
    },
    '--learning_rate': {
        'min': 0.000001
    }
}


def get_tuning_config(config):
    """
    >>> get_tuning_config('--lowercase?')
    BinaryParam(opt='--lowercase')

    >>> get_tuning_config('--ngram 2?')
    IntegerParam(opt='--ngram', init=2, min=1)

    >>> get_tuning_config('--ngram 2..?')
    IntegerParam(opt='--ngram', init=2, min=2)

    >>> get_tuning_config('--ngram 2..5?')
    IntegerParam(opt='--ngram', init=3, min=2, max=5)

    >>> get_tuning_config('-b 10..25?')
    IntegerParam(opt='-b', init=17, min=10, max=25)

    >>> get_tuning_config('--learning_rate 0.5?')
    FloatParam(opt='--learning_rate', init=0.5, min=1e-06, format='%.1g')

    >>> get_tuning_config('--learning_rate 0.50?')
    FloatParam(opt='--learning_rate', init=0.5, min=1e-06, format='%.2g')

    >>> get_tuning_config('--l1 1e-07?')
    LogParam(opt='--l1', init=1e-07, min=1e-11, format='%.0e')

    >>> get_tuning_config('--l1 1.0E-07?')
    LogParam(opt='--l1', init=1e-07, min=1e-11, format='%.1e')

    >>> get_tuning_config('--l1 ..1.2e-07..?')
    LogParam(opt='--l1', init=1.2e-07, format='%.1e')

    >>> get_tuning_config('--l1 1e-10..1e-05?')
    LogParam(opt='--l1', init=3e-08, min=1e-10, max=1e-05, format='%.0e')

    >>> get_tuning_config('--loss_function squared/hinge/percentile?')
    ValuesParam(opt='--loss_function', values=['squared', 'hinge', 'percentile'])

    >>> get_tuning_config('--loss_function /hinge/percentile?')
    ValuesParam(opt='--loss_function', values=['', 'hinge', 'percentile'])
    """
    if isinstance(config, basestring):
        config = config.split()

    if len(config) > 2:
        raise ValueError('Cannot parse: %r' % (config, ))

    first = config[0]

    assert first.startswith('-'), config

    if first.startswith('--'):
        prefix = '--'
        first = first[2:]
    else:
        prefix = '-'
        first = first[1:]

    if len(config) == 1:
        first = first[:-1]
        if '/' in first:
            return ValuesParam(opt='', values=[(prefix + x if x else '') for x in first.split('/')])
        else:
            return BinaryParam(prefix + first)

    value = config[-1]
    value = value[:-1]

    if '/' in value:
        return ValuesParam(opt=config[0], values=value.split('/'))

    is_log = 'e' in value.lower()

    if value.count('..') == 2:
        min, init, max = value.split('..')
        format = sorted([get_format(min), get_format(init), get_format(max)])[-1]
        is_float = '.' in min or '.' in init or '.' in max

        params = {
            'opt': config[0],
            'min': min,
            'init': init,
            'max': max,
            'format': format
        }

    elif '..' in value:
        min, max = value.split('..')
        is_float = '.' in min or '.' in max
        format = sorted([get_format(min), get_format(max)])[-1]

        params = {
            'opt': config[0],
            'min': min,
            'max': max,
            'format': format
        }

    else:
        is_float = '.' in value
        format = get_format(value)

        params = {
            'opt': config[0],
            'init': value,
            'format': format
        }

    for key, value in DEFAULTS.get(config[0], {}).items():
        if key not in params:
            params[key] = value

    if is_log:
        type = LogParam
    elif is_float:
        type = FloatParam
    else:
        type = IntegerParam

    return type(**params)


def vw_optimize_over_cv(vw_filename, folds, args, metric, config,
                        predictions_filename=None, raw_predictions_filename=None, workers=None, other_metrics=[]):
    # we only depend on scipy if parameter tuning is enabled
    import scipy.optimize

    # predictions_filename is unused currently

    gridsearch_params = []
    tunable_params = []
    base_args = []
    assert isinstance(args, list), args

    for param in args:
        if isinstance(param, ValuesParam):
            gridsearch_params.append(param)
        elif isinstance(param, BaseParam):
            tunable_params.append(param)
        else:
            base_args.append(param)

    if predictions_filename:
        predictions_filename_tmp = predictions_filename + '.tuning'
    else:
        predictions_filename_tmp = get_temp_filename('predictions')
    if raw_predictions_filename:
        raw_predictions_filename_tmp = raw_predictions_filename + '.tuning'
    else:
        raw_predictions_filename_tmp = None

    extra_args = ['']
    cache = {}
    best_result = [None, None]

    y_true = _load_first_float_from_each_string(vw_filename)
    y_true = np.array(y_true)

    def run(params):
        log('Parameters: %r', params)
        args = extra_args[:]

        for param_config, param in zip(tunable_params, params):
            extra_arg = param_config.get_extra_args(param)
            if extra_arg:
                args.append(extra_arg)

        args = ' '.join(str(x) for x in args)
        args = re.sub('\s+', ' ', args).strip()

        if args in cache:
            return cache[args]

        log('Trying vw %s...', args)

        try:
            # XXX use return value
            vw_cross_validation(
                folds,
                args,
                workers=workers,
                p_fname=predictions_filename_tmp,
                r_fname=raw_predictions_filename_tmp)
        except BaseException, ex:
            log(str(ex))
            log('Result vw %s... error', args, log_level=1)
            cache[args] = 0.0
            return 0.0

        y_pred = _load_first_float_from_each_string(predictions_filename_tmp)
        y_pred = np.array(y_pred)
        assert len(y_true) == len(y_pred), (vw_filename, len(y_true), predictions_filename_tmp, len(y_pred), os.getpid())
        result = calculate_score(metric, y_true, y_pred, config)

        if not is_loss(metric):
            result = -result

        is_best = ''
        if best_result[0] is None or result < best_result[0]:
            is_best = '*' if best_result[0] is not None else ''
            best_result[0] = result
            best_result[1] = args
            if predictions_filename:
                os.rename(predictions_filename_tmp, predictions_filename)
            if raw_predictions_filename:
                os.rename(raw_predictions_filename_tmp, raw_predictions_filename)

        unlink(predictions_filename_tmp, raw_predictions_filename_tmp)

        def frmt(x):
            if isinstance(x, float):
                return '%.4f' % x
            return str(x)

        other_results = ' '.join(['%s=%s' % (m, frmt(calculate_score(m, y_true, y_pred, config))) for m in other_metrics])
        if other_results:
            other_results = '  ' + other_results

        log('Result vw %s... %s=%.4f%s%s', args, metric, abs(result), is_best, other_results, log_level=1 + bool(is_best))

        cache[args] = result
        return result

    already_done = {}

    log('Grid-search: %r', gridsearch_params)

    for params in expand(gridsearch_params):
        params_normalized = vw_normalize_params(base_args + params)
        if params_normalized != params:
            log('Normalized params %r %r -> %r', base_args, params, params_normalized, log_level=-1)
        params_as_str = ' '.join(params_normalized)
        if params_as_str in already_done:
            log('Skipping %r (same as %r)', ' '.join(params), ' '.join(already_done[params_as_str]), log_level=-1)
            continue
        already_done[params_as_str] = params

        extra_args[0] = params_as_str
        try:
            run([None] * len(tunable_params))
        except Exception:
            traceback.print_exc()
            continue

        scipy.optimize.minimize(run, [x.packed_init() for x in tunable_params], method='Nelder-Mead', options={'xtol': 0.001, 'ftol': 0.001})

    return best_result


def vw_normalize_params(params):
    """
    >>> vw_normalize_params(['--ngram', '1'])
    []
    >>> vw_normalize_params(['--ngram', '1', '--skips', '1'])
    []
    >>> vw_normalize_params(['--skips', '1'])
    []
    >>> vw_normalize_params(['--ngram', '2', '--skips', '1'])
    ['--ngram', '2', '--skips', '1']
    """
    params = ' '.join(params)
    params = params.replace('--ngram 1', '')
    if '--ngram' not in params:
        params = re.sub('--skips \d+', '', params)
    params = re.sub('\s+', ' ', params)
    return params.split()


def expand(gridsearch_params, only=None):
    for item in _expand(gridsearch_params, only=only):
        yield [x for x in item if x]


def _expand(gridsearch_params, only=None):
    if not gridsearch_params:
        yield []
        return

    first_arg = gridsearch_params[0]

    if isinstance(first_arg, basestring):
        skip = True
    elif only is not None and getattr(first_arg, 'opt', '') not in only:
        skip = True
    else:
        skip = False

    if skip:
        for inner in _expand(gridsearch_params[1:], only=only):
            yield [first_arg] + inner
        return

    for first_arg_variant in first_arg.enumerate_all():
        for inner in _expand(gridsearch_params[1:], only=only):
            yield [first_arg_variant] + inner


def get_language(doc):
    import pycld2
    if isinstance(doc, unicode):
        doc = doc.encode('utf-8')
    try:
        return pycld2.detect(doc, bestEffort=True)[2][0][0].lower()
    except Exception, ex:
        sys.stderr.write('Cannot detect language of %r\n%s\n' % (doc, ex))


def get_stemmer(language, stemmers={}):
    if language in stemmers:
        return stemmers[language]
    from nltk.stem import SnowballStemmer
    try:
        stemmers[language] = SnowballStemmer(language)
    except Exception:
        stemmers[language] = 0

    return stemmers[language]


def stem_words(words):
    base_stemmer = False
    result = []
    for word in words:
        if len(word) > 2:
            language = None
            try:
                language = get_language(word)
                stemmer = get_stemmer(language)
                if stemmer:
                    word = stemmer.stem(word)
                else:
                    if base_stemmer is False:
                        base_language = get_language(' '.join(words))
                        base_stemmer = get_stemmer(base_language)
                    if base_stemmer:
                        language = base_language
                        word = base_stemmer.stem(word)
            except Exception, ex:
                sys.stderr.write('Cannot stem %r %r: %s\n' % (language, word, ex))
        result.append(word)
    return result


class Preprocessor(object):
    ALL_OPTIONS = 'htmlunescape lowercase strip_punct stem'.split()

    @classmethod
    def parse_options(cls, string):
        parser = PassThroughOptionParser()
        for opt in PREPROCESSING_BINARY_OPTS:
            parser.add_option(opt, action='store_true')
        options, args = parser.parse_args(string.split())
        return options.__dict__

    @classmethod
    def from_options(cls, options):
        if isinstance(options, list):
            options = ' '.join(x for x in options if isinstance(x, basestring))

        if isinstance(options, basestring):
            options = cls.parse_options(options)

        for opt in cls.ALL_OPTIONS:
            if options[opt]:
                break
        else:
            return None

        return cls(**options)

    def to_options(self):
        return ['--%s' % opt for opt in self.ALL_OPTIONS if getattr(self, opt, None)]

    def __init__(self, htmlunescape=False, lowercase=False, strip_punct=False, stem=False, replace_currency=False, replace_numbers=False, normalize_space=True, **ignored):
        self.normalize_space = normalize_space
        self.htmlunescape = htmlunescape
        self.lowercase = lowercase
        self.strip_punct = strip_punct
        self.stem = stem
        if self.stem:
            stem_words(["testing"])
            self.lowercase = True
            self.strip_punct = True

    def __str__(self):
        return ' '.join(self.to_options())

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, ', '.join('%s=%r' % (name, getattr(self, name, None)) for name in self.ALL_OPTIONS))

    def process_text(self, text):
        orig = text
        try:
            text = text.decode('utf-8', errors='ignore')

            # quite costly
            # if self.normalize_space:
            #     text = u''.join(u' ' if unicodedata.category(x)[:1] in 'CZ' else x for x in text)

            if self.htmlunescape:
                text = htmlparser_unescape(text)

            if self.lowercase:
                text = text.lower()

            if self.strip_punct:
                words = re.findall(r"(?u)\b\w\w+\b", text)
            else:
                words = text.split()

            if self.stem:
                words = stem_words(words)

            text = u' '.join(words)
            return text.encode('utf-8')
        except Exception:
            sys.stderr.write('Failed to process\norig=%r\ntext=%r\n' % (orig, text))
            traceback.print_exc()
            raise

    def process_row(self, row):
        assert isinstance(row, list), row
        return [self.process_text(item) for item in row]

    def process_rows(self, rows):
        return [self.process_row(row) for row in rows]


def read_labels(filename, source, format, n_classes, label_index, ignoreheader):
    labels_counts = {}
    examples_count = 0

    log('Reading labels from %s', filename or 'stdin')
    rows_source = open_anything(filename or source, format, ignoreheader=ignoreheader)

    all_integers = True
    all_floats = True
    y_true = []

    for row in rows_source:
        label = row[label_index]
        y_true.append(label)
        n = labels_counts[label] = labels_counts.get(label, 0) + 1
        examples_count += 1
        if n == 1:
            if all_integers is True:
                try:
                    int(label)
                except Exception:
                    all_integers = False
            if all_floats is True:
                try:
                    float(label)
                except Exception:
                    all_floats = False

    if hasattr(source, 'seek'):
        source.seek(0)

    log('Counted %r examples', examples_count)

    if not labels_counts:
        sys.exit('empty: %s' % filename)

    labels = labels_counts.keys()

    def format_classes():
        classes = [(-count, name) for (name, count) in labels_counts.items()]
        classes.sort()
        classes = ['%s: %.2f%%' % (name, -100.0 * count / examples_count) for (count, name) in classes]
        if len(classes) > 6:
            del classes[3:-3]
            classes[3:3] = ['...']
        return ', '.join(classes)

    if all_integers:
        labels = [int(x) for x in labels]
        labels.sort()
        max_label = labels[-1]
        if n_classes:
            if n_classes != len(labels):
                log('Expected %r classes, but found %r', n_classes, len(labels))
        if n_classes == 0:
            n_classes = max_label
        log('Found %r integer classes: %s', len(labels), format_classes(), log_level=1)

        # no mapping in this case
        labels = None
        y_true = np.array([int(x) for x in y_true])

    elif all_floats:
        labels = [float(x) for x in labels]
        log('Found float responses: %s..%s', min(labels), max(labels), log_level=1)
        if n_classes is not None:
            sys.exit('Float responses, not compatible with multiclass')
        labels = None
        y_true = np.array([float(x) for x in y_true])

    else:
        log('Found %r textual labels: %s', len(labels), format_classes(), log_level=1)

        if n_classes is None and len(labels) != 2:
            sys.exit('Found textual labels, expecting multiclass option. Pass 0 to auto-set number of classes to %r, e.g. "--oaa 0"' % len(labels))

        n_classes = len(labels)

        labels.sort()

    return labels_counts, y_true, labels, n_classes


def _make_proper_list(s, type=None):
    if not s:
        return s

    if isinstance(s, basestring):
        result = s.split(',')
        if type is not None:
            result = [type(x) for x in result]
        return result

    result = []
    if isinstance(s, list):
        for x in s:
            result.extend(_make_proper_list(x, type))
    else:
        raise TypeError('Expected list of string: %r' % (s, ))
    return result


def parse_weight(config, labels=None):
    """
    >>> parse_weight('A:B:2', ['A:B', 'another_label'])
    {'A:B': 2.0}

    >>> parse_weight('A:B:2')
    {'A:B': 2.0}
    """
    if not config:
        return None

    if config == ['balanced']:
        return config

    if labels and not isinstance(labels, list):
        raise TypeError('must be list, not %r' % type(labels))

    config = _make_proper_list(config)

    if not config:
        return None

    result = {}

    for item in config:
        if ':' not in item:
            sys.exit('Weight must be specified as CLASS:WEIGHT, cannot parse %r' % item)
        label, weight = item.rsplit(':', 1)

        if labels is not None and label not in labels:
            sys.exit('Label %r is not recognized. Expected: %r' % (label, labels))

        try:
            weight = float(weight)
        except Exception:
            weight = None

        if weight is None or weight < 0:
            sys.exit('Weight must be specified as CLASS:WEIGHT(float), %r is not recognized' % (item, ))

        if label in result:
            sys.exit('Label %r specified more than once' % label)
        result[label] = weight

    return result


def get_sample_weight(y_true, config):
    if config is None:
        return None
    N = len(y_true)
    result = np.zeros(N)
    updated = np.zeros(N)

    for klass, weight in config.items():
        assert isinstance(klass, (int, long, float)), [klass]
        result += np.multiply(np.ones(N) * weight, y_true == klass)
        updated += y_true == klass

    result += (updated == 0)

    return result


def get_balanced_weights(labels_counts):
    min_count = float(min(labels_counts.values()))

    result = {}
    for label in labels_counts:
        result[label] = min_count / labels_counts[label]

    log('Calculated balanced weights: %s', ' '.join('%s: %g' % (k, w) for (k, w) in sorted(result.items())), log_level=1)

    return result


def convert_row_to_vw(row, columnspec, preprocessor=None, labels=None, weights=None):
    assert isinstance(columnspec, list), columnspec

    # labels maps user labels into VW label. Can be None.
    assert labels is None or isinstance(labels, dict), labels

    # weights maps user label into weight. Can be None.
    assert labels is None or isinstance(weights, dict), weights

    if len(row) != len(columnspec):
        sys.exit('Expected %r columns (%r), got %r (%r)' % (len(columnspec), columnspec, len(row), row))

    y = None
    x = []
    info = []
    for item, spec in zip(row, columnspec):
        if spec == 'y':
            y = item
        elif spec == 'text':
            x.append(item)
        elif spec == 'info':
            info.append(item)
        elif spec == 'drop' or not spec:
            continue
        else:
            sys.exit('Spec item %r not understood' % spec)

    if info:
        info = " '%s" % ';'.join(info) + ' '
    else:
        info = ''

    if labels:
        vw_y = labels.get(y)
        if vw_y is None:
            sys.exit('Unexpected label: %s', limited_repr(y), log_level=2)
    else:
        if y is None:
            vw_y = ''
        else:
            vw_y = y

    if weights is not None:
        weight = weights.get(y, 1)
        if weight == 1:
            weight = ''
        else:
            weight = str(weight) + ' '
    else:
        weight = ''

    if preprocessor:
        x = preprocessor.process_row(x)
        text = '  '.join(x)
    else:
        text = ' '.join(x)
    text = text.replace(':', ' ').replace('|', ' ')
    text = text.strip()
    text = '%s %s%s| %s\n' % (vw_y, weight, info, text)
    return text


def _convert_any_to_vw(source, format, output, labels, weights, preprocessor, columnspec, ignoreheader):
    assert format != 'vw'
    assert isinstance(columnspec, list)

    if labels:
        labels = dict((label, 1 + labels.index(label)) for label in labels)

    rows_source = open_anything(source, format, ignoreheader=ignoreheader)
    output = open(output, 'wb')

    for row in rows_source:
        vw_line = convert_row_to_vw(row, columnspec, preprocessor=preprocessor, labels=labels, weights=weights)
        output.write(vw_line)

    output.flush()
    os.fsync(output.fileno())
    output.close()


def convert_any_to_vw(source, format, output_filename, columnspec, labels, weights, preprocessor, ignoreheader, workers):
    assert format != 'vw'
    preprocessor = preprocessor or ''

    assert isinstance(preprocessor, basestring), preprocessor

    start = time.time()

    workers = _workers(workers)
    batches, total_lines = split_file(source, nfolds=workers, ignoreheader=ignoreheader, log_level=-1, minfolds=None)

    batches_out = [x + '.out' for x in batches]

    labels = ','.join(labels or [])

    try:
        commands = []

        common_cmd = [quote(sys.executable), quote(__file__), '--format', format]

        if labels:
            common_cmd += ['--labels', quote(labels)]

        if weights:
            weights = ['%s:%s' % (x, weights[x]) for x in weights if weights[x] != 1]
            weights = ','.join(weights)
            common_cmd += ['--weight', quote(weights)]

        if columnspec:
            common_cmd += ['--columnspec', quote(','.join(str(x) for x in columnspec))]

        common_cmd.append(preprocessor)

        for batch in batches:
            cmd = common_cmd + ['--tovw_simple', batch + '.out', '-d', batch]
            commands.append(' '.join(cmd))

        if not run_subprocesses(commands, workers=workers, log_level=-1):
            sys.exit(1)

        cmd = 'cat ' + ' '.join(batches_out)
        if output_filename:
            cmd += ' > %s' % output_filename

        system(cmd, log_level=-1)

    finally:
        unlink(*batches)
        unlink(*batches_out)

    took = time.time() - start
    log('Generated %s in %.1f seconds', output_filename, took)
    if not output_filename.startswith('/dev/'):
        log('\n'.join(open(output_filename).read(200).split('\n')) + '...')


metrics_shortcuts = {
    'mse': 'mean_squared_error',
    'auc': 'roc_auc_score',
    'brier': 'brier_score_loss',
    'acc': 'accuracy_score',
    'precision': 'precision_score',
    'recall': 'recall_score',
    'f1': 'f1_score',
    'cm': 'confusion_matrix',
}

metrics_param = {
    'mean_squared_error': 'y_score',
    'roc_auc_score': 'y_score',
    'brier_score_loss': 'y_prob',
    'accuracy_score': 'y_pred',
    'precision_score': 'y_pred',
    'recall_score': 'y_pred',
    'f1_score': 'y_pred',
    'confusion_matrix': 'y_pred',
}


def is_loss(metric_name):
    if metric_name.endswith('_w'):
        metric_name = metric_name[:-2]
    metric_name = metrics_shortcuts.get(metric_name, metric_name)
    if metric_name.endswith('_loss') or metric_name.endswith('_error'):
        return True


def calculate_score(metric, y_true, y_pred, config):
    sample_weight = get_sample_weight(y_true, config.get('weight_metric'))

    n_classes = config.get('n_classes')
    threshold = config.get('threshold')
    min_label = config.get('min_label')
    max_label = config.get('max_label')

    if n_classes is None and threshold is None:
        sys.exit('Bad config: missing n_classes and threshold:\n%s' % pprint.pformat(config))

    if n_classes is not None and threshold is not None:
        sys.exit('Bad config: both n_classes and threshold are present, expected only one:\n%s' % pprint.pformat(config))

    extra_args = {'sample_weight': sample_weight}
    if metric.endswith('_w'):
        metric = metric[:-2]
    else:
        extra_args = {}

    fullname = metrics_shortcuts.get(metric)

    import sklearn.metrics
    func = getattr(sklearn.metrics, fullname)

    metric_type = metrics_param.get(fullname)

    if metric_type == 'y_prob':
        # brier_score_loss
        assert threshold is not None, 'Cannot apply %s/%s on multiclass' % (metric, fullname)
        assert min_label is not None and max_label is not None, config
        delta = float(max_label - min_label)
        assert delta
        y_true = (y_true - min_label) / delta
        y_pred = (y_pred - min_label) / delta
        y_pred = np.minimum(y_pred, 1)
        y_pred = np.maximum(y_pred, 0)
        return func(y_true, y_pred, **extra_args)
    elif metric_type == 'y_score':
        # auc, mse
        assert threshold is not None, 'Cannot apply %s/%s on multiclass' % (metric, fullname)
        return func(y_true, y_pred, **extra_args)
    elif metric_type == 'y_pred':
        if threshold is not None:
            y_true_norm = y_true > threshold
            y_pred_norm = y_pred > threshold
            return func(y_true_norm, y_pred_norm, **extra_args)
        else:
            return func(y_true, y_pred, **extra_args)
    else:
        sys.exit('Unknown metric: %r' % metric)


def main_tune(metric, config, source, format, args, preprocessor_base, nfolds, ignoreheader, workers):
    if preprocessor_base is None:
        preprocessor_base = []
    else:
        preprocessor_base = preprocessor_base.to_options()

    if not metric:
        sys.exit('Provide metric to optimize for with --metric auc|acc|mse|f1')

    optimization_metric = metric[0]
    other_metrics = metric[1:]

    best_preprocessor_opts = None
    best_vw_options = None
    best_result = None
    already_done = {}

    preprocessor_variants = list(expand(args, only=PREPROCESSING_BINARY_OPTS))
    log('Trying preprocessor variants: %s', pprint.pformat(preprocessor_variants), log_level=-1)

    for my_args in preprocessor_variants:
        preprocessor = Preprocessor.from_options(preprocessor_base + my_args)
        preprocessor_opts = ' '.join(preprocessor.to_options() if preprocessor else [])

        if len(preprocessor_variants) != 1 or preprocessor_opts:
            log('vwoptimize preprocessor = %s', preprocessor_opts, log_level=1)

        previously_done = already_done.get(str(preprocessor))

        if previously_done:
            log('Same as %s', previously_done)
            continue

        already_done[str(preprocessor)] = preprocessor_opts

        if format == 'vw':
            vw_filename = source
            # XXX preprocessor can make sense too
            assert not preprocessor, 'TODO'
        else:
            vw_filename = get_temp_filename('vw_filename')
            convert_any_to_vw(
                source=source,
                format=format,
                output_filename=vw_filename,
                columnspec=config.get('columnspec'),
                labels=config.get('labels'),
                weights=config.get('weights'),
                preprocessor=preprocessor_opts,
                ignoreheader=ignoreheader,
                workers=workers)

        folds, total = split_file(vw_filename, nfolds)
        for fname in folds:
            assert os.path.exists(fname), fname

        vw_args = [x for x in my_args if x not in PREPROCESSING_BINARY_OPTS]

        try:
            this_best_result, this_best_options = vw_optimize_over_cv(
                vw_filename,
                folds,
                vw_args,
                optimization_metric,
                config,
                workers=workers,
                other_metrics=other_metrics)
        finally:
            unlink(*folds)

        is_best = ''
        if this_best_result is not None and (best_result is None or this_best_result < best_result):
            best_result = this_best_result
            best_vw_options = this_best_options
            best_preprocessor_opts = preprocessor_opts
            is_best = '*'

        if preprocessor_opts:
            print 'Best options with %s: %s' % (preprocessor_opts or 'no preprocessing', this_best_options, )
        print 'Best %s with %r: %.4f%s' % (optimization_metric, preprocessor_opts or 'no preprocessing', abs(this_best_result or 0.0), is_best)
        # print 'Improvement over no l1=%.4f. Improvement over initial guess=%.4f' % (no_l1_result - best_result[0], initial_l1_result - best_result[0])

    # XXX don't show this if preprocessor is not enabled and not tuned
    print 'Best preprocessor options: %s' % (best_preprocessor_opts or '<none>', )
    print 'Best vw options: %s' % (best_vw_options, )
    print 'Best %s: %.4f' % (optimization_metric, abs(best_result))
    # print 'Improvement over no l1=%.4f. Improvement over initial guess=%.4f' % (no_l1_result - best_result[0], initial_l1_result - best_result[0])
    preprocessor = Preprocessor.from_options(best_preprocessor_opts)
    return best_vw_options, preprocessor


def format_item(counts, weight, hash, ignore_single=None):
    if ignore_single:
        top_items = [(v, k) for (k, v) in counts[hash].items() if v > 1]
    else:
        top_items = [(v, k) for (k, v) in counts[hash].items()]
    if not top_items:
        return
    top_items.sort(reverse=True)
    top_items = ', '.join('%s %s' % (k, v) for (v, k) in top_items)
    return '%g %s' % (weight, top_items)


def parseaudit(source):
    weights = {}
    counts = {}  # hash -> text -> count
    heap = []
    line = None

    while True:
        line = source.readline()
        if not line:
            break
        line = line.rstrip()
        if not line.startswith('\t'):
            continue

        for feature in set(line.strip().split()):
            text, hash, value, weight = feature.split(':')[:4]
            weight = weight.split('@')[0]
            weight = float(weight)
            value = float(value)
            assert value == 1, value
            if not weight:
                continue

            c = counts.setdefault(hash, {})
            c.setdefault(text, 0)
            c[text] += 1

            if hash in weights:
                assert weights.get(hash) == weight, (hash, text, weight, weights.get(hash))
                continue

            weights[hash] = weight
            heapq.heappush(heap, (-weight, hash))

    while heap:
        w, hash = heapq.heappop(heap)
        item = format_item(counts, -w, hash)
        if item:
            print item


def main_streaming(source, format, columnspec, vw_args, vw_options, preprocessor, labels, weights, ignoreheader):
    vw_cmd = 'vw %s' % vw_args

    if vw_options.input_regressor:
        vw_cmd += ' -i %s' % vw_options.input_regressor

    if vw_options.final_regressor:
        vw_cmd += ' -f %s' % vw_options.final_regressor

    if vw_options.predictions:
        vw_cmd += ' -p %s' % vw_options.predictions

    if vw_options.raw_predictions:
        vw_cmd += ' -r %s' % vw_options.raw_predictions

    if vw_options.audit:
        vw_cmd += ' -a'

    vw_cmd = vw_cmd.split()

    if format == 'vw':
        if source is None:
            popen = subprocess.Popen(vw_cmd, stdin=sys.stdin, preexec_fn=die_if_parent_dies)
        else:
            vw_cmd.extend(['-d', source])
            popen = subprocess.Popen(vw_cmd, preexec_fn=die_if_parent_dies)
    else:
        popen = subprocess.Popen(vw_cmd, stdin=subprocess.PIPE, preexec_fn=die_if_parent_dies)
        for row in open_anything(source, format, ignoreheader=ignoreheader):
            line = convert_row_to_vw(row, columnspec=columnspec, preprocessor=preprocessor, labels=labels, weights=weights)
            popen.stdin.write(line)
            # subprocess.Popen is unbuffered by default
        popen.stdin.close()

    popen.wait()


def main(to_cleanup):
    parser = PassThroughOptionParser()

    # cross-validation and parameter tuning options
    parser.add_option('--cv', action='store_true')
    parser.add_option('--workers', type=int)
    parser.add_option('--nfolds', type=int)
    parser.add_option('--metric', action='append')

    # class weight option
    parser.add_option('--weight', action='append', help='Class weights to use in CLASS:WEIGHT format', default=[])
    parser.add_option('--weight_train', action='append', help='Class weight to use (during training only), in CLASS:WEIGHT format', default=[])
    parser.add_option('--weight_metric', action='append', help='Class weight to use (for weighted metrics only), in CLASS:WEIGHT format', default=[])

    # vowpal wabbit arguments (those that we care about. everything else is passed through)
    parser.add_option('-r', '--raw_predictions')
    parser.add_option('-p', '--predictions')
    parser.add_option('-f', '--final_regressor')
    parser.add_option('-i', '--input_regressor')
    parser.add_option('-d', '--data')
    parser.add_option('-a', '--audit', action='store_true')
    parser.add_option('--readconfig')
    parser.add_option('--writeconfig')

    # preprocessing options:
    parser.add_option('--labels')
    for opt in Preprocessor.ALL_OPTIONS:
        parser.add_option('--%s' % opt, action='store_true')
    parser.add_option('--columnspec')

    # should be 'count'
    parser.add_option('--ignoreheader', action='store_true')

    # using preprocessor standalone:
    parser.add_option('--tovw')
    parser.add_option('--tovw_simple')
    parser.add_option('--format', help='File format, one of vw|tsv|csv. If not provided, will be guessed from file extension or from file contents')

    # using as perf
    parser.add_option('--report', action='store_true')
    parser.add_option('--toperrors')
    parser.add_option('--threshold', type=float)

    # logging and debugging and misc
    parser.add_option('--morelogs', action='count', default=0)
    parser.add_option('--lesslogs', action='count', default=0)
    parser.add_option('--keeptmp', action='store_true')
    parser.add_option('--savefeatures')
    parser.add_option('--parseaudit', action='store_true')

    options, args = parser.parse_args()

    globals()['LOG_LEVEL'] += options.lesslogs - options.morelogs
    globals()['KEEPTMP'] = options.keeptmp

    if options.parseaudit:
        parseaudit(sys.stdin)
        sys.exit(0)

    config = {}

    if options.readconfig:
        config = json.load(open(options.readconfig))
        log('vwoptimize config = %s', options.readconfig, log_level=1)

        if 'regressor' in config and not options.input_regressor and '-t' in args:
            options.input_regressor = os.path.normpath(os.path.join(os.path.dirname(options.readconfig), config['regressor']))
            if not os.path.exists(options.input_regressor):
                sys.exit('Cannot find %r referenced from %r' % (options.input_regressor, options.readconfig))

    if options.input_regressor:
        args = ['-i', options.input_regressor] + args

    used_stdin = False
    if options.data is None or options.data in STDIN_NAMES:
        used_stdin = True
        filename = None
    else:
        filename = options.data

    labels = _make_proper_list(options.labels)

    if labels and len(labels) <= 1:
        sys.exit('Expected comma-separated list of labels: --labels %r\n' % options.labels)

    if labels:
        config['labels'] = labels

    weight = parse_weight(options.weight, labels)
    weight_train = parse_weight(options.weight_train, labels) or weight
    weight_metric = parse_weight(options.weight_metric, labels) or weight

    if weight_train:
        config['weight_train'] = weight_train

    if weight_metric:
        config['weight_metric'] = weight_metric

    if options.columnspec:
        config['columnspec'] = _make_proper_list(options.columnspec)

    preprocessor_from_options = Preprocessor.from_options(options.__dict__)

    if preprocessor_from_options:
        if config.get('preprocessor'):
            log('Preprocessor specified in config (%s) and on command line (%s), going with the latter.', config['preprocessor'], preprocessor_from_options, log_level=2)
        config['preprocessor'] = str(preprocessor_from_options)
        preprocessor = preprocessor_from_options
    elif config.get('preprocessor'):
        preprocessor = Preprocessor.from_options(config['preprocessor'])
    else:
        preprocessor = None

    format = options.format

    if not format and filename:
        format = get_format_from_filename(filename)

    if format and format not in ('vw', 'csv', 'tsv'):
        sys.exit('--format must one of vw,csv,tsv, not %r' % format)

    format = format or config.get('format')

    if not format:
        format = 'vw'

    config['format'] = format

    if options.tovw_simple:
        assert not options.workers or options.workers == 1, options.workers
        assert 'balaced' not in str(options.weight_train), '"balanced" not supported here: %r' % options.weight_train
        _convert_any_to_vw(
            filename,
            format,
            options.tovw_simple,
            labels,
            weight_train,
            preprocessor,
            config.get('columnspec'),
            ignoreheader=options.ignoreheader)
        sys.exit(0)

    if options.threshold is not None:
        config['threshold'] = options.threshold

    vw_multiclass_opts = 'oaa|ect|csoaa|log_multi|recall_tree'

    n_classes_cmdline = re.findall('--(?:%s)\s+(\d+)' % vw_multiclass_opts, ' '.join(args))
    if n_classes_cmdline:
        config['n_classes'] = max(int(x) for x in n_classes_cmdline)

    need_tuning = 0

    for arg in args:
        if arg.endswith('?'):
            need_tuning = 1
            break

    can_do_streaming = True
    if config.get('n_classes') == 0 or options.metric or options.cv or options.toperrors or need_tuning or options.savefeatures or options.tovw or options.tovw_simple:
        can_do_streaming = False

    if can_do_streaming:
        main_streaming(
            source=filename,
            format=format,
            columnspec=config.get('columnspec'),
            vw_args=' '.join(args),
            vw_options=options,
            preprocessor=preprocessor,
            labels=labels,
            weights=config.get('weights'),
            ignoreheader=options.ignoreheader)
        sys.exit(0)

    labels_counts = None
    y_true = None

    need_to_read_labels = options.metric or 'labels' not in config or options.toperrors
    #  ^ This is not correct condition. we only need labels for weight_metric
    #  ^ This also won't be right for "vw_average_loss" label, which is parsed from VW output, not calculated

    if filename is None:
        filename = None
        from StringIO import StringIO
        source = StringIO(sys.stdin.read())
    else:
        source = None

    if need_to_read_labels:
        if format == 'vw':
            label_index = 0
        else:
            if 'columnspec' not in config:
                config['columnspec'] = ['y', 'text']
            try:
                label_index = config['columnspec'].index('y')
            except ValueError:
                label_index = None
        if label_index is not None:
            labels_counts, y_true, config['labels'], config['n_classes'] = read_labels(filename, source, format, config.get('n_classes'), label_index, options.ignoreheader)
            if config['n_classes'] is None:
                config['min_label'] = min(y_true)
                config['max_label'] = max(y_true)
                config['threshold'] = (config['min_label'] + config['max_label']) / 2.0
                log('Setting threshold from data = %g', config['threshold'])

    if config.get('n_classes'):
        args = re.sub('(--(?:%s)\s+)(0)' % vw_multiclass_opts, '\\g<1>' + str(config['n_classes']), ' '.join(args)).split()

    if config.get('weight_train') == ['balanced']:
        config['weight_train'] = balanced_weights = get_balanced_weights(labels_counts)

    if config.get('weight_metric') == ['balanced']:
        config['weight_metric'] = balanced_weights or get_balanced_weights(labels_counts)

    if config.get('weight_metric'):
        if config.get('labels'):
            config['weight_metric'] = dict((config['labels'].index(key) + 1, weight) for (key, weight) in config['weight_metric'].items())
        else:
            config['weight_metric'] = dict((float(key), weight) for (key, weight) in config['weight_metric'].items())

    if options.tovw:
        assert format != 'vw', 'Input should be csv or tsv'  # XXX
        convert_any_to_vw(
            source=source or filename,
            format=format,
            output_filename=options.tovw,
            preprocessor=config.get('preprocessor'),
            columnspec=config.get('columnspec'),
            labels=config.get('labels'),
            weights=config.get('weight_train'),
            ignoreheader=options.ignoreheader,
            workers=options.workers)
        sys.exit(0)

    options.metric = _make_proper_list(options.metric) or []

    if options.report:
        if not options.metric:
            options.metric = ['mse']
        if options.predictions in STDIN_NAMES:
            if used_stdin:
                sys.exit('Can only use stdin in one argument')
            predictions = sys.stdin
        elif options.predictions:
            predictions = options.predictions
        else:
            sys.exit('Must provide -p')

        y_pred = np.array(_load_first_float_from_each_string(predictions))
        assert y_true is not None

        for metric in options.metric:
            print '%s: %g' % (metric, calculate_score(metric, y_true, y_pred, config))

        sys.exit(0)

    need_tuning = 0

    index = 0
    while index < len(args):
        arg = args[index]
        if arg.startswith('-'):
            next_arg = args[index + 1] if index + 1 < len(args) else ''
            if arg.endswith('?'):
                need_tuning += 1
                args[index] = get_tuning_config(arg)
            elif next_arg.endswith('?'):
                need_tuning += 1
                args[index:index + 2] = [get_tuning_config(arg + ' ' + next_arg)]
        index += 1

    if need_tuning:
        assert not options.audit, '-a incompatible with parameter tuning'
        config['vw_train_options'], preprocessor = main_tune(
            metric=options.metric,
            config=config,
            source=source or filename,
            format=format,
            args=args,
            preprocessor_base=preprocessor,
            nfolds=options.nfolds,
            ignoreheader=options.ignoreheader,
            workers=options.workers)
        config['preprocessor'] = str(preprocessor)
    else:
        config['vw_train_options'] = ' '.join(args)

    if format == 'vw':
        assert not preprocessor or not preprocessor.to_options(), preprocessor
        vw_source = source or filename
    else:
        vw_filename = get_temp_filename('vw')
        to_cleanup.append(vw_filename)

        convert_any_to_vw(
            source=source or filename,
            format=format,
            output_filename=vw_filename,
            preprocessor=config.get('preprocessor'),
            columnspec=config.get('columnspec'),
            labels=config.get('labels'),
            weights=config.get('weight_train'),
            ignoreheader=options.ignoreheader,
            workers=options.workers)

        vw_source = vw_filename

    if options.cv:
        # XXX we could skip --cv here if we make main_tune() keep final predictions and raw_predictions for us
        try:
            folds, total_lines = split_file(vw_source, nfolds=options.nfolds)

            assert len(folds) >= 2, folds

            cv_predictions = options.predictions
            if not cv_predictions:
                cv_predictions = get_temp_filename('cvpred')
                to_cleanup.append(cv_predictions)

            cv_pred = vw_cross_validation(
                folds,
                config['vw_train_options'],
                workers=options.workers,
                p_fname=cv_predictions,
                r_fname=options.raw_predictions,
                audit=options.audit)

            assert y_true is not None

            for metric in options.metric:
                value = calculate_score(metric, y_true, cv_pred, config)
                print 'cv %s: %g' % (metric, value)

        finally:
            unlink(*folds)

    if options.cv:
        # all of these are related to CV if --cv is enabled
        options.predictions = None
        options.raw_predictions = None
        options.audit = None

    final_regressor = options.final_regressor

    config_tmp_filename = None
    if options.writeconfig:
        log('write config = %s', options.writeconfig, log_level=1)
        assert options.writeconfig != options.final_regressor, options.writeconfig

        if final_regressor:
            config['regressor'] = os.path.relpath(final_regressor, os.path.dirname(options.writeconfig))

        config_tmp_filename = options.writeconfig + '.tmp'
        to_cleanup.append(config_tmp_filename)
        output_fobj = open(config_tmp_filename, 'w')
        json.dump(config, output_fobj, sort_keys=True, indent=4)
        output_fobj.write('\n')
        output_fobj.close()

    if not final_regressor and options.savefeatures:
        final_regressor = get_temp_filename('final_regressor')

    final_regressor_tmp = None
    if final_regressor:
        final_regressor_tmp = final_regressor + '.tmp'
        to_cleanup.append(final_regressor_tmp)

    if final_regressor_tmp or not (options.cv or need_tuning):
        vw_cmd = 'vw %s' % config['vw_train_options']

        if isinstance(vw_source, basestring):
            assert os.path.exists(vw_source), vw_source
            vw_cmd += ' -d %s' % (vw_source, )
            vw_stdin = None
        else:
            vw_stdin = vw_source.getvalue()

        if final_regressor_tmp:
            vw_cmd += ' -f %s' % final_regressor_tmp

        predictions_fname = options.predictions

        if options.metric or options.toperrors:
            if not predictions_fname:
                predictions_fname = get_temp_filename('pred')
                to_cleanup.append(predictions_fname)

        if predictions_fname:
            vw_cmd += ' -p %s' % predictions_fname

        if options.raw_predictions:
            vw_cmd += ' -r %s' % options.raw_predictions

        if options.audit:
            vw_cmd += ' -a'

        system(vw_cmd, log_level=0, stdin=vw_stdin)

        y_pred = None

        if options.metric and predictions_fname:
            assert y_true is not None
            y_pred = np.array(_load_first_float_from_each_string(predictions_fname))

            for metric in options.metric:
                print '%s: %g' % (metric, calculate_score(metric, y_true, y_pred, config))

        if options.toperrors:
            assert y_true is not None

            if y_pred is None:
                y_pred = np.array(_load_first_float_from_each_string(predictions_fname))

            errors = []

            for yp, yt, example in zip(y_pred, y_true, open_anything(source or filename, format, ignoreheader=options.ignoreheader)):
                # add hash of the example as a second item so that we get a mix of false positives and false negatives for a given error level
                errors.append((abs(yp - yt), hash(repr(example)), yp, example))

            errors.sort(reverse=True)

            if '.' in options.toperrors:
                min_error = float(options.toperrors)
                errors = [x for x in errors if x[0] >= min_error]
            else:
                count = int(options.toperrors)
                errors = errors[:count]

            output = csv.writer(sys.stdout)

            for err, _hash, yp, example in errors:
                row = [yp]
                if isinstance(example, list):
                    row.extend(example)
                else:
                    row.append(str(example))
                output.writerow(row)

    if final_regressor_tmp:
        os.rename(final_regressor_tmp, final_regressor)

    if config_tmp_filename:
        os.rename(config_tmp_filename, options.writeconfig)

    if options.savefeatures:
        vw_cmd = 'vw'

        if isinstance(vw_source, basestring):
            assert os.path.exists(vw_source), vw_source
            vw_cmd += ' -d %s' % (vw_source, )
            vw_stdin = None
        else:
            vw_stdin = vw_source.getvalue()

        regressor = final_regressor or options.input_regressor

        assert regressor
        vw_cmd += ' -i %s -t -a' % regressor
        to_cleanup.append(options.savefeatures + '.tmp')
        system(vw_cmd + ' | %s %s --parseaudit > %s.tmp' % (sys.executable, __file__, options.savefeatures))
        os.rename(options.savefeatures + '.tmp', options.savefeatures)

    unlink(*to_cleanup)


if __name__ == '__main__':
    to_cleanup = []
    try:
        main(to_cleanup)
    finally:
        unlink(*to_cleanup)
