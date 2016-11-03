#!/usr/bin/env python
"""Wrapper for Vowpal Wabbit that does cross-validation and hyper-parameter tuning"""
__version__ = '0.6.0dev'
__gitdescribe__ = 'GIT'
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
from itertools import izip_longest
from collections import deque
from pipes import quote
import numpy as np


csv.field_size_limit(10000000)
LOG_LEVEL = 1
TMPID = str(os.getpid())
KEEPTMP = False
STDIN_NAMES = ('/dev/stdin', '-')
STDOUT_NAMES = ('/dev/stdout', 'stdout')
VW_CMD = 'vw'
VOWPAL_WABBIT_ERRORS = "error|won't work right|errno|can't open|vw::vw_exception"
DEFAULT_COLUMNSPEC = 'y,text,*'
METRIC_FORMAT = 'mean'

AWK_TRAINSET = "awk '(NR - $fold) % KFOLDS != 0' VW |"
AWK_TESTSET = "awk '(NR - $fold) % KFOLDS == 0' VW |"
PERL_TRAINSET = "perl -nE 'if ((++$NR - $fold) % KFOLDS != 0) { print $_ }' VW |"
PERL_TESTSET = "perl -nE 'if ((++$NR - $fold) % KFOLDS == 0) { print $_ }' VW |"

if 'darwin' in sys.platform:
    # awk is slow on Mac OS X
    FOLDSCRIPT = 'perl'
else:
    FOLDSCRIPT = 'awk'


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


def _unlink_one(filename):
    if not os.path.exists(filename):
        return
    try:
        os.unlink(filename)
    except Exception:
        sys.stderr.write('Failed to unlink %r\n' % filename)
        traceback.print_exc()


def unlink(*filenames):
    if KEEPTMP:
        return
    for filename in filenames:
        if not filename:
            continue
        if not isinstance(filename, basestring):
            sys.exit('unlink() expects filenames as str or None, not %r\n' % (filename, ))
        _unlink_one(filename)
        # vowpal wabbit might create this and then not clean up
        _unlink_one(filename + '.writing')


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
    fname = '%s/%s.%s.%s' % (TMP_PREFIX, TMPID, counter[0], suffix)
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


def log_always(*args, **kwargs):
    kwargs['log_level'] = LOG_LEVEL
    return log(*args, **kwargs)


def vw_failed(msg=''):
    if msg:
        sys.exit('%s failed: %s' % (VW_CMD, msg))
    else:
        sys.exit('%s failed' % (VW_CMD, ))


def flush_and_close(fileobj):
    fileobj.flush()
    try:
        os.fsync(fileobj.fileno())
    except OSError:
        pass
    fileobj.close()


def write_file(filename, data):
    if isinstance(data, list):
        data = ''.join(data)
    else:
        assert isinstance(data, str), type(data)
    if filename in STDOUT_NAMES:
        sys.stdout.write(data)
    else:
        fobj = open(filename, 'w')
        fobj.write(data)
        flush_and_close(fobj)


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
            sys.exit('Bad line: %r' % (orig_line, ))
        prefix_items = items[0].strip().split()
        if not prefix_items or not prefix_items or prefix_items[0].startswith("'"):
            yield (None, orig_line)
        else:
            yield (prefix_items[0], orig_line)


def open_anything(source, format, ignoreheader, force_unbuffered=False):
    source = open_regular_or_compressed(source)

    if force_unbuffered:
        # simply disabling buffering is not enough, see this for details: http://stackoverflow.com/a/6556862
        source = iter(source.readline, '')

    if format == 'vw':
        return _read_lines_vw(source)

    if format == 'tsv':
        reader = csv.reader(source, csv.excel_tab)
        if ignoreheader:
            reader.next()
    elif format == 'csv':
        reader = csv.reader(source, csv.excel)
        if ignoreheader:
            reader.next()
    else:
        raise ValueError('format not supported: %s' % format)

    return reader


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


def system(cmd, log_level=1):
    if isinstance(cmd, deque):
        for item in cmd:
            system(item, log_level=log_level)
        return

    sys.stdout.flush()
    start = time.time()

    popen = Popen(cmd, shell=True, log_level=log_level)

    if popen.stdout is not None or popen.stderr is not None:
        out, err = popen.communicate()
    else:
        out, err = '', ''

    retcode = popen.wait()

    if retcode:
        log('%s [%.1fs] %s', '-' if retcode == 0 else '!', time.time() - start, cmd, log_level=log_level - 1)

    if retcode:
        sys.exit(1)

    return (out or '') + (err or '')


def split_file(source, nfolds=None, ignoreheader=False, log_level=0, minfoldsize=10000):
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
    foldsize = max(foldsize, minfoldsize)
    nfolds = int(math.ceil(total_lines / float(foldsize)))

    folds = []

    current_fold = -1
    count = foldsize
    current_fileobj = None
    total_count = 0
    for line in source:
        if count >= foldsize:
            if current_fileobj is not None:
                flush_and_close(current_fileobj)
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
        flush_and_close(current_fileobj)

    if total_count != total_lines:
        sys.exit('internal error: total_count=%r total_lines=%r source=%r' % (total_count, total_lines, source))

    return folds, total_lines


def _workers(workers):
    if workers is not None and workers <= 1:
        return 1
    if workers is None or workers <= 0:
        import multiprocessing
        return 1 + multiprocessing.cpu_count()
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


def Popen(params, **kwargs):

    if isinstance(params, dict):
        params = params.copy()
        args = params.pop('args')
        params.update(kwargs)
    else:
        args = params
        params = kwargs

    log_level = params.pop('log_level', 0)
    params.setdefault('preexec_fn', die_if_parent_dies)

    if isinstance(args, list):
        command = ' '.join(args)
    else:
        command = args

    name = params.pop('name', None)
    if name:
        log('+ [%s] %s', name, command, log_level=log_level)
    else:
        log('+ %s', command, log_level=log_level)

    popen = subprocess.Popen(args, **params)
    return popen


def run_subprocesses(cmds, workers=None, log_level=None):
    for item in cmds:
        if isinstance(item, deque):
            for subitem in item:
                assert isinstance(subitem, dict), subitem
        else:
            assert isinstance(item, dict), item

    workers = _workers(workers)
    cmds_queue = deque(cmds)
    queue = deque()
    success = False
    outputs = {}

    try:
        while queue or cmds_queue:
            if cmds_queue and len(queue) <= workers:
                cmd = cmds_queue.popleft()

                if isinstance(cmd, deque):
                    this_cmd = cmd.popleft()
                    followup = cmd
                else:
                    this_cmd = cmd
                    followup = None

                popen = Popen(this_cmd, shell=True)
                popen._cmd = this_cmd
                popen._name = this_cmd.get('name', '')
                popen._followup = followup
                queue.append(popen)
            else:
                popen = queue.popleft()

                if popen.stdout is not None or popen.stderr is not None:
                    out, err = popen.communicate()
                    out = (out or '') + (err or '')
                    outputs.setdefault(popen._name, []).append(out)

                retcode = popen.wait()

                if retcode:
                    log('failed: %s', popen._cmd.get('args', popen._cmd), log_level=3)
                    return None, None
                else:
                    log('%s %s', '-' if retcode == 0 else '!', popen._cmd, log_level=log_level)

                if popen._followup:
                    cmds_queue.append(popen._followup)

        success = True

    finally:
        if not success:
            kill(*queue, verbose=True)

    return success, outputs


def _as_dict(lst, name):
    if isinstance(lst, list):
        lst = ' '.join(lst).strip()
        lst = re.sub('\s+', ' ', lst)
    return {'args': lst, 'shell': True, 'name': name}


def get_vw_command(
        to_cleanup,
        source,
        vw_args='',
        initial_regressor=None,
        final_regressor=None,
        predictions=None,
        raw_predictions=None,
        audit=False,
        feature_mask_retrain=None,
        readable_model=None,
        only_test=False,
        name=''):
    data_filename = ''
    data_pipeline = ''

    if source is None:
        pass
    elif isinstance(source, basestring):
        if '|' in source:
            data_pipeline = source
            if not data_pipeline.strip().endswith('|'):
                data_pipeline += ' |'
        else:
            assert os.path.exists(source), source
            data_filename = '-d %s' % source
    elif isinstance(source, list):
        assert source and os.path.exists(source[0]), source
        data_pipeline = 'cat %s |' % ' '.join(quote(x) for x in source)
    else:
        raise TypeError('Expected string or list, not %r' % (source, ))

    if feature_mask_retrain:
        if final_regressor:
            intermediate_model_filename = final_regressor + '.feature_mask'
        else:
            intermediate_model_filename = get_temp_filename('feature_mask')
        to_cleanup.append(intermediate_model_filename)
    else:
        intermediate_model_filename = final_regressor

    final_options = []

    if audit:
        final_options += ['-a']

    if readable_model:
        final_options += ['--readable_model', readable_model]

    vw_args = vw_args.split()

    if '--cache_file' in vw_args:
        sys.exit('Dont provide --cache_file, one will be added automatically.')

    if '-c' in vw_args or '--cache_file' in vw_args:
        remove_option(vw_args, '-c', 0)
        remove_option(vw_args, '--cache_file', 1)
        if final_regressor:
            cache_file = final_regressor + '.cache'
        else:
            cache_file = get_temp_filename('cache')
        vw_args.extend(['--cache_file', cache_file])
        to_cleanup.append(cache_file)

    training_command = [
        data_pipeline,
        VW_CMD,
        data_filename,
        '-i %s' % initial_regressor if initial_regressor else '',
        '-f %s' % intermediate_model_filename if intermediate_model_filename else '',

        # In case of --feature_mask_retrain, we must decide whether to output predictions from first or second vw
        # command. Since predictions of the second one will probably overfit, using the first one.
        '-p %s' % predictions if predictions else '',
        '-r %s' % raw_predictions if raw_predictions else '',

        '-t' if only_test else '',
    ] + vw_args

    if only_test:
        return _as_dict(training_command + final_options, name=name)

    if not feature_mask_retrain:
        return deque([_as_dict(training_command + final_options, name=name)])

    if isinstance(feature_mask_retrain, bool):
        feature_mask_retrain = ''
    else:
        if not isinstance(feature_mask_retrain, basestring):
            raise TypeError('feature_mask_retrain must be bool or str, not %r' % (feature_mask_retrain, ))

    assert data_pipeline or data_filename

    assert '--feature_mask' not in vw_args, vw_args

    feature_mask_retrain = feature_mask_retrain.split()

    if '-c' in feature_mask_retrain or '--cache_file' in feature_mask_retrain:
        remove_option(feature_mask_retrain, '-c', 0)
        remove_option(feature_mask_retrain, '--cache_file', 0)
        cache_file = final_regressor + '.cache'
        feature_mask_retrain.extend(['--cache_file', cache_file])
        to_cleanup.append(cache_file)

    return deque([
        _as_dict(training_command, name=name + "1"),
        _as_dict([
            data_pipeline,
            VW_CMD,
            data_filename,
            '--quiet' if '--quiet' in vw_args else '',
            '-f %s' % final_regressor if final_regressor else '',
            '--feature_mask %s' % intermediate_model_filename,
            '-i %s' % intermediate_model_filename] + feature_mask_retrain + final_options, name=name + "2"),
    ])


def vw_cross_validation(
        vw_filename,
        kfold,
        vw_args,
        workers=None,
        with_predictions=False,
        with_raw_predictions=False,
        feature_mask_retrain=False,
        calc_num_features=False,
        capture_output=False):

    if hasattr(capture_output, '__contains__') and '' in capture_output:
        capture_output = True

    workers = _workers(workers)
    commands = []
    p_filenames = []
    r_filenames = []
    readable_models = []
    to_cleanup = []

    vw_args = vw_args.replace('--quiet', '')

    # Split into folds is done like this (example for 3 folds)
    # Example -> fold:
    # 1 -> 1
    # 2 -> 2
    # 3 -> 3
    # 4 -> 1
    # and so on

    if kfold is None:
        trainset = vw_filename
        testset = None
        kfold = 1
    else:
        assert kfold > 1, kfold
        if FOLDSCRIPT == 'awk':
            trainset = AWK_TRAINSET
            testset = AWK_TESTSET
        elif FOLDSCRIPT == 'perl':
            trainset = PERL_TRAINSET
            testset = PERL_TESTSET
        else:
            raise AssertionError('foldscript=%r not understood' % FOLDSCRIPT)

        trainset = trainset.replace('KFOLDS', str(kfold)).replace('VW', vw_filename)
        testset = testset.replace('KFOLDS', str(kfold)).replace('VW', vw_filename)

    model_filename = get_temp_filename('model') + '.$fold'

    if with_predictions:
        p_filename = '%s.predictions' % model_filename
    else:
        p_filename = None

    if with_raw_predictions:
        r_filename = '%s.raw' % model_filename
    else:
        r_filename = None

    if calc_num_features:
        readable_model = model_filename + '.readable'
    else:
        readable_model = None

    cleanup_tmpl = []

    base_training_command = get_vw_command(
        cleanup_tmpl,
        trainset,
        vw_args=vw_args,
        final_regressor=model_filename,
        predictions=None if testset else p_filename,
        raw_predictions=None if testset else r_filename,
        feature_mask_retrain=feature_mask_retrain,
        readable_model=readable_model,
        name='train' if testset else 'test')

    for item in base_training_command:
        if capture_output is True or item['name'] in capture_output:
            item['stderr'] = subprocess.PIPE
        else:
            item['args'] += ' --quiet'

    if testset:
        testing_command = get_vw_command(
            cleanup_tmpl,
            testset,
            initial_regressor=model_filename,
            predictions=p_filename,
            raw_predictions=r_filename,
            only_test=True,
            name='test')

        # XXX this also can be moved inside
        if capture_output is True or 'test' in capture_output:
            testing_command['stderr'] = subprocess.PIPE
        else:
            testing_command['args'] += ' --quiet'

        base_training_command.append(testing_command)

    for item in base_training_command:
        log("+ %s", item['args'])

    commands = []

    for this_fold in xrange(1, kfold + 1):
        this_fold = str(this_fold)
        training_command = deque([x.copy() for x in base_training_command])
        for cmd in training_command:
            cmd['args'] = cmd['args'].replace('$fold', this_fold)
        commands.append(training_command)

        for filename in [model_filename, p_filename, r_filename, readable_model] + cleanup_tmpl:
            if not filename:
                continue
            filename = filename.replace('$fold', this_fold)
            assert not os.path.exists(filename), filename
            to_cleanup.append(filename)

        if p_filename:
            p_filenames.append(p_filename.replace('$fold', this_fold))

        if r_filename:
            r_filenames.append(r_filename.replace('$fold', this_fold))

        if readable_model:
            readable_models.append(readable_model.replace('$fold', this_fold))

    try:
        success, outputs = run_subprocesses(commands, workers=workers, log_level=-1)

        if not success:
            vw_failed()

        for name in to_cleanup:
            if not os.path.exists(name):
                vw_failed('missing %r' % (name, ))

        predictions = []
        for items in izip_longest(*[open(x) for x in p_filenames]):
            predictions.extend([x for x in items if x is not None])

        raw_predictions = []
        for items in izip_longest(*[open(x) for x in r_filenames]):
            raw_predictions.extend([x for x in items if x is not None])

        num_features = [get_num_features(name) for name in readable_models]

        outputs = dict((key, [parse_vw_output(out) for out in value]) for (key, value) in outputs.items())

        return predictions, raw_predictions, num_features, outputs

    finally:
        unlink(*to_cleanup)


def get_num_features(filename):
    counting = False
    count = 0
    for line in open(filename):
        if counting:
            count += 1
        else:
            if line.strip() == ':0':
                counting = True
    return count


def parse_vw_output(output):
    result = {}
    for line in output.split('\n'):
        if line.count(' = ') == 1:
            key, value = line.split(' = ')
            key = key.replace(' ', '_').replace("'", '').lower()
            result[key] = value
        else:
            if re.search(VOWPAL_WABBIT_ERRORS, line.lower()):
                sys.exit('vw failed: %s' % line.strip())
    return result


def _load_first_float_from_each_string(file, size=None, with_text=False):
    filename = file
    if isinstance(file, list):
        filename = file
    elif hasattr(file, 'read'):
        pass
    elif isinstance(file, basestring):
        if file in STDOUT_NAMES:
            sys.exit('Will not read %s' % file)
        file = open(file)
    else:
        raise AssertionError(limited_repr(file))

    result = []
    result_text = []

    for line in file:
        try:
            text = line.strip()
            if with_text:
                result_text.append(text)
            result.append(float(text.split()[0]))
        except:
            sys.stderr.write('Error while parsing %r\nin %r\n' % (line, limited_repr(filename)))
            raise

    if size is not None:
        if len(result) < size:
            sys.exit('Too few items in %s: found %r, expecting %r' % (limited_repr(filename), len(result), size))

        if len(result) > size:
            mult = int(len(result) / size)
            if size * mult == len(result):
                # if --passes N option was used, then the number of predictions will be N times higher
                return np.array(result[-size:])

            sys.exit('Too many items in %s: found %r, expecting multiply of %r' % (limited_repr(filename), len(result), size))

    if with_text:
        return np.array(result), result_text
    else:
        return np.array(result)


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


def vw_optimize_over_cv(vw_filename, kfold, args, metric, config,
                        workers=None, other_metrics=[],
                        feature_mask_retrain=False, show_num_features=False):
    # we only depend on scipy if parameter tuning is enabled
    import scipy.optimize

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

    extra_args = ['']
    cache = {}
    best_result = [None, None]

    calculated_metrics = [x for x in [metric] + other_metrics if not x.startswith('vw')]
    vw_metrics = [x for x in [metric] + other_metrics if x.startswith('vw')]

    if calculated_metrics:
        y_true = _load_first_float_from_each_string(vw_filename)
    else:
        y_true = None

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

        log('Trying %s %s...', VW_CMD, args)

        try:
            y_pred_txt, raw_pred_txt, num_features, outputs = vw_cross_validation(
                vw_filename,
                kfold,
                args,
                workers=workers,
                with_predictions=bool(calculated_metrics),
                feature_mask_retrain=feature_mask_retrain,
                calc_num_features=show_num_features,
                capture_output=set([_get_stage(m) for m in vw_metrics]))
        except KeyboardInterrupt:
            raise
        except BaseException, ex:
            if type(ex) is not SystemExit:
                traceback.print_exc()
            log('Result %s %s... error: %s', VW_CMD, args, ex, log_level=1)
            cache[args] = 0.0
            return 0.0

        if y_true is not None:
            if calculated_metrics and len(y_true) != len(y_pred_txt):
                sys.exit('Internal error: expected %r predictions, got %r' % (len(y_true), len(y_pred_txt)))

            if raw_pred_txt and len(y_true) != len(raw_pred_txt):
                sys.exit('Internal error: expected %r raw predictions, got %r' % (len(y_true), len(raw_pred_txt)))

        y_pred = _load_first_float_from_each_string(y_pred_txt)
        result = calculate_or_extract_score(metric, y_true, y_pred, config, outputs)

        if isinstance(result, basestring):
            sys.exit('Cannot calculate %r: %s' % (metric, result))

        if isinstance(result, list):
            result = np.mean(result)

        if not isinstance(result, (int, long, float)):
            sys.exit('Bad metric for tuning: %s (value=%r)' % (metric, result))

        if not is_loss(metric):
            result = -result

        is_best = ' '
        if best_result[0] is None or result < best_result[0]:
            is_best = '*' if best_result[0] is not None else ' '
            best_result[0] = result
            best_result[1] = args

        other_scores = [(m, _frmt_score_short(calculate_or_extract_score(m, y_true, y_pred, config, outputs))) for m in other_metrics]
        other_results = ' '.join(['%s=%s' % x for x in other_scores])

        if num_features:
            other_results += ' num_features=%s' % _frmt_score(num_features)

        if other_results:
            other_results = '  ' + other_results

        other_results = (is_best + other_results).rstrip()

        log('Result %s %s... %s=%s%s', VW_CMD, args, metric, _frmt_score_short(result), other_results, log_level=1 + bool(is_best))

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
        run([None] * len(tunable_params))
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
    ALL_OPTIONS = 'htmlunescape lowercase strip_punct stem split_chars'.split()
    ALL_OPTIONS_DASHDASH = ['--%s' % x for x in ALL_OPTIONS]

    @classmethod
    def parse_options(cls, string):
        parser = PassThroughOptionParser()
        for opt in cls.ALL_OPTIONS_DASHDASH:
            parser.add_option(opt, action='store_true')
        options, args = parser.parse_args(string.split())
        return options.__dict__

    @classmethod
    def from_options(cls, options):
        if not options:
            return None

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

    def __init__(self, htmlunescape=False, lowercase=False, strip_punct=False, stem=False, split_chars=False, normalize_space=True, **ignored):
        self.normalize_space = normalize_space
        self.htmlunescape = htmlunescape
        self.lowercase = lowercase
        self.strip_punct = strip_punct
        self.stem = stem
        self.split_chars = split_chars
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

            if self.split_chars:
                words = [' '.join(w) for w in words]
                text = u' __ '.join(words)
            else:
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


def read_y_true(filename, format, columnspec, ignoreheader, named_labels):
    log('Reading labels from %s', filename or 'stdin')
    rows_source = open_anything(filename, format, ignoreheader=ignoreheader)

    if format == 'vw':
        label_index = 0
    else:
        label_index = columnspec.index('y')

    y_true = []

    for row in rows_source:
        label = row[label_index]
        if named_labels is not None and label not in named_labels:
            sys.exit('Unexpected label in %s: %r (allowed: %s)' % (filename, label, named_labels))
        y_true.append(label)

    if not named_labels:
        y_true = _load_first_float_from_each_string(y_true)

    return y_true


def _make_proper_list(s, type=None):
    if isinstance(s, basestring):
        result = s.split(',')
        if type is not None:
            result = [type(x) for x in result]
        return result

    if not s:
        return s

    result = []
    if isinstance(s, list):
        for x in s:
            result.extend(_make_proper_list(x, type))
    else:
        raise TypeError('Expected list of string: %r' % (s, ))
    return result


def proper_label(s):
    if '|' in s or ' ' in s or ':' in s:
        sys.exit('Not a proper label: %r' % s)
    return s


def parse_weight(config, named_labels=None):
    """
    >>> parse_weight('A:B:2', ['A:B', 'another_label'])
    {'A:B': 2.0}

    >>> parse_weight('A:B:2')
    Traceback (most recent call last):
     ...
    SystemExit: Weight must be specified as CLASS(float):WEIGHT, 'A:B' not recognized
    """
    if not config:
        return None

    if named_labels is not None and not isinstance(named_labels, list):
        raise TypeError('must be list, not %r' % type(named_labels))

    config = _make_proper_list(config)

    if not config:
        return None

    result = {}

    for item in config:
        if ':' not in item:
            sys.exit('Weight must be specified as CLASS:WEIGHT, cannot parse %r' % item)
        label, weight = item.rsplit(':', 1)

        if named_labels is None:
            try:
                label = float(label)
            except Exception:
                sys.exit('Weight must be specified as CLASS(float):WEIGHT, %r not recognized' % (label, ))
        else:
            if label not in named_labels:
                sys.exit('Label %r is not recognized. Expected: %r' % (label, named_labels))

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
        result += np.multiply(np.ones(N) * weight, y_true == klass)
        updated += y_true == klass

    result += (updated == 0)

    return result


def convert_row_to_vw(row, columnspec, preprocessor=None, weights=None, named_labels=None):
    assert isinstance(columnspec, list), columnspec

    if columnspec[-1] == '*':
        del columnspec[-1]
        while len(columnspec) < len(row):
            columnspec.append(columnspec[-1])

    if len(row) != len(columnspec):
        sys.exit('Expected %r columns (%r), got %r (%r)' % (len(columnspec), columnspec, len(row), row))

    y = ''
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

    if named_labels is None:
        if y:
            weight_key = float(y)
        else:
            weight_key = y
    else:
        weight_key = y
        if y not in named_labels:
            sys.exit('Label not recognized: %r (allowed: %r)' % (y, named_labels))

    if weights is not None:
        weight = weights.get(weight_key, 1)
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
        text = re.sub('\s+', ' ', text)
    text = text.replace(':', ' ').replace('|', ' ')
    text = text.strip()
    text = '%s %s%s| %s\n' % (y, weight, info, text)
    return text


def _convert_any_to_vw(source, format, output, weights, preprocessor, columnspec, named_labels, ignoreheader):
    assert format != 'vw'
    assert isinstance(columnspec, list)

    if named_labels is not None:
        assert not isinstance(named_labels, basestring)
        named_labels = set(named_labels)

    rows_source = open_anything(source, format, ignoreheader=ignoreheader)
    output = open(output, 'wb')

    for row in rows_source:
        vw_line = convert_row_to_vw(row, columnspec, preprocessor=preprocessor, weights=weights, named_labels=named_labels)
        output.write(vw_line)

    flush_and_close(output)


def convert_any_to_vw(source, format, output_filename, columnspec, named_labels, weights, preprocessor, ignoreheader, workers):
    assert format != 'vw'
    preprocessor = preprocessor or ''

    assert isinstance(preprocessor, basestring), preprocessor

    start = time.time()

    if source is None:
        from cStringIO import StringIO
        source = StringIO(sys.stdin.read())

    workers = _workers(workers)
    # XXX do os.stat on the source and decide on number of workers based on file size (e.g. less than 50k per worker does not make much sense)
    batches, total_lines = split_file(source, nfolds=workers, ignoreheader=ignoreheader, log_level=-1)

    batches_out = [x + '.out' for x in batches]

    try:
        commands = []

        common_cmd = [quote(sys.executable), quote(__file__), '--format', format]

        if named_labels:
            common_cmd += ['--named_labels', ','.join(named_labels)]

        if weights:
            weights = ['%s:%s' % (x, weights[x]) for x in weights if weights[x] != 1]
            weights = ','.join(weights)
            common_cmd += ['--weight', quote(weights)]

        if columnspec:
            common_cmd += ['--columnspec', quote(','.join(str(x) for x in columnspec))]

        common_cmd.append(preprocessor)

        for batch in batches:
            cmd = common_cmd + ['--tovw_simple', batch + '.out', '-d', batch]
            commands.append({'args': ' '.join(cmd)})

        success, outputs = run_subprocesses(commands, workers=workers, log_level=-1)
        if not success:
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
    'rmse': 'root_mean_squared_error',
    'auc': 'roc_auc_score',
    'brier': 'brier_score_loss',
    'acc': 'accuracy_score',
    'precision': 'precision_score',
    'recall': 'recall_score',
    'f1': 'f1_score',
    'cm': 'confusion_matrix',
    'hinge': 'hinge_loss',
}

metrics_param = {
    'mean_squared_error': 'y_score',
    'root_mean_squared_error': 'y_score',
    'hinge_loss': 'y_score',
    'roc_auc_score': 'y_score',
    'brier_score_loss': 'y_prob',
    'log_loss': 'y_prob',
    'accuracy_score': 'y_pred',
    'precision_score': 'y_pred',
    'recall_score': 'y_pred',
    'f1_score': 'y_pred',
    'confusion_matrix': 'y_pred',
    'matthews_corrcoef': 'y_pred',
}


def root_mean_squared_error(*args, **kwargs):
    import sklearn.metrics
    return math.sqrt(sklearn.metrics.mean_squared_error(*args, **kwargs))


def is_loss(metric_name):
    if metric_name.endswith('_w'):
        metric_name = metric_name[:-2]
    metric_name = metrics_shortcuts.get(metric_name, metric_name)
    if 'loss' in metric_name or metric_name.endswith('_error'):
        return True


def calculate_or_extract_score(metric, y_true, y_pred, config, outputs):
    try:
        if metric.startswith('vw'):
            return extract_score(metric, outputs)
        return calculate_score(metric, y_true, y_pred, config)
    except Exception, ex:
        return '%s: %s' % (type(ex).__name__, ex)


def _parse_vw_metric(metric):
    if metric.startswith('vw_train'):
        _prefix, stage, metric_name = metric.split('_', 2)
    else:
        _prefix, metric_name = metric.split('_', 1)
        stage = 'test'
    return stage, metric_name


def _get_stage(metric):
    return _parse_vw_metric(metric)[0]


def extract_score(metric, outputs):
    stage, metric = _parse_vw_metric(metric)
    outputs = outputs.get(stage)

    if not outputs:
        raise ValueError('error: No output for stage %r' % stage)

    values = [x.get(metric) for x in outputs]

    for item in values:
        if item is None:
            raise ValueError('Metric (%s)%s not found. Available metrics: %s' % (stage, metric, outputs[0].keys()))

    try:
        values = [float(x) for x in values]
    except Exception:
        if values[0].endswith(' h'):
            return values
        return None

    return values


def calculate_score(metric, y_true, y_pred, config):
    sample_weight = get_sample_weight(y_true, config.get('weight_metric'))

    threshold = config.get('threshold')
    min_label = config.get('min_label')
    max_label = config.get('max_label')

    extra_args = {'sample_weight': sample_weight}
    if metric.endswith('_w'):
        metric = metric[:-2]
    else:
        extra_args = {}

    fullname = metrics_shortcuts.get(metric, metric)

    import sklearn.metrics
    if fullname in globals():
        func = globals()[fullname]
    else:
        func = getattr(sklearn.metrics, fullname, None)
        if func is None:
            sys.exit('Cannot find %r in sklearn.metrics' % (fullname, ))

    metric_type = metrics_param.get(fullname)

    if metric_type == 'y_prob':
        # brier_score_loss
        if min_label is None or max_label is None:
            raise ValueError('Cannot calculate on multiclass')
        delta = float(max_label - min_label)
        assert delta
        y_true = (y_true - min_label) / delta
        y_pred = (y_pred - min_label) / delta
        y_pred = np.minimum(y_pred, 1)
        y_pred = np.maximum(y_pred, 0)
        return func(y_true, y_pred, **extra_args)
    elif metric_type == 'y_score':
        # auc, mse
        return func(y_true, y_pred, **extra_args)
    elif metric_type == 'y_pred':
        if threshold is not None:
            y_true = y_true > threshold
            y_pred = y_pred > threshold
        return func(y_true, y_pred, **extra_args)
    else:
        raise ValueError('Unknown metric: %r' % metric)


def main_tune(metric, config, filename, format, args, preprocessor_base, kfold, ignoreheader, workers, feature_mask_retrain, show_num_features):
    if preprocessor_base is None:
        preprocessor_base = []
    else:
        preprocessor_base = preprocessor_base.to_options()

    if not metric:
        sys.exit('Provide metric to optimize for with "--metric METRIC"')

    optimization_metric = metric[0]
    other_metrics = metric[1:]

    best_preprocessor_opts = None
    best_vw_options = None
    best_result = None
    already_done = {}

    preprocessor_variants = list(expand(args, only=Preprocessor.ALL_OPTIONS_DASHDASH))
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
            assert isinstance(filename, basestring), filename
            vw_filename = filename

            # XXX preprocessor can make sense too
            assert not preprocessor, 'TODO'
        else:
            vw_filename = get_temp_filename('vw_filename')
            to_cleanup.append(vw_filename)
            convert_any_to_vw(
                source=filename,
                format=format,
                output_filename=vw_filename,
                columnspec=config.get('columnspec'),
                named_labels=config.get('named_labels'),
                weights=config.get('weights'),
                preprocessor=preprocessor_opts,
                ignoreheader=ignoreheader,
                workers=workers)

        vw_args = [x for x in my_args if x not in Preprocessor.ALL_OPTIONS_DASHDASH]

        this_best_result, this_best_options = vw_optimize_over_cv(
            vw_filename,
            kfold,
            vw_args,
            optimization_metric,
            config,
            workers=workers,
            other_metrics=other_metrics,
            feature_mask_retrain=feature_mask_retrain,
            show_num_features=show_num_features)

        is_best = ''
        if this_best_result is not None and (best_result is None or this_best_result < best_result):
            best_result = this_best_result
            best_vw_options = this_best_options
            best_preprocessor_opts = preprocessor_opts
            is_best = '*'

        if preprocessor_opts:
            log_always('Best options with %s = %s', preprocessor_opts or 'no preprocessing', this_best_options)
        log_always('Best %s with %r = %s%s', optimization_metric, preprocessor_opts or 'no preprocessing', _frmt_score_short(this_best_result), is_best)
        # print 'Improvement over no l1=%.4f. Improvement over initial guess=%.4f' % (no_l1_result - best_result[0], initial_l1_result - best_result[0])

    # XXX don't show this if preprocessor is not enabled and not tuned
    log_always('Best preprocessor options = %s', best_preprocessor_opts or '<none>')
    log_always('Best vw options = %s', best_vw_options)
    log_always('Best %s = %s', optimization_metric, _frmt_score_short(best_result))
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


def mean_h(values):
    if not values:
        return str(values)
    suffix = ''
    if isinstance(values[0], basestring):
        if all(x.endswith(' h') for x in values):
            values = [x[:-2] for x in values]
            values = [float(x) for x in values]
            suffix = ' h'
        else:
            return str(values)
    return np.mean(values), suffix


def _frmt_score(x):
    suffix = ''
    if isinstance(x, list):
        if METRIC_FORMAT == 'mean':
            x, suffix = mean_h(x)
        else:
            return str(x)
    if isinstance(x, float):
        if x < 0:
            x = -x
        return '%g%s' % (x, suffix)
    return str(x)


def _frmt_score_short(x):
    suffix = ''
    if isinstance(x, basestring):
        return x.strip().split()[0].strip(':')
    if isinstance(x, list):
        if METRIC_FORMAT == 'mean':
            x, suffix = mean_h(x)
        else:
            return str(x)
    if isinstance(x, float):
        if x < 0:
            x = -x
        return '%.4f%s' % (x, suffix)
    return str(x)


def read_argument(args, name):
    assert isinstance(args, list), args
    for item in args:
        if name is None:
            return item
        if item == name:
            name = None


def remove_option(args, name, argument):
    """
    >>> remove_option(["--cache_file", "hello"], "--cache_file", 1)
    []

    >>> remove_option(["-b", "25", "--cache_file", "hello", "-k"], "--cache_file", 1)
    ['-b', '25', '-k']

    >>> remove_option(["-b", "25", "--cache_file", "hello", "-k"], "--cache_file", 0)
    ['-b', '25', 'hello', '-k']
    """
    assert isinstance(args, list), args
    index = 0
    while index < len(args):
        if name == args[index]:
            del args[index:index + 1 + int(argument)]
        else:
            index += 1
    return args


def print_toperrors(toperrors, y_true, y_pred, y_pred_text, filename, format, ignoreheader):
    assert y_true is not None
    assert y_pred is not None
    assert filename is not None
    assert len(y_true) == len(y_pred), (len(y_true), len(y_pred))

    errors = []

    for yp, yp_text, yt, example in zip(y_pred, y_pred_text, y_true, open_anything(filename, format, ignoreheader=ignoreheader)):
        # add hash of the example as a second item so that we get a mix of false positives and false negatives for a given error level
        errors.append((abs(yp - yt), hash(repr(example)), yp_text.strip(), example))

    errors.sort(reverse=True)

    if '.' in toperrors:
        min_error = float(toperrors)
        errors = [x for x in errors if x[0] >= min_error]
    else:
        count = int(toperrors)
        errors = errors[:count]

    output = csv.writer(sys.stdout)

    for err, _hash, yp_text, example in errors:
        row = [yp_text]
        if isinstance(example, list):
            row.extend(example)
        else:
            row.append(str(example))
        output.writerow(row)


def cleanup_vw_train_options(vw_args):
    vw_args = vw_args.split()
    remove_option(vw_args, '--quiet', 0)
    remove_option(vw_args, '-q', 0)
    remove_option(vw_args, '--progress', 1)
    remove_option(vw_args, '-P', 1)
    remove_option(vw_args, '--threads', 0)
    return ' '.join(vw_args)


def main(to_cleanup):
    parser = PassThroughOptionParser()

    # cross-validation and parameter tuning options
    parser.add_option('--kfold', type=int)
    parser.add_option('--workers', type=int)
    parser.add_option('--metric', action='append')
    parser.add_option('--metricformat')

    # class weight option
    parser.add_option('--weight', action='append', help='Class weights to use in CLASS:WEIGHT format', default=[])
    parser.add_option('--weight_train', action='append', help='Class weight to use (during training only), in CLASS:WEIGHT format', default=[])
    parser.add_option('--weight_metric', action='append', help='Class weight to use (for weighted metrics only), in CLASS:WEIGHT format', default=[])

    # vowpal wabbit arguments (those that we care about. everything else is passed through)
    parser.add_option('-r', '--raw_predictions')
    parser.add_option('-p', '--predictions')
    parser.add_option('-f', '--final_regressor')
    parser.add_option('--readable_model')
    parser.add_option('-i', '--initial_regressor')
    parser.add_option('-d', '--data')
    parser.add_option('-a', '--audit', action='store_true')

    parser.add_option('--readconfig')
    parser.add_option('--writeconfig')

    # preprocessing options:
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
    parser.add_option('--parseaudit', action='store_true')
    parser.add_option('--linemode', action='store_true')

    # extra
    parser.add_option('--vw')
    parser.add_option('--feature_mask_retrain', action='store_true')
    parser.add_option('--feature_mask_retrain_args')

    parser.add_option('--tmpid')
    parser.add_option('--foldscript')

    options, args = parser.parse_args()

    globals()['LOG_LEVEL'] += options.lesslogs - options.morelogs + (args.count('--quiet'))
    globals()['KEEPTMP'] = options.keeptmp
    globals()['METRIC_FORMAT'] = options.metricformat or METRIC_FORMAT

    if options.tmpid:
        globals()['TMPID'] = options.tmpid

    if options.foldscript:
        assert options.foldscript in ('perl', 'awk'), options.foldscript
        globals()['FOLDSCRIPT'] = options.foldscript

    if options.parseaudit:
        parseaudit(sys.stdin)
        sys.exit(0)

    if options.kfold is not None and options.kfold <= 1:
        sys.exit('kfold parameter must > 1')

    if options.feature_mask_retrain_args:
        options.feature_mask_retrain = options.feature_mask_retrain_args

    config = {}

    if options.readconfig:
        config = json.load(open(options.readconfig))
        log('vwoptimize config = %s', options.readconfig, log_level=1)

        if 'regressor' in config and options.initial_regressor is None:
            options.initial_regressor = os.path.normpath(os.path.join(os.path.dirname(options.readconfig), config['regressor']))
            if not os.path.exists(options.initial_regressor):
                sys.exit('Cannot find %r referenced from %r' % (options.initial_regressor, options.readconfig))

        globals()['VW_CMD'] = config.get('vw') or VW_CMD

    if options.vw:
        globals()['VW_CMD'] = options.vw

    used_stdin = False
    if options.data is None or options.data in STDIN_NAMES:
        used_stdin = True
        filename = None
    else:
        filename = options.data
        if not os.path.exists(filename):
            sys.exit('File not found: %s' % filename)

    named_labels = read_argument(args, '--named_labels')
    named_labels = _make_proper_list(named_labels, proper_label)
    if named_labels is not None:
        config['named_labels'] = named_labels

    weight = parse_weight(options.weight, config.get('named_labels'))
    weight_train = parse_weight(options.weight_train, config.get('named_labels')) or weight
    weight_metric = parse_weight(options.weight_metric, config.get('named_labels')) or weight

    if weight_train:
        config['weight_train'] = weight_train

    if weight_metric:
        config['weight_metric'] = weight_metric

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

    if options.columnspec:
        config['columnspec'] = _make_proper_list(options.columnspec)
    elif 'columnspec' not in config:
        config['columnspec'] = _make_proper_list(DEFAULT_COLUMNSPEC)

    config['format'] = format

    if options.tovw_simple:
        assert not options.workers or options.workers == 1, options.workers
        _convert_any_to_vw(
            filename,
            format,
            options.tovw_simple,
            weight_train,
            preprocessor,
            config.get('columnspec'),
            config.get('named_labels'),
            ignoreheader=options.ignoreheader)
        sys.exit(0)

    if options.threshold is not None:
        config['threshold'] = options.threshold

    need_tuning = 0

    for arg in args:
        if arg.endswith('?'):
            need_tuning = 1
            break

    options.metric = _make_proper_list(options.metric) or ['vw_average_loss']
    show_num_features = 'num_features' in options.metric
    options.metric = [x for x in options.metric if 'num_features' != x]

    # there are metrics that we calculate from y_true and y_pred, these are listed below. Using these requires extra
    # pass over input to read y_true
    calculated_metrics = [x for x in options.metric if not x.startswith('vw')]

    # these are the metrics we extract from vw output (for "average loss" use "vw_average_loss")
    vw_metrics = [x for x in options.metric if x.startswith('vw')]

    y_true = None

    if calculated_metrics or options.kfold or need_tuning or options.feature_mask_retrain is not None or options.toperrors:
        # cannot work with stdin, write it to a temp file
        if filename is None:
            filename = get_temp_filename(format)
            to_cleanup.append(filename)
            fobj = open(filename, 'wb')
            for line in sys.stdin:
                fobj.write(line)
            flush_and_close(fobj)

    if options.tovw:
        assert format != 'vw', 'Input should be csv or tsv'
        convert_any_to_vw(
            filename,
            format=format,
            output_filename=options.tovw,
            preprocessor=config.get('preprocessor'),
            columnspec=config.get('columnspec'),
            named_labels=config.get('named_labels'),
            weights=config.get('weight_train'),
            ignoreheader=options.ignoreheader,
            workers=options.workers)
        sys.exit(0)

    is_multiclass = any([read_argument(args, '--' + x) for x in 'oaa ect csoaa log_multi recall_tree'.split()])

    if calculated_metrics or options.toperrors:
        assert filename is not None
        y_true = read_y_true(filename, format, config.get('columnspec'), options.ignoreheader, config.get('named_labels'))
        if not len(y_true):
            sys.exit('%s is empty' % filename)
        if not config.get('named_labels') and not is_multiclass:
            min_label = np.min(y_true)
            max_label = np.max(y_true)
            config.setdefault('min_label', min_label)
            config.setdefault('max_label', max_label)
            config.setdefault('threshold', (min_label + max_label) / 2.0)

    if options.report:
        assert options.metric
        assert calculated_metrics
        if options.predictions in STDIN_NAMES:
            if used_stdin:
                sys.exit('Can only use stdin in one argument')
            predictions = sys.stdin
        elif options.predictions:
            predictions = options.predictions
        else:
            sys.exit('Must provide -p')

        assert y_true is not None
        y_pred = _load_first_float_from_each_string(predictions, len(y_true))

        for metric in options.metric:
            log_always('%s = %s', metric, _frmt_score(calculate_score(metric, y_true, y_pred, config)))

        sys.exit(0)

    index = 0
    while index < len(args):
        arg = args[index]
        if arg.startswith('-'):
            next_arg = args[index + 1] if index + 1 < len(args) else ''
            if arg.endswith('?'):
                args[index] = get_tuning_config(arg)
            elif next_arg.endswith('?'):
                args[index:index + 2] = [get_tuning_config(arg + ' ' + next_arg)]
        index += 1

    metric = options.metric or ['vw_average_loss']

    if need_tuning:
        # QQQ --initial_regressor is not passed there
        assert not options.audit, '-a incompatible with parameter tuning'
        vw_args, preprocessor = main_tune(
            metric=metric,
            config=config,
            filename=filename,
            format=format,
            args=args,
            preprocessor_base=preprocessor,
            kfold=options.kfold,
            ignoreheader=options.ignoreheader,
            workers=options.workers,
            feature_mask_retrain=options.feature_mask_retrain,
            show_num_features=show_num_features)
        if vw_args is None:
            sys.exit('tuning failed')
        config['preprocessor'] = str(preprocessor)
        config['vw_train_options'] = cleanup_vw_train_options(vw_args)
    else:
        vw_args = ' '.join(args)
        if options.initial_regressor == '' and config.get('vw_train_options'):
            vw_args = vw_args + ' ' + config.get('vw_train_options')
        else:
            config['vw_train_options'] = cleanup_vw_train_options(vw_args)

    vw_filename = None

    if filename:
        if format == 'vw':
            assert not preprocessor or not preprocessor.to_options(), preprocessor
            vw_filename = filename
        else:
            vw_filename = get_temp_filename('vw')
            to_cleanup.append(vw_filename)

            convert_any_to_vw(
                source=filename,
                format=format,
                output_filename=vw_filename,
                preprocessor=config.get('preprocessor'),
                columnspec=config.get('columnspec'),
                named_labels=config.get('named_labels'),
                weights=config.get('weight_train'),
                ignoreheader=options.ignoreheader,
                workers=options.workers)

    if options.kfold and not need_tuning:
        # QQQ --initial_regressor is not passed there
        # XXX we could skip --cv here if we make main_tune() keep final predictions and raw_predictions for us

        assert vw_filename

        cv_pred_txt, raw_cv_pred_txt, num_features, outputs = vw_cross_validation(
            vw_filename,
            options.kfold,
            vw_args,
            workers=options.workers,
            with_predictions=bool(calculated_metrics) or options.predictions or options.toperrors,
            with_raw_predictions=bool(options.raw_predictions),
            feature_mask_retrain=options.feature_mask_retrain,
            calc_num_features=show_num_features,
            capture_output=set([_get_stage(m) for m in vw_metrics]))

        cv_pred = _load_first_float_from_each_string(cv_pred_txt)

        for metric in options.metric:
            value = calculate_or_extract_score(metric, y_true, cv_pred, config, outputs)
            if value is not None:
                log_always('%s-fold %s = %s', options.kfold, metric, _frmt_score(value))

        if show_num_features and num_features:
            log_always('%s-fold num_features = %s', options.kfold, _frmt_score(num_features))

        if options.predictions:
            write_file(options.predictions, cv_pred_txt)

        if options.raw_predictions:
            write_file(options.raw_predictions, raw_cv_pred_txt)

        if options.toperrors:
            print_toperrors(options.toperrors, y_true, cv_pred, cv_pred_txt, filename=filename, format=format, ignoreheader=options.ignoreheader)

        # all of these are related to CV if --cv is enabled
        options.predictions = None
        options.raw_predictions = None
        options.toperror = None

    final_regressor = options.final_regressor

    config_tmp_filename = None
    if options.writeconfig:
        log('write config = %s', options.writeconfig, log_level=1)
        assert options.writeconfig != options.final_regressor, options.writeconfig

        if final_regressor:
            config['regressor'] = os.path.relpath(final_regressor, os.path.dirname(options.writeconfig))

        config['vw'] = VW_CMD

        config_tmp_filename = options.writeconfig + '.tmp'
        to_cleanup.append(config_tmp_filename)
        output_fobj = open(config_tmp_filename, 'w')
        json.dump(config, output_fobj, sort_keys=True, indent=4)
        output_fobj.write('\n')
        output_fobj.close()

    final_regressor_tmp = None
    if final_regressor:
        final_regressor_tmp = final_regressor + '.tmp'
        to_cleanup.append(final_regressor_tmp)

    if final_regressor_tmp or not (options.kfold or need_tuning):
        my_args = vw_args

        predictions_fname = options.predictions

        if calculated_metrics or options.toperrors:
            if not predictions_fname or predictions_fname in STDOUT_NAMES:
                predictions_fname = get_temp_filename('pred')
                to_cleanup.append(predictions_fname)

        if options.readable_model:
            readable_model = options.readable_model
        elif show_num_features:
            readable_model = get_temp_filename('readable_model')
            to_cleanup.append(readable_model)
        else:
            readable_model = None

        vw_cmd = get_vw_command(
            to_cleanup,
            vw_filename,
            my_args,
            initial_regressor=options.initial_regressor,
            final_regressor=final_regressor_tmp,
            predictions=predictions_fname,
            raw_predictions=options.raw_predictions,
            audit=options.audit,
            feature_mask_retrain=options.feature_mask_retrain,
            readable_model=readable_model)

        if len(vw_cmd) == 1 and vw_filename is None:
            vw_cmd = vw_cmd[-1]

            # don't want to capture stderr here, so vw_ metrics don't work there

            if format == 'vw':
                popen = Popen(vw_cmd, stdin=sys.stdin, log_level=1)
            else:
                popen = Popen(vw_cmd, stdin=subprocess.PIPE, log_level=1)
                for row in open_anything(sys.stdin, format, ignoreheader=options.ignoreheader, force_unbuffered=options.linemode):
                    # XXX weights
                    line = convert_row_to_vw(row, columnspec=config.get('columnspec'), preprocessor=preprocessor)
                    popen.stdin.write(line)
                    # subprocess.Popen is unbuffered by default
                popen.stdin.close()

            if popen.wait() != 0:
                sys.exit(1)
        else:
            system(vw_cmd)

        if options.predictions in STDOUT_NAMES and options.predictions != predictions_fname:
            for line in open(predictions_fname):
                sys.stdout.write(line)

        y_pred = None
        y_pred_text = None

        if options.toperrors or calculated_metrics:
            assert predictions_fname is not None
            assert y_true is not None
            y_pred, y_pred_text = _load_first_float_from_each_string(predictions_fname, len(y_true), with_text=True)

        for metric in calculated_metrics:
            log_always('%s = %s', metric, _frmt_score(calculate_score(metric, y_true, y_pred, config)))

        if show_num_features and readable_model:
            log_always('num_features = %s', get_num_features(readable_model))

        if options.toperrors:
            print_toperrors(options.toperrors, y_true, y_pred, y_pred_text, filename=filename, format=format, ignoreheader=options.ignoreheader)

    if final_regressor_tmp:
        os.rename(final_regressor_tmp, final_regressor)

    if config_tmp_filename:
        os.rename(config_tmp_filename, options.writeconfig)

    unlink(*to_cleanup)


if __name__ == '__main__':
    to_cleanup = []
    try:
        main(to_cleanup)
    finally:
        unlink(*to_cleanup)
