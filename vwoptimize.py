#!/usr/bin/env python
# coding: utf-8
"""Wrapper for Vowpal Wabbit that does cross-validation and hyper-parameter tuning"""
__version__ = '0.10.2dev'
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
import json
import pprint
import unicodedata
from itertools import izip, izip_longest
from collections import deque
from pipes import quote
import numpy as np
try:
    import threading
except Exception:
    import dummy_threading as threading


csv.field_size_limit(10000000)
MINIMUM_LOG_IMPORTANCE = 1
TMPID = str(os.getpid())
TMP_PREFIX = None
KEEPTMP = False
STDIN_NAMES = ('/dev/stdin', '-')
STDOUT_NAMES = ('/dev/stdout', 'stdout')
VW_CMD = 'vw'
VOWPAL_WABBIT_ERRORS = "error|won't work right|errno|can't open|vw::vw_exception|need a cache file for multiple passes|cannot be specified"
DEFAULT_COLUMNSPEC = 'y,text,*'
METRIC_FORMAT = 'mean'
DEFAULT_METRICS = ['vw_average_loss']

AWK_TRAINSET = "awk '(NR - $fold) % KFOLDS != 0' VW |"
AWK_TESTSET = "awk '(NR - $fold) % KFOLDS == 0' VW |"
PERL_TRAINSET = "perl -nE 'if ((++$NR - $fold) % KFOLDS != 0) { print $_ }' VW |"
PERL_TESTSET = "perl -nE 'if ((++$NR - $fold) % KFOLDS == 0) { print $_ }' VW |"
options = None

if 'darwin' in sys.platform:
    # awk is slow on Mac OS X
    FOLDSCRIPT = 'perl'
else:
    FOLDSCRIPT = 'awk'


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
    assert TMP_PREFIX
    fname = '%s/%s.%s.%s' % (TMP_PREFIX, TMPID, counter[0], suffix)
    assert not os.path.exists(fname), 'internal error: %s' % fname
    return fname


log_lock = threading.RLock()


def log(s, *params, **kwargs):
    importance = kwargs.pop('importance', None) or 0
    assert not kwargs, kwargs
    if importance >= MINIMUM_LOG_IMPORTANCE:
        with log_lock:
            try:
                sys.stdout.flush()
            except IOError:
                pass
            try:
                s = s % params
            except Exception:
                s = '%s %r' % (s, params)
            sys.stderr.write('%s\n' % (s, ))


def log_always(*args, **kwargs):
    kwargs['importance'] = MINIMUM_LOG_IMPORTANCE
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
    if isinstance(data, np.ndarray):
        data = '\n'.join(str(x) for x in data)
    elif isinstance(data, list):
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
        if ext in ['vw', 'csv', 'tsv', 'tab']:
            return ext


class simple_reader(object):

    def __init__(self, source):
        self.source = source

    def __iter__(self):
        return self

    def next(self):
        row = self.source.next().split("\t")
        row[-1] = row[-1].rstrip()
        return row


def open_anything(source, format, ignoreheader, force_unbuffered=False):
    source = open_regular_or_compressed(source)

    if force_unbuffered:
        # simply disabling buffering is not enough, see this for details: http://stackoverflow.com/a/6556862
        source = iter(source.readline, '')

    if format == 'vw':
        return source

    if format == 'tsv':
        reader = csv.reader(source, csv.excel_tab)
        if ignoreheader:
            reader.next()
    elif format == 'csv':
        reader = csv.reader(source, csv.excel)
        if ignoreheader:
            reader.next()
    elif format == 'tab':
        reader = simple_reader(source)
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

    def _match_long_opt(self, opt):
        # This disable shortcuts so that '--ignorecount' is not parsed '--ignore' which conflics with "vw --ignore"
        if opt in self._long_opt:
            return opt
        raise optparse.BadOptionError(opt)


def system(cmd, importance=1, repeat_on_error=0):
    if isinstance(cmd, deque):
        results = []
        for item in cmd:
            r = system(item, importance=importance, repeat_on_error=repeat_on_error)
            results.append(r)
        return '\n'.join(results).strip()

    sys.stdout.flush()
    start = time.time()

    popen = Popen(cmd, shell=True, importance=importance)

    if popen.stdout is not None or popen.stderr is not None:
        out, err = popen.communicate()
    else:
        out, err = '', ''

    retcode = popen.wait()

    if retcode:
        log_always('%s [%.1fs] %s', '-' if retcode == 0 else '!', time.time() - start, get_command_name(cmd))

    if retcode:
        if repeat_on_error > 0:
            return system(cmd, importance=importance, repeat_on_error=repeat_on_error - 1)
        sys.exit(1)

    return (out or '') + (err or '')


def split_file(source, nfolds=None, ignoreheader=False, importance=0, minfoldsize=10000):
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


def get_command_name(params):
    if not isinstance(params, dict):
        return str(params)
    name = params.get('name', None)
    args = params.get('args')

    if isinstance(args, list):
        args = ' '.join(args)

    if name:
        return '[%s] %s' % (name, args)
    else:
        return str(args)


def Popen(params, **kwargs):
    command_name = get_command_name(params)

    if isinstance(params, dict):
        params = params.copy()
        args = params.pop('args')
        params.pop('name', None)
        params.update(kwargs)
    else:
        args = params
        params = kwargs

    importance = params.pop('importance', None)
    if importance is None:
        importance = 0

    params.setdefault('preexec_fn', die_if_parent_dies)

    log('+ %s', command_name, importance=importance)

    popen = subprocess.Popen(args, **params)
    return popen


def run_subprocesses(cmds, workers=None, importance=None):
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

                popen = Popen(this_cmd, shell=True, importance=importance)
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
                    log_always('failed: %s', popen._cmd.get('args', get_command_name(popen._cmd)))
                    return None, outputs
                else:
                    log('%s %s', '-' if retcode == 0 else '!', get_command_name(popen._cmd), importance=importance)

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
        readable_model=None,
        only_test=False,
        fix_cache_file=False,
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

    intermediate_model_filename = final_regressor

    final_options = []

    if audit:
        final_options += ['-a']

    if readable_model:
        final_options += ['--readable_model', readable_model]

    vw_args = vw_args.split()

    if fix_cache_file:
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
        '-p %s' % predictions if predictions else '',
        '-r %s' % raw_predictions if raw_predictions else '',
        '-t' if only_test else '',
    ] + vw_args

    if only_test:
        return _as_dict(training_command + final_options, name=name)

    return deque([_as_dict(training_command + final_options, name=name)])


def vw_cross_validation(
        vw_filename,
        kfold,
        vw_args,
        vw_test_args,
        workers=None,
        with_predictions=False,
        with_raw_predictions=False,
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

    model_prefix = get_temp_filename('model') + '.$fold'
    model_filename = model_prefix + '.bin' if testset else None

    if with_predictions:
        p_filename = '%s.predictions' % model_prefix
    else:
        p_filename = None

    if with_raw_predictions:
        r_filename = '%s.raw' % model_prefix
    else:
        r_filename = None

    if calc_num_features:
        readable_model = model_prefix + '.readable'
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
        readable_model=readable_model,
        fix_cache_file=kfold > 1,
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
            vw_args=vw_test_args,
            initial_regressor=model_filename,
            predictions=p_filename,
            raw_predictions=r_filename,
            only_test=True,
            fix_cache_file=kfold > 1,
            name='test')

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
        success, outputs = run_subprocesses(commands, workers=workers, importance=-1)

        # check outputs first, the might be a valuable error message there
        outputs = dict((key, [parse_vw_output(out) for out in value]) for (key, value) in outputs.items())

        if not success:
            vw_failed()

        for name in to_cleanup:
            if not os.path.exists(name):
                vw_failed('missing %r' % (name, ))

        predictions = []
        for items in izip_longest(*[open(x) for x in p_filenames]):
            predictions.extend([float(x.split()[0]) for x in items if x is not None])

        if predictions:
            if np.equal(0, np.max(np.abs(np.mod(predictions[:10000], 1)))):
                predictions = np.array(predictions, dtype=int)
            else:
                predictions = np.array(predictions)

        raw_predictions = []
        for items in izip_longest(*[open(x) for x in r_filenames]):
            raw_predictions.extend([x for x in items if x is not None])

        num_features = [get_num_features(name) for name in readable_models]

        return predictions, raw_predictions, num_features, outputs

    finally:
        unlink(*to_cleanup)


def extract_test_args(vw_args):
    if isinstance(vw_args, basestring):
        vw_args = vw_args.split()

    test_args = []

    loss_function = read_argument(vw_args, '--loss_function')

    if loss_function:
        test_args.append('--loss_function ' + loss_function)

    test_opts = [
        '--probabilities',
        '--onethread',
    ]

    for opt in test_opts:
        if opt in vw_args:
            test_args.append(opt)

    return ' '.join(test_args)


def vw_validation(
        to_cleanup,
        vw_filename,
        vw_validation_filename,
        vw_args,
        vw_test_args,
        workers=None,
        with_predictions=False,
        with_raw_predictions=False,
        calc_num_features=False,
        capture_output=False):

    assert os.path.exists(vw_validation_filename), vw_validation_filename

    if hasattr(capture_output, '__contains__') and '' in capture_output:
        capture_output = True

    vw_args = vw_args.replace('--quiet', '')
    model_prefix = get_temp_filename('model')
    model_filename = model_prefix + '.bin'
    to_cleanup.append(model_filename)

    if with_predictions:
        p_filename = '%s.predictions' % model_prefix
        to_cleanup.append(p_filename)
    else:
        p_filename = None

    if with_raw_predictions:
        r_filename = '%s.raw' % model_prefix
        to_cleanup.append(r_filename)
    else:
        r_filename = None

    if calc_num_features:
        readable_model = model_prefix + '.readable'
        to_cleanup.append(readable_model)
    else:
        readable_model = None

    command = get_vw_command(
        to_cleanup,
        vw_filename,
        vw_args=vw_args,
        final_regressor=model_filename,
        predictions=None,
        raw_predictions=None,
        readable_model=readable_model,
        fix_cache_file=True,
        name='train')

    for item in command:
        if capture_output is True or item['name'] in capture_output:
            item['stderr'] = subprocess.PIPE
        else:
            item['args'] += ' --quiet'

    training_out = system(command, importance=-1, repeat_on_error=1)

    outputs = {}
    if training_out:
        outputs['train'] = [parse_vw_output(training_out)]

    testing_command = get_vw_command(
        to_cleanup,
        vw_validation_filename,
        vw_args=vw_test_args,
        initial_regressor=model_filename,
        predictions=p_filename,
        raw_predictions=r_filename,
        only_test=True,
        name='test')

    if capture_output is True or 'test' in capture_output:
        testing_command['stderr'] = subprocess.PIPE
    else:
        testing_command['args'] += ' --quiet'

    validation_out = system(testing_command, importance=-1, repeat_on_error=1)
    if validation_out:
        outputs['test'] = [parse_vw_output(validation_out)]

    for name in to_cleanup:
        if not os.path.exists(name):
            vw_failed('missing %r' % (name, ))

    if p_filename:
        predictions = []
        for line in open(p_filename):
            predictions.append(float(line.split()[0]))
        predictions = np.array(predictions)
    else:
        predictions = []

    if r_filename:
        raw_predictions = open(r_filename).readlines()
    else:
        raw_predictions = []

    if readable_model:
        num_features = get_num_features(readable_model)
    else:
        num_features = None

    return predictions, raw_predictions, num_features, outputs, model_filename


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


def _load_predictions(file, size=None, with_text=False, named_labels=None, with_weights=False, examples=None):
    filename = file
    if isinstance(file, list):
        filename = file
    elif hasattr(file, 'read'):
        pass
    elif isinstance(file, basestring):
        if file in STDOUT_NAMES:
            sys.exit('Will not read %s' % file)
        file = open_regular_or_compressed(file)
    else:
        raise AssertionError(limited_repr(file))

    result = []
    result_text = []
    importance_weights = [] if with_weights else None

    for line in file:
        try:
            text = line.strip()
            if with_text:
                result_text.append(text)
            items = text.split()
            label = items[0]
            if importance_weights is not None:
                if len(items) >= 2:
                    w = items[1]
                    if w.startswith("'") or '|' in w:
                        w = 1.0
                    else:
                        w = float(w)
                    importance_weights.append(w)
            if named_labels is not None:
                if label not in named_labels:
                    sys.exit('Unexpected label %r from %r' % (label, filename))
                result.append(label)
            else:
                result.append(float(label))
        except:
            sys.stderr.write('Error while parsing %r\nin %r\n' % (line, limited_repr(filename)))
            raise

        if examples is not None and len(result) >= examples:
            break

    if size is not None:
        if len(result) < size:
            sys.exit('Too few items in %s: found %r, expecting %r' % (limited_repr(filename), len(result), size))

        if len(result) > size:
            mult = int(len(result) / size)
            if size * mult == len(result):
                # if --passes N option was used, then the number of predictions will be N times higher
                result = result[-size:]
            else:
                sys.exit('Too many items in %s: found %r, expecting multiply of %r' % (limited_repr(filename), len(result), size))

    retvalue = [np.array(result)]

    if with_text:
        retvalue.append(result_text)

    if with_weights:
        if not importance_weights:
            retvalue.append(None)
        else:
            if len(importance_weights) != len(result):
                sys.exit('Could not parse importance weights')
            importance_weights = np.array(importance_weights)
            retvalue.append(importance_weights)

    if len(retvalue) == 1:
        return retvalue[0]

    return tuple(retvalue)


class BaseParam(object):

    PRINTABLE_KEYS = 'opt init min max values format extra omit'.split()
    _cast = None

    @classmethod
    def cast(cls, value):
        if value is None:
            return None
        if value == '':
            return None
        if cls._cast is None:
            return value
        return cls._cast(value)

    def pack(self, value):
        if self._pack is None:
            return value
        return self._pack(value)

    def unpack(self, value):
        if self._unpack is None:
            return value
        return self._unpack(value)

    def __init__(self, opt, init=None, min=None, max=None, format=None, pack=None, unpack=None, extra=None, omit=None, merge=False):
        self.opt = opt
        self.init = self.cast(init)
        self.min = self.cast(min)
        self.max = self.cast(max)
        self.format = format
        self._pack = pack
        self._unpack = unpack
        self.extra = None
        self.omit = omit
        self.separator = '' if merge else ' '

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
        if init is None:
            init = self.avg(self.min, self.max)
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
        param = self.cast(param)
        format = self.format or '%s'
        extra_arg = format % param
        return self.opt + self.separator + extra_arg + ' '.join(self.extra or [])


class IntegerParam(BaseParam):
    _cast = int


class RandIntParam(BaseParam):
    _cast = int


class FloatParam(BaseParam):
    _cast = float


class LogParam(BaseParam):
    _cast = float

    def __init__(self, opt, **kwargs):
        BaseParam.__init__(self, opt, pack=np.log, unpack=np.exp, **kwargs)


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

    def get_extra_args(self, param):
        if param:
            return self.opt
        else:
            return ''


def get_format(value):
    """
    >>> get_format("1e-5")
    '%.0e'

    >>> get_format("1e5")
    '%.0e'

    >>> get_format("0.")
    '%.0f'

    >>> get_format("0.5")
    '%.1f'

    >>> get_format("0.5")
    '%.1f'

    >>> get_format("0.50")
    '%.2f'

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
        return '%%.%sf' % len(x)


def parse_tuning_args(args):
    """
    >>> parse_tuning_args('--lowercase_features?'.split())
    [BinaryParam(opt='--lowercase_features')]

    >>> parse_tuning_args('--onethread --lowercase_features?'.split())
    ['--onethread', BinaryParam(opt='--lowercase_features')]
    """
    assert isinstance(args, list), type(args)
    args = args[:]
    index = 0
    while index < len(args):
        arg = args[index]
        if arg.startswith('-'):
            next_arg = args[index + 1] if index + 1 < len(args) else ''
            if arg.endswith('?'):
                args[index] = get_tuning_config(arg)
            elif next_arg.endswith('?') and not next_arg.startswith('-'):
                args[index:index + 2] = [get_tuning_config(arg + ' ' + next_arg)]
        index += 1
    return args


def get_tuning_config(config):
    """
    >>> get_tuning_config('--hash_seed randint(10000)?')
    RandIntParam(opt='--hash_seed', min=0, max=10000)

    >>> get_tuning_config('--lowercase?')
    BinaryParam(opt='--lowercase')

    >>> get_tuning_config('--nn 1..10?')
    IntegerParam(opt='--nn', min=1, max=10)

    >>> get_tuning_config('--nn 1..10??')
    IntegerParam(opt='--nn', min=1, max=10, omit=True)

    >>> get_tuning_config('--nn 0..10?')
    IntegerParam(opt='--nn', min=0, max=10)

    >>> get_tuning_config('--ngram 2..5?')
    IntegerParam(opt='--ngram', min=2, max=5)

    >>> get_tuning_config('-b 10..25?')
    IntegerParam(opt='-b', min=10, max=25)

    >>> get_tuning_config('--learning_rate 0.5?')
    FloatParam(opt='--learning_rate', init=0.5, format='%.1f')

    >>> get_tuning_config('--learning_rate 0.50?')
    FloatParam(opt='--learning_rate', init=0.5, format='%.2f')

    >>> get_tuning_config('--l1 1e-07?')
    LogParam(opt='--l1', init=1e-07, format='%.0e')

    >>> get_tuning_config('--l1 1.0E-07?')
    LogParam(opt='--l1', init=1e-07, format='%.1e')

    >>> get_tuning_config('--l1 ..1.2e-07..?')
    LogParam(opt='--l1', init=1.2e-07, format='%.1e')

    >>> get_tuning_config('--l1 1e-10..1e-05?')
    LogParam(opt='--l1', min=1e-10, max=1e-05, format='%.0e')

    >>> get_tuning_config('--loss_function squared/hinge/percentile?')
    ValuesParam(opt='--loss_function', values=['squared', 'hinge', 'percentile'])

    >>> get_tuning_config('--loss_function /hinge/percentile?')
    ValuesParam(opt='--loss_function', values=['', 'hinge', 'percentile'])

    >>> get_tuning_config('--classweight 1:0.1..1.5?')
    FloatParam(opt='--classweight 1:', min=0.1, max=1.5, format='%.1f')

    >>> get_tuning_config('--classweight 1:0.1..1.5?').get_extra_args(1)
    '--classweight 1:1.0'
    """
    if isinstance(config, basestring):
        config = config.split()

    assert config[-1].endswith('?'), config

    if config[-1].endswith('??'):
        config[-1] = config[-1][:-1]
        omit = True
    else:
        omit = None

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
            # XXX recursive definition? need a proper parser then '(--ngram 2/3? --skips 0..3?)/--ngram 1?'
            return ValuesParam(opt='', values=[(prefix + x if x else '') for x in first.split('/')])
        else:
            return BinaryParam(prefix + first)

    value = config[-1]
    value = value[:-1]

    if value.count(':') == 1:
        value_prefix, value = value.split(':')
        opt = config[0] + ' ' + value_prefix + ':'
        merge = True
    else:
        opt = config[0]
        merge = False

    params = {
        'opt': opt,
        'merge': merge,
        'omit': omit,
    }

    if '/' in value:
        return ValuesParam(values=value.split('/'), **params)

    is_log = 'e' in value.lower()
    type = None

    if value.count('..') == 2:
        min, init, max = value.split('..')
        format = sorted([get_format(min), get_format(init), get_format(max)])[-1]
        is_float = '.' in min or '.' in init or '.' in max

        params.update({
            'min': min,
            'init': init,
            'max': max,
            'format': format
        })

    elif '..' in value:
        min, max = value.split('..')
        is_float = '.' in min or '.' in max
        format = sorted([get_format(min), get_format(max)])[-1]

        params.update({
            'min': min,
            'max': max,
            'format': format
        })

    elif value.startswith('randint('):
        value = value[8:].rstrip(')')
        min = 0
        value = int(value)
        params.update({
            'min': 0,
            'max': value,
        })

        type = RandIntParam

        return RandIntParam(**params)

    else:
        is_float = '.' in value
        format = get_format(value)

        params.update({
            'init': value,
            'format': format
        })

    if type is None:
        if is_log:
            type = LogParam
        elif is_float:
            type = FloatParam
        else:
            type = IntegerParam

    return type(**params)


def split_metrics(metrics):
    show_num_features = 'num_features' in metrics
    calculated_metrics = [x for x in metrics if not x.startswith('vw') and x != 'num_features']
    vw_metrics = [x for x in metrics if x.startswith('vw')]
    return calculated_metrics, vw_metrics, show_num_features


def best_result_update(best_result, result, args):
    if best_result is None:
        return ''
    best_marker = ''
    with log_lock:
        for marker, (best_value, best_args) in best_result.items():
            if result >= best_value:
                continue
            best_result[marker] = (result, args)
            if (len(marker), marker) > (len(best_marker), best_marker):
                best_marker = marker

    return best_marker


def run_cached(cache, cache_key, func, *args, **kwargs):
    cache_key = str(cache_key)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    result = func(*args, **kwargs)
    cache[cache_key] = result
    return result


table = None


def run_single_iteration(vw_filename,
                         vw_validation_filename,
                         vw_test_filename,
                         kfold,
                         args,
                         workers,
                         metrics,
                         y_true,
                         sample_weight,
                         config,
                         best_result,
                         with_predictions,
                         validation_holdout):
    global table

    list_args = [x for x in args if x.strip()]

    assert isinstance(args, list), args
    args = ' '.join(str(x) for x in args)
    args = re.sub('\s+', ' ', args).strip()

    calculated_metrics, vw_metrics, show_num_features = split_metrics(metrics)
    metric = metrics[0]

    log('Trying %s %s...', VW_CMD, args, importance=-1)
    cleanup = []
    test_args = extract_test_args(args)
    model_filename = None

    try:
        if vw_validation_filename is not None:
            y_pred, raw_pred_text, num_features, outputs, model_filename = vw_validation(
                cleanup,
                vw_filename,
                vw_validation_filename,
                vw_args=args,
                vw_test_args=test_args,
                workers=workers,
                with_predictions=with_predictions or bool(calculated_metrics),
                calc_num_features=show_num_features,
                capture_output=set([_get_stage(m) for m in vw_metrics]))
        else:
            if vw_test_filename is not None:
                sys.exit('--test not implemented for kfold')
            y_pred, raw_pred_text, num_features, outputs = vw_cross_validation(
                vw_filename,
                kfold,
                vw_args=args,
                vw_test_args=test_args,
                workers=workers,
                with_predictions=with_predictions or bool(calculated_metrics),
                calc_num_features=show_num_features,
                capture_output=set([_get_stage(m) for m in vw_metrics]))
    except KeyboardInterrupt:
        raise
    except BaseException, ex:
        if type(ex) is not SystemExit:
            traceback.print_exc()
        log('Result %s %s : error: %s', VW_CMD, args, ex, importance=2)
        return (None, None)
    else:
        if y_true is not None:
            if calculated_metrics and len(y_true) != len(y_pred):
                sys.exit('Internal error: expected %r predictions, got %r' % (len(y_true), len(y_pred)))

            if raw_pred_text and len(y_true) != len(raw_pred_text):
                sys.exit('Internal error: expected %r raw predictions, got %r' % (len(y_true), len(raw_pred_text)))

        results_h = []

        if validation_holdout and y_pred is not None and y_true is not None:
            assert 0 < validation_holdout < 1, validation_holdout
            val_size = int(round(len(y_true) * (1.0 - validation_holdout)))
            y_true_val = y_true[:val_size]
            assert len(y_true_val), len(y_true_val)
            results = [calculate_or_extract_score(m, y_true_val, y_pred[:val_size], config, outputs, sample_weight[:val_size] if sample_weight is not None else None, num_features=num_features) for m in metrics]
            y_true_h = y_true[val_size:]
            assert len(y_true_h), len(y_true_h)
            if len(y_true_h):
                results_h = [calculate_or_extract_score(m, y_true_h, y_pred[val_size:], config, outputs, sample_weight[val_size:] if sample_weight is not None else None, num_features=num_features) for m in metrics]
        else:
            results = [calculate_or_extract_score(m, y_true, y_pred, config, outputs, sample_weight, num_features=num_features) for m in metrics]

        result = results[0]

        if isinstance(result, basestring):
            sys.exit('Cannot calculate %r: %s' % (metric, result))

        if isinstance(result, list):
            try:
                result, suffix = mean_h(result)
            except Exception:
                log_always("Failed to calculate mean from %r", result)
                raise

        if not isinstance(result, (int, long, float)):
            sys.exit('Bad metric for tuning: %s (value=%r)' % (metric, result))

        if not is_loss(metric):
            result = -result

        is_best = best_result_update(best_result, result, args)

        values = [_frmt_score(x) for x in results]
        values[1:] = [x.split()[0].rstrip(':') for x in values[1:]]
        values[0] += is_best or ' '

        values = ['%s=%s' % (x, y) for (x, y) in zip(metrics, values)]

        if results_h:
            values_h = [_frmt_score(x) for x in results_h]
            values += ['%s(hold)=%s' % (x, y) for (x, y) in zip(metrics, values_h)]

        values = ['Result', VW_CMD] + list_args + [':'] + values

        if table is None or len(table) != len(values):
            table = [len(x) for x in values]
        else:
            table = [max(len(x), t) for (x, t) in zip(values, table)]

        new_values = []
        for value, size in zip(values, table):
            value += ' ' * (size - len(value))
            new_values.append(value)

        new_values = ' '.join(new_values).rstrip()

        log(new_values, importance=2 + int(len(is_best)))

        return result, is_best
    finally:
        unlink(*cleanup)


class InterruptOptimization(Exception):
    pass


MARKER_LOCALBEST = '+'
MARKER_BRANCHBEST = '* '
MARKER_BEST = '** '


def vw_optimize(vw_filename, vw_validation_filename, vw_test_filename, y_true, kfold, args, metrics, config, sample_weight, workers, best_result, validation_holdout):
    gridsearch_params = []
    tunable_params = []
    base_args = []
    assert isinstance(args, list), args

    for param in args:
        if isinstance(param, (ValuesParam, BinaryParam)):
            gridsearch_params.append(param)
        elif isinstance(param, BaseParam):
            tunable_params.append(param)
        else:
            base_args.append(param)

    extra_args = ''
    if best_result is None:
        best_result = {}
    cache = {}

    def run(params):
        log('Parameters: %r', params, importance=-1)
        args = [extra_args]

        for param_config, param in zip(tunable_params, params):
            extra_arg = param_config.get_extra_args(param)
            if extra_arg:
                args.append(extra_arg)

        result, is_best = run_cached(
            cache,
            args,
            run_single_iteration,
            vw_filename,
            vw_validation_filename,
            vw_test_filename,
            kfold,
            args,
            workers,
            metrics,
            y_true,
            sample_weight,
            config,
            best_result,
            with_predictions=False,
            validation_holdout=validation_holdout)

        return result

    already_done = {}

    gridsearch_params = expand(gridsearch_params, withextra=True)
    log('Grid-search: %r', gridsearch_params)
    initial_params_init = [x.packed_init() for x in tunable_params]
    initial_params_db = Simple1NN()

    best_result[MARKER_BRANCHBEST] = (float('inf'), None)

    for _score, params, params_vector in gridsearch_params:
        if tunable_params:
            best_result[MARKER_LOCALBEST] = (float('inf'), None)

        params_normalized = vw_normalize_params(base_args + params)

        if params_normalized != params:
            log('Normalized params %r %r -> %r', base_args, params, params_normalized, importance=-1)

        params_as_str = ' '.join(params_normalized)

        if params_as_str in already_done:
            log('Skipping %r (same as %r)', ' '.join(params), ' '.join(already_done[params_as_str]), importance=-1)
            continue

        already_done[params_as_str] = params

        extra_args = params_as_str
        need_separator = False

        if tunable_params:
            import scipy.optimize

            options = {'xtol': 0.001, 'ftol': 0.001}

            results = initial_params_db.find_nearest(params_vector)
            if results:
                t_params = np.mean(results, axis=0)
            else:
                t_params = initial_params_init

            try:
                optresult = scipy.optimize.minimize(run, t_params, method='Nelder-Mead', options=options)
            except InterruptOptimization, ex:
                log(str(ex), importance=1)
            else:
                need_separator = True
                initial_params_db.add_observation(np.array(params_vector), optresult.x)
        else:
            try:
                run([])
            except InterruptOptimization, ex:
                log(str(ex), importance=1)

        if need_separator:
            log('', importance=1)

    return best_result[MARKER_BRANCHBEST]


def setup_hyperopt_Trials(domain, workers):
    from vwoptimizelib.third_party.hyperopt import base
    from vwoptimizelib.third_party.hyperopt.utils import coarse_utcnow
    from vwoptimizelib.third_party.hyperopt.fmin import FMinIter
    import threading
    from Queue import Queue
    import traceback

    class MT_Trials(base.Trials):
        """Multithreading-enabled Trials implementation for hyperopt.
        """

        async = True

        def __init__(self, domain, poolsize=None):
            base.Trials.__init__(self)
            self.domain = domain
            self.queue = Queue()
            self.pool = []
            if poolsize is None:
                import multiprocessing
                poolsize = max(1, multiprocessing.cpu_count() - 1)
                log('setting poolsize = %s', poolsize)
            self.poolsize = poolsize
            self.alive = True
            for _ in xrange(self.poolsize):
                worker = threading.Thread(target=self._worker_thread, args=tuple())
                worker.daemon = True
                worker.start()
                self.pool.append(worker)

        def shutdown(self):
            self.alive = False
            for _ in xrange(len(self.pool)):
                self.queue.put(None)

        def insert_trial_docs(self, docs):
            """ trials - something like is returned by self.new_trial_docs()
            """
            docs = [self.assert_valid_trial(base.SONify(doc))
                    for doc in docs]
            result = base.Trials._insert_trial_docs(self, docs)
            for doc in docs:
                self.queue.put(doc)
            return result

        def _handle_one_trial(self, trial):
            assert trial['state'] == base.JOB_STATE_NEW
            trial['state'] = base.JOB_STATE_RUNNING
            now = coarse_utcnow()
            trial['book_time'] = now
            trial['refresh_time'] = now
            spec = base.spec_from_misc(trial['misc'])
            ctrl = base.Ctrl(self.trials, current_trial=trial)
            try:
                result = self.domain.evaluate(spec, ctrl)
            except Exception as e:
                traceback.print_exc()
                trial['state'] = base.JOB_STATE_ERROR
                trial['misc']['error'] = (str(type(e)), str(e))
                trial['refresh_time'] = coarse_utcnow()
            else:
                trial['state'] = base.JOB_STATE_DONE
                trial['result'] = result
                trial['refresh_time'] = coarse_utcnow()

        def _worker_thread(self):
            print_exc = traceback.print_exc
            while self.alive:
                trial = self.queue.get()
                if trial is None:
                    break
                try:
                    self._handle_one_trial(trial)
                except BaseException:
                    print_exc()
                    sys.stderr.write('When handling trial: %r\n\n' % (trial, ))
                    break

    class FMinIter2(FMinIter):

        def __init__(self, algo, domain, trials, rstate, async=None,
                     max_queue_len=1,
                     poll_interval_secs=1.0,
                     max_evals=sys.maxsize,
                     verbose=0,
                     ):
            self.algo = algo
            self.domain = domain
            self.trials = trials
            if async is None:
                self.async = trials.async
            else:
                self.async = async
            self.poll_interval_secs = poll_interval_secs
            self.max_queue_len = max_queue_len
            self.max_evals = max_evals
            self.rstate = rstate

    if workers == 1:
        return base.Trials(), FMinIter2

    trials = MT_Trials(domain=domain, poolsize=workers)
    return trials, FMinIter2


def str_int(x):
    return str(int(x))


def vw_optimize_hyperopt(vw_filename, vw_validation_filename, vw_test_filename, y_true, kfold, args, metrics, config, sample_weight, workers, best_result, rounds, validation_holdout):
    assert isinstance(args, list), args
    from vwoptimizelib.third_party.hyperopt import hp, base

    space = {}
    format = {}

    unique_id = [0]

    def convert_to_hyperopt(param):

        name = param.opt + ' uid=%s' % unique_id[0]
        unique_id[0] += 1

        if isinstance(param, LogParam):
            assert param.min is not None and param.max is not None, 'For hyperopt, min and max needed for %s' % param.opt
            distrib = hp.loguniform(name, np.log(param.min), np.log(param.max))
        elif isinstance(param, RandIntParam):
            assert param.max is not None
            distrib = hp.randint(name, param.max)
        elif isinstance(param, FloatParam):
            assert param.min is not None and param.max is not None, 'For hyperopt, min and max needed for %s' % param.opt
            distrib = hp.uniform(name, param.min, param.max)
        elif isinstance(param, ValuesParam):
            distrib = hp.choice(name, param.values)
        elif isinstance(param, BinaryParam):
            distrib = hp.choice(name, ['', True])
        elif isinstance(param, IntegerParam):
            assert param.min is not None and param.max is not None, 'For hyperopt, min and max needed for %s' % param.opt
            distrib = hp.quniform(name, param.min, param.max, 1)
            format[param.opt] = str_int
        else:
            raise TypeError(param)

        if param.omit:
            distrib = hp.choice(name + '_outer', ['', distrib])

        return distrib

    gridparams = []
    base_args = []
    tunable_params = []
    hierarch_params = [x.replace('-', '') for x in (options.hyperopt_hierarchy or '').split(',')]
    tunable_params_dict = {}

    for param in args:
        if isinstance(param, BaseParam) and param.opt.replace('-', '') in hierarch_params:
            assert isinstance(param, (ValuesParam, BinaryParam))
            gridparams.append(param)
        elif 'all_categorical' in hierarch_params and isinstance(param, (ValuesParam, BinaryParam)):
            gridparams.append(param)
        elif isinstance(param, BaseParam):
            tunable_params.append(param)
            tunable_params_dict[param.opt] = param
        else:
            base_args.append(param)

    choices = []

    already_seen = set()

    for grid_param in expand(gridparams):
        assert isinstance(grid_param, list), grid_param
        grid_param = vw_normalize_params(grid_param)
        grid_param = ' '.join(grid_param)

        if grid_param in already_seen:
            # filter out things like "--ngram 1 --skips 1"
            continue

        already_seen.add(grid_param)

        local_space = {}
        for param in tunable_params:
            local_space[param.opt] = convert_to_hyperopt(param)

        if local_space:
            choices.append((grid_param, local_space))
        else:
            choices.append(grid_param)

    space = {'grid': hp.choice('grid', choices)}

    if best_result is None:
        best_result = {}

    best_result[MARKER_BRANCHBEST] = (float('inf'), None)

    def run(params):
        log('Parameters: %r', params, importance=-1)
        args = base_args[:]

        assert len(params) == 1, params
        params = params.pop('grid')
        assert len(params) == 2, params

        base, rest = params

        args.append(base)

        for key, value in sorted(rest.items()):
            if value == '' or value == 0 or value is None:
                args.append('')
                continue
            param_config = tunable_params_dict[key]
            # unpack() is only used by LogParam and hyperopt does exponentiation of its own
            param_config._unpack = None
            args.append(param_config.get_extra_args(value))

        result, is_best = run_single_iteration(
            vw_filename,
            vw_validation_filename,
            vw_test_filename,
            kfold,
            args,
            workers,
            metrics,
            y_true,
            sample_weight,
            config,
            best_result,
            with_predictions=False,
            validation_holdout=validation_holdout)

        if result is None:
            return float('inf')

        return result

    env_rseed = os.environ.get('HYPEROPT_FMIN_SEED', '')
    if env_rseed:
        rstate = np.random.RandomState(int(env_rseed))
    else:
        rstate = np.random.RandomState()

    domain = base.Domain(run, space, pass_expr_memo_ctrl=False)
    trials, FMinIter2 = setup_hyperopt_Trials(domain, workers)

    if '.' not in options.hyperopt_alg:
        options.hyperopt_alg = 'vwoptimizelib.third_party.hyperopt.%s.suggest' % options.hyperopt_alg
    alg = _import(options.hyperopt_alg)
    rval = FMinIter2(algo=alg, domain=domain, trials=trials, rstate=rstate, max_queue_len=workers or 1, poll_interval_secs=0.1)

    assert rounds is not None
    try:
        rval.run(rounds)
    finally:
        if hasattr(trials, 'shutdown'):
            trials.shutdown()

    return best_result[MARKER_BRANCHBEST]


class Simple1NN(object):

    def __init__(self):
        self.observations = []

    def add_observation(self, x, y):
        self.observations.append((x, y))

    def find_nearest(self, query_x):
        nearest_items = []
        nearest_distance = float('inf')
        for x, y in self.observations:
            dist = np.sum(np.abs(x - query_x))
            # print 'query=%s x=%s y=%s dist=%s' % (query_x, x, y, dist)
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_items = [y]
            elif dist == nearest_distance:
                nearest_items.append(y)
        return nearest_items


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
    params = params.replace('--skips 0', '')
    if '--ngram' not in params:
        params = re.sub('--skips \d+', '', params)
    params = re.sub('\s+', ' ', params)
    return params.split()


def expand(gridsearch_params, only=None, withextra=False):
    result = list(_expand(gridsearch_params, only=only))
    result.sort()
    if withextra:
        return result
    return [x[1] for x in result]


def _expand(gridsearch_params, only=None, score_mult=1):
    if not gridsearch_params:
        yield (0, [], [])
        return

    first_arg = gridsearch_params[0]

    if isinstance(first_arg, basestring):
        skip = True
    elif only is not None and getattr(first_arg, 'opt', '') not in only:
        skip = True
    else:
        skip = False

    if skip:
        for inner_score, inner_cmd, inner_vector in _expand(gridsearch_params[1:], only=only, score_mult=score_mult):
            yield (inner_score, _filter([first_arg] + inner_cmd), inner_vector)
        return

    for index, first_arg_variant in enumerate(first_arg.enumerate_all()):
        for inner_score, inner, inner_vector in _expand(gridsearch_params[1:], only=only, score_mult=score_mult * 1.01):
            new_inner_vector = [index] + inner_vector
            yield (index * score_mult + inner_score, _filter([first_arg_variant] + inner), new_inner_vector)


def _filter(lst):
    return [x for x in lst if x]


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


def chinese_simplify(unistr, cache={}):
    u"""
    This function does the same as hanziconv's module toSimplified() but order of magnitude faster

    >>> print chinese_simplify(u'繁簡轉換器')
    繁简转换器

    >>> from hanziconv import HanziConv
    >>> from hanziconv.charmap import traditional_charmap
    >>> HanziConv.toSimplified(traditional_charmap) == chinese_simplify(traditional_charmap)
    True

    >>> import timeit
    >>> timeit.timeit(lambda : chinese_simplify(traditional_charmap), number=1000) # doctest:+SKIP
    0.1961040496826172

    >>> toSimplified = HanziConv.toSimplified
    >>> timeit.timeit(lambda : toSimplified(traditional_charmap), number=1000) # doctest:+SKIP
    4.9171209335327150
    """
    table = cache.get('table')
    if table is None:
        from hanziconv.charmap import traditional_charmap, simplified_charmap
        table = dict((ord(char1), char2) for char1, char2 in izip(reversed(traditional_charmap), reversed(simplified_charmap)))
        cache['table'] = table
    return unistr.translate(table)


def get_regex(range_name, cache={}):
    result = cache.get(range_name)
    if result is not None:
        return result
    result = re.compile(_generate_regex(range_name, RANGES[range_name]))
    cache[range_name] = result
    return result


def _generate_regex(name, range):
    result = []
    ignored = 0
    included = 0
    for item in range:
        try:
            count = 0
            if len(item) == 1:
                count = 1
                result.append(unichr(item[0]))
            else:
                count = item[1] - item[0] + 1
                result.append(unichr(item[0]) + '-' + unichr(item[1]))
        except ValueError, ex:
            if 'unichr() arg not in range' in str(ex):
                ignored += count
            else:
                raise
        else:
            included += count

    if ignored:
        log_always("%s: Ignored %s characters (left with %s)", name, ignored, included)

    if not included:
        sys.exit('%s: empty range' % name)

    return u'[' + u''.join(result) + u']'


# ideograph total chars=75640
# hangul total chars=11735
# hiragana total chars=97
# katakana total chars=223
# combined total chars=87688
RANGES = {
    'combined': [[4352, 4607], [12272, 12283], [12288, 12290], [12293, 12295], [12330, 12335], [12343], [12347], [12350, 12351], [12353, 12438], [12441, 12543], [12593, 12686], [12688, 12703], [12784, 12828], [12832, 12871], [12896, 12923], [12926], [12928, 12976], [12992, 13003], [13008, 13054], [13144, 13168], [13280, 13310], [13312, 19893], [19968, 40907], [43360, 43388], [44032, 55203], [55216, 55238], [55243, 55291], [63744, 64045], [64048, 64109], [64112, 64217], [65041, 65042], [65105], [65377], [65380, 65470], [65474, 65479], [65482, 65487], [65490, 65495], [65498, 65500], [127488], [127504, 127537], [127552, 127560], [131072, 173782], [173824, 177972], [194560, 195101]],
    'hangul': [[4352, 4607], [12334, 12335], [12593, 12686], [12800, 12828], [12896, 12923], [12926], [43360, 43388], [44032, 55203], [55216, 55238], [55243, 55291], [65440, 65470], [65474, 65479], [65482, 65487], [65490, 65495], [65498, 65500]],
    'hiragana': [[12353, 12438], [12441, 12448], [12540], [65392], [127488]],
    'ideographs': [[12272, 12283], [12288, 12290], [12293, 12295], [12330, 12333], [12343], [12347], [12350, 12351], [12688, 12703], [12832, 12871], [12928, 12976], [12992, 13003], [13144, 13168], [13280, 13310], [13312, 19893], [19968, 40907], [63744, 64045], [64048, 64109], [64112, 64217], [65041, 65042], [65105], [65377], [65380], [127504, 127506], [127508, 127537], [127552, 127560], [131072, 173782], [173824, 177972], [194560, 195101]],
    'katakana': [[12441, 12444], [12448, 12543], [12784, 12799], [13008, 13054], [65381, 65439], [127507]]
}


class Preprocessor(object):
    ur"""
    >>> Preprocessor(split_ideographs=True, chinese_simplify=True).process_text(u'hello 繁簡轉換器'.encode('utf8'))
    'hello \xe7\xb9\x81 \xe7\xae\x80 \xe8\xbd\xac \xe6\x8d\xa2 \xe5\x99\xa8'
    """

    ALL_OPTIONS_BINARY = '''
        htmlunescape
        lowercase
        strip_punct
        stem
        split_chars
        split_ideographs
        split_hangul
        split_hiragana
        split_katakana
        split_combined
        chinese_simplify
        NFKC
        remove_duplicate_words
    '''.strip().split()

    ALL_OPTIONS_INT = '''
        max_words
        max_length
        max_length_offset
        max_word_size
    '''.strip().split()

    ALL_OPTIONS = ALL_OPTIONS_BINARY + ALL_OPTIONS_INT

    ALL_OPTIONS_DASHDASH = ['--%s' % x for x in ALL_OPTIONS]

    @classmethod
    def init_option_parser(cls, parser):
        for opt in cls.ALL_OPTIONS_BINARY:
            parser.add_option('--%s' % opt, action='store_true')
        for opt in cls.ALL_OPTIONS_INT:
            parser.add_option('--%s' % opt, type=int)

    @classmethod
    def parse_options(cls, string):
        parser = PassThroughOptionParser()
        cls.init_option_parser(parser)
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
            if options[opt] is not None:
                break
        else:
            return None

        return cls(**options)

    def to_options(self):
        result = ['--%s' % opt for opt in self.ALL_OPTIONS_BINARY if getattr(self, opt, None)]
        result += ['--%s %s' % (opt, getattr(self, opt)) for opt in self.ALL_OPTIONS_INT if getattr(self, opt, None)]
        return result

    def __init__(self, **kwargs):
        for option in self.ALL_OPTIONS_BINARY:
            setattr(self, option, kwargs.get(option, False))

        for option in self.ALL_OPTIONS_INT:
            value = kwargs.get(option)
            if value is not None:
                value = int(value)
            setattr(self, option, value)

        if self.stem:
            stem_words(["testing"])
            self.lowercase = True
            self.strip_punct = True

        if self.split_chars or self.split_combined:
            self.split_ideographs = False
            self.split_hangul = False
            self.split_hiragana = False
            self.split_katakana = False

        if self.split_chars:
            self.split_combined = False

        for range in RANGES:
            if getattr(self, 'split_%s' % range):
                setattr(self, 'split_%s' % range, get_regex(range))

    def __str__(self):
        return ' '.join(self.to_options())

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, ', '.join('%s=%r' % (name, getattr(self, name, None)) for name in self.ALL_OPTIONS))

    def process_text(self, text):
        orig = text
        try:
            text = text.decode('utf-8', errors='ignore')

            if self.max_length_offset is not None:
                text = text[self.max_length_offset:]

            if self.max_length is not None:
                text = text[:self.max_length]

            # quite costly
            # if self.normalize_space:
            #     text = u''.join(u' ' if unicodedata.category(x)[:1] in 'CZ' else x for x in text)

            if self.htmlunescape:
                text = htmlparser_unescape(text)

            if self.NFKC:
                text = unicodedata.normalize('NFKC', text)

            if self.lowercase:
                text = text.lower()

            if self.chinese_simplify:
                text = chinese_simplify(text)

            if self.strip_punct:
                words = re.findall(r"(?u)\b\w\w+\b", text)
            else:
                words = text.split()

            if self.max_words is not None:
                words = words[:self.max_words]

            if self.max_word_size is not None:
                words = [x[:self.max_word_size] for x in words]

            if self.stem:
                words = stem_words(words)

            if self.remove_duplicate_words:
                seen = set()
                new_words = []
                for word in words:
                    if word not in seen:
                        new_words.append(word)
                        seen.add(word)
                words = new_words

            if self.split_chars:
                words = [' '.join(w) for w in words]
                text = u' __ '.join(words)
            else:
                text = u' '.join(words)
                if self.split_combined:
                    text = self.split_combined.sub(ur" \g<0> ", text)
                else:
                    if self.split_ideographs:
                        text = self.split_ideographs.sub(ur" \g<0> ", text)

                    if self.split_hiragana:
                        text = self.split_hiragana.sub(ur" \g<0> ", text)

                    if self.split_katakana:
                        text = self.split_katakana.sub(ur" \g<0> ", text)

                    if self.split_hangul:
                        text = self.split_hangul.sub(ur" \g<0> ", text)

                    text = re.sub(r'\s+', ' ', text.strip())

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


def read_y_true(filename, format, columnspec, ignoreheader, named_labels, remap_label, examples=None):
    log('Reading labels from %s', filename or 'stdin')
    if format == 'vw':
        return _load_predictions(filename, named_labels=named_labels, with_weights=True, examples=examples)

    rows_source = open_anything(filename, format, ignoreheader=ignoreheader)

    label_index = columnspec.index('y')

    weight_index = None
    try:
        weight_index = columnspec.index('weight_metric')
    except ValueError:
        try:
            weight_index = columnspec.index('weight')
        except ValueError:
            pass

    y_true = []
    weights = []

    for row in rows_source:
        label = row[label_index]
        if remap_label is not None:
            label = remap_label.get(label, label)
        if named_labels is None:
            label = float(label)
        elif label not in named_labels:
            sys.exit('Unexpected label in %s: %r (allowed: %s)' % (filename, label, named_labels))
        y_true.append(label)

        if weight_index is not None:
            w = float(row[weight_index].strip() or '1')
            weights.append(w)

        if examples is not None and len(y_true) >= examples:
            break

    y_true = np.array(y_true)
    if weights:
        weights = np.array(weights)

    return y_true, (weights if weight_index is not None else None)


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


class ParseError(ValueError):
    pass


def parse_mapping(config):
    """
    >>> parse_mapping('1:-1,2:1')
    {'1': '-1', '2': '1'}

    >>> parse_mapping('1:-1,2:1'.split(','))
    {'1': '-1', '2': '1'}
    """
    if not config:
        return None

    config = _make_proper_list(config)

    if not config:
        return None

    result = {}

    for item in config:
        if ':' not in item:
            raise ParseError(item)
        key, value = item.rsplit(':', 1)

        if key in result:
            log_always('Label %r specified more than once', key)

        result[key] = value

    return result


def parse_weight(config, named_labels=None):
    """
    >>> parse_weight('A:B:2', ['A:B', 'another_label'])
    {'A:B': '2'}

    >>> parse_weight('A:B:2')
    Traceback (most recent call last):
     ...
    SystemExit: Weight must be specified as CLASS(float):WEIGHT, 'A:B' not recognized
    """
    if named_labels is not None and not isinstance(named_labels, list):
        raise TypeError('must be list, not %r' % type(named_labels))

    try:
        config = parse_mapping(config)
    except ParseError, item:
        sys.exit('Weight must be specified as CLASS:WEIGHT, cannot parse %s' % item)

    if not config:
        return config

    result = {}

    for label, weight in config.items():
        if named_labels is None:
            try:
                float(label)
            except Exception:
                sys.exit('Weight must be specified as CLASS(float):WEIGHT, %r not recognized' % (label, ))
        else:
            if label not in named_labels:
                sys.exit('Label %r is not recognized. Expected: %r' % (label, named_labels))

        try:
            float(weight)
        except Exception:
            weight = None

        if weight is None or weight < 0:
            sys.exit('Weight must be specified as CLASS:WEIGHT(float), %r is not recognized' % (item, ))

        result[label] = weight

    return result


def get_sample_weight(y_true, config):
    if config is None:
        return None
    N = len(y_true)
    result = np.zeros(N)
    updated = np.zeros(N)

    for klass, weight in config.items():
        klass = float(klass)
        weight = float(weight)
        result += np.multiply(np.ones(N) * weight, y_true == klass)
        updated += y_true == klass

    result += (updated == 0)

    return result


def process_text(preprocessor, text):
    if preprocessor is not None:
        text = preprocessor.process_text(text)
    else:
        text = re.sub('\s+', ' ', text)
    text = text.replace(':', ' ').replace('|', ' ')
    text = text.strip()
    return text


def convert_row_to_vw(row, columnspec, preprocessor, weights, named_labels, remap_label):
    if isinstance(row, basestring):
        if not row.strip():
            return row
        assert '|' in row, row
        assert columnspec is None, '--columnspec not supported with .vw format'
        if preprocessor is None and not weights and not remap_label:
            return row

        items = re.split(r'([^\s]*\|[^\s]*)', row)
        label = items[0]

        if preprocessor is None:
            rest = ''.join(items[1:])
        else:
            if len(items) <= 2:
                sys.exit('Cannot parse: %r' % (row, ))
            processed = []
            for item in items[1:]:
                if '|' in item:
                    processed.append(item)
                else:
                    processed.append(' ' + preprocessor.process_text(item) + ' ')
            rest = ''.join(processed)

        if remap_label is not None:
            new_label = remap_label.get(label.strip())
            if new_label is not None:
                label = new_label + ' '

        if weights:
            label_items = label.split(' ', 2)
            y = label_items[0]

            if named_labels is not None and y not in named_labels:
                sys.exit('Label not recognized: %r' % (row, ))

            class_weight = weights.get(y, 1)
            if class_weight is None or float(class_weight) == 1.0:
                # don't need to update label/weight part
                pass
            else:
                weight_token = label_items[1] if len(label_items) >= 2 else None

                if not weight_token or not weight_token.strip() or weight_token.startswith("'") or weight_token.startswith("|"):
                    example_weight = 1
                    rest_label = ' '.join(label_items[1:])
                else:
                    example_weight = float(weight_token)
                    rest_label = ' '.join(label_items[2:])

                final_weight = example_weight * float(class_weight)

                if final_weight == 1:
                    label = y + ' ' + rest_label
                else:
                    label = y + ' ' + str(final_weight) + ' ' + rest_label

        return label + rest.rstrip() + '\n'

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
    last_namespace = None
    example_weight = None
    example_weight1 = None

    for item, spec in zip(row, columnspec):
        if spec == 'y':
            y = item
        elif spec == 'text' or spec.startswith('text_'):
            namespace = spec[5:]
            if not x or namespace != last_namespace:
                x.append('|' + namespace)
            x.append(process_text(preprocessor, item))
            if '|' in item:
                last_namespace = None
            else:
                last_namespace = namespace
        elif spec == 'vw' or spec.startswith('vw_'):
            namespace = spec[3:]
            if not item.startswith('|') and (not x or namespace != last_namespace):
                x.append('|' + namespace)
            x.append(item)
            if '|' in item:
                last_namespace = None
            else:
                last_namespace = namespace
        elif spec == 'info':
            info.append(item)
        elif spec == 'drop' or not spec:
            continue
        elif spec == 'weight':
            example_weight = item
        elif spec == 'weight_train':
            example_weight1 = item
        elif spec == 'weight_metric':
            pass  # used by read_y_true
        else:
            sys.exit('Spec item %r not understood' % spec)

    example_weight = example_weight1 or example_weight

    if info:
        info = " '%s" % ';'.join(info) + ' '
    else:
        info = ''

    if named_labels is not None and y not in named_labels:
        sys.exit('Label not recognized: %r' % (row, ))

    if remap_label is not None:
        y = remap_label.get(y, y)

    class_weight = weights.get(y) if weights is not None else None

    if example_weight is not None and class_weight is not None:
        weight = float(example_weight) * float(class_weight)
    elif example_weight is None:
        weight = class_weight
    else:
        weight = example_weight

    if weight is None:
        weight = ''
    else:
        weight = ' ' + str(weight).strip()

    text = y + weight + info + ' ' + ' '.join(x) + '\n'
    return text


def _convert_any_to_vw(source, format, output, weights, preprocessor, columnspec, named_labels, remap_label, ignoreheader):
    if named_labels is not None:
        assert not isinstance(named_labels, basestring)
        named_labels = set(named_labels)

    rows_source = open_anything(source, format, ignoreheader=ignoreheader)
    output = open(output, 'wb')

    for row in rows_source:
        try:
            vw_line = convert_row_to_vw(row, columnspec, preprocessor=preprocessor, weights=weights, named_labels=named_labels, remap_label=remap_label)
        except Exception:
            log_always('Failed to parse: %r', row)
            raise
        output.write(vw_line)

    flush_and_close(output)


def convert_any_to_vw(source, format, output_filename, columnspec, named_labels, remap_label, weights, preprocessor, ignoreheader, workers):
    preprocessor = preprocessor or ''

    assert isinstance(preprocessor, basestring), preprocessor

    log('preprocessor = %s', preprocessor or '', importance=1 if preprocessor else 0)

    start = time.time()

    if source is None:
        from cStringIO import StringIO
        source = StringIO(sys.stdin.read())

    workers = _workers(workers)
    # XXX do os.stat on the source and decide on number of workers based on file size (e.g. less than 50k per worker does not make much sense)
    batches, total_lines = split_file(source, nfolds=workers, ignoreheader=ignoreheader, importance=-1)

    batches_out = [x + '.out' for x in batches]

    try:
        commands = []

        common_cmd = [quote(sys.executable), quote(__file__), '--format', format]

        if TMP_PREFIX:
            common_cmd += ['--tmp', TMP_PREFIX]

        if named_labels:
            common_cmd += ['--named_labels', ','.join(named_labels)]

        if remap_label:
            common_cmd += ['--remap_label', ','.join('%s:%s' % item for item in remap_label.items())]

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

        success, outputs = run_subprocesses(commands, workers=workers, importance=-1)
        if not success:
            sys.exit(1)

        unlink(*batches)

        cmd = 'cat ' + ' '.join(batches_out)
        if output_filename:
            cmd += ' > %s' % output_filename

        system(cmd, importance=-1)

    finally:
        unlink(*batches)
        unlink(*batches_out)

    took = time.time() - start
    log('Generated %s in %.1f seconds', output_filename, took)
    if not output_filename.startswith('/dev/'):
        log('\n'.join(open(output_filename).read(200).split('\n')) + '...')


def _import(path):
    _NONE = object()
    if isinstance(path, list):
        if not path:
            raise ImportError('Cannot import from empty list: %r' % (path, ))
        for item in path[:-1]:
            try:
                return _import(item)
            except ImportError:
                pass
        return _import(path[-1])
    if not isinstance(path, basestring):
        return path
    if '.' not in path:
        raise ImportError("Cannot import %r (required format: [path/][package.]module.class)" % path)
    if '/' in path:
        package_path, path = path.rsplit('/', 1)
        sys.path = [package_path] + sys.path
    else:
        package_path = None
    try:
        module, item = path.rsplit('.', 1)
        x = __import__(module)
        for attr in path.split('.')[1:]:
            oldx = x
            x = getattr(x, attr, _NONE)
            if x is _NONE:
                raise ImportError('Cannot import %r from %r' % (attr, oldx))
        return x
    finally:
        try:
            sys.path.remove(package_path)
        except ValueError:
            pass


metrics_shortcuts = {
    'mse': 'mean_squared_error',
    'rmse': 'root_mean_squared_error',
    'mae': 'mean_absolute_error',
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
    'mean_absolute_error': 'y_score',
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
    'recall_at_precision': 'y_score',
    'count_pos': 'y_pred',
    'kendall_tau': 'y_score',
    'count': 'y_score',
    'tp': 'y_pred',
    'fp': 'y_pred',
    'tn': 'y_pred',
    'fn': 'y_pred',
}


def root_mean_squared_error(*args, **kwargs):
    import sklearn.metrics
    return math.sqrt(sklearn.metrics.mean_squared_error(*args, **kwargs))


def is_loss(metric_name):
    metric_name = metrics_shortcuts.get(metric_name, metric_name)
    if 'loss' in metric_name or metric_name.endswith('_error'):
        return True


def calculate_or_extract_score(metric, y_true, y_pred, config, outputs, sample_weight, num_features=None):
    if metric == 'num_features':
        return num_features
    try:
        if metric.startswith('vw'):
            return extract_score(metric, outputs)
        return calculate_score(metric, y_true, y_pred, config, sample_weight)
    except Exception, ex:
        if MINIMUM_LOG_IMPORTANCE <= 0:
            traceback.print_stack()
            traceback.print_exc()
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
    if not outputs:
        raise ValueError('error: No output captured from vw')

    orig_outputs = outputs

    stage, metric = _parse_vw_metric(metric)
    outputs = (outputs or {}).get(stage)

    if not outputs:
        raise ValueError('error: No output for stage %r. Available: %r' % (stage, ', '.join(orig_outputs.keys())))

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


def recall_at_precision(*args, **kwargs):
    from sklearn.metrics import precision_recall_curve
    metric_param = kwargs.pop('metric_param')
    required_precision = _parse_number_or_fraction(metric_param)
    precision, recall, thresholds = precision_recall_curve(*args, **kwargs)

    for pr, r in izip(precision, recall):
        if pr >= required_precision:
            return r


def count_pos(y_true, y_pred, sample_weight=None):
    assert sample_weight is None
    return sum(y_true)


def kendall_tau(y_true, y_score):
    from scipy.stats import kendalltau
    ret_score = kendalltau(y_true, y_score)[0]
    return ret_score if not np.isnan(ret_score) else 0.0


def count(y_true, y_pred, sample_weight=None):
    return len(y_pred)


def tp(y_true, y_pred, sample_weight=None):
    result = y_true == y_pred
    result = np.multiply(result, y_true > 0)
    if sample_weight is not None:
        result = np.multiply(result, sample_weight)
    return sum(result)


def fp(y_true, y_pred, sample_weight=None):
    result = y_true != y_pred
    result = np.multiply(result, ~y_true)
    if sample_weight is not None:
        result = np.multiply(result, sample_weight)
    return sum(result)


def tn(y_true, y_pred, sample_weight=None):
    result = y_true == y_pred
    result = np.multiply(result, ~y_true)
    if sample_weight is not None:
        result = np.multiply(result, sample_weight)
    return sum(result)


def fn(y_true, y_pred, sample_weight=None):
    result = y_true != y_pred
    result = np.multiply(result, y_true)
    if sample_weight is not None:
        result = np.multiply(result, sample_weight)
    return sum(result)


def calculate_score(metric, y_true, y_pred, config, sample_weight, logged_thresholds=set([0, 0.5])):
    sample_weight_from_class_config = get_sample_weight(y_true, config.get('weight_metric'))
    if sample_weight is None:
        sample_weight = sample_weight_from_class_config
    else:
        assert len(sample_weight) == len(y_true), 'sample_weight len=%s y_true len=%s' % (len(sample_weight), len(y_true))
        if sample_weight_from_class_config is not None:
            sample_weight = np.multiply(sample_weight, sample_weight_from_class_config)

    if metric == 'weight':
        if sample_weight is not None:
            return sum(sample_weight)
        return len(y_pred)

    threshold = config.get('threshold')
    min_label = config.get('min_label')
    max_label = config.get('max_label')

    extra_args = {}

    if '=' in metric:
        assert metric.count('=') == 1, metric
        metric, metric_param = metric.split('=')
        extra_args['metric_param'] = metric_param

    if metric.endswith('_offset'):
        assert 'metric_param' in extra_args
        offset = int(extra_args.pop('metric_param'))
        metric = metric[:-len('_offset')]
        if sample_weight is not None:
            sample_weight = sample_weight[offset:]
        y_true = y_true[offset:]
        y_pred = y_pred[offset:]

    extra_args['sample_weight'] = sample_weight

    fullname = metrics_shortcuts.get(metric, metric)

    if fullname in ('precision_score', 'recall_score', 'f1_score'):
        extra_args['average'] = 'binary'

    if '.' in fullname:
        if ':' in fullname:
            metric_type, fullname = fullname.split(':')
        else:
            metric_type = 'y_score'
        func = _import(fullname)
    elif fullname in globals():
        metric_type = metrics_param[fullname]
        func = globals()[fullname]
    else:
        import sklearn.metrics
        func = getattr(sklearn.metrics, fullname, None)
        if func is None:
            sys.exit('Cannot find %r in sklearn.metrics' % (fullname, ))
        metric_type = metrics_param[fullname]

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
            if threshold not in logged_thresholds:
                log('threshold = %.3f', threshold, importance=1)
                logged_thresholds.add(threshold)
            y_true = y_true > threshold
            y_pred = y_pred > threshold
        return func(y_true, y_pred, **extra_args)
    else:
        raise ValueError('Unknown metric_type: %r (must be y_score, y_pred or y_prob)' % metric_type)


def _log_classification_report(prefix, *args, **kwargs):
    result = classification_report(*args, **kwargs)
    maxwidth = {}

    for line in result:
        for column, item in enumerate(line):
            maxwidth[column] = max(maxwidth.get(column, 0), len(item))

    for line in result:
        if prefix:
            sys.stderr.write(prefix)
        for column, item in enumerate(line):
            frmt = '%' + str(maxwidth[column]) + 's '
            sys.stderr.write(frmt % item)
        sys.stderr.write('\n')


def log_classification_report(*args, **kwargs):
    try:
        _log_classification_report(*args, **kwargs)
    except Exception, ex:
        sys.stderr.write(str(ex) + '\n')


def classification_report(y_true, y_pred, labels=None, sample_weight=None, digits=4, threshold=None):
    # this function is copied from https://github.com/scikit-learn/scikit-learn/blob/412996f/sklearn/metrics/classification.py#L1341 (c) respective authors
    # I pulled it here to fix formatting bug.
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if labels is None:
        from sklearn.utils.multiclass import unique_labels

        if threshold is not None:
            y_true = y_true > threshold
            y_pred = y_pred > threshold

        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    last_line_heading = 'avg / total'
    target_names = ['%s' % l for l in labels]

    results = [["", "precision", "recall", "f1-score", "support", "accuracy"]]

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average=None,
                                                  sample_weight=sample_weight)

    for i, label in enumerate(labels):
        values = [target_names[i]]
        for v in (p[i], r[i], f1[i]):
            values += ["{0:0.{1}f}".format(v, digits)]
        values += ["{0}".format(s[i])]
        accuracy = accuracy_score(y_true == label, y_pred == label, sample_weight=sample_weight)
        values += ["{0:0.{1}f}".format(accuracy, digits)]
        results.append(values)

    values = [last_line_heading]
    for v in (np.average(p, weights=s),
              np.average(r, weights=s),
              np.average(f1, weights=s)):
        values += ["{0:0.{1}f}".format(v, digits)]
    values += ['{0}'.format(np.sum(s))]
    accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    values += ["{0:0.{1}f}".format(accuracy, digits)]
    results.append(values)

    return results


def _id(x):
    return x


def main_tune(metric, config, filename, validation, test, validation_holdout, format, y_true, sample_weight, args, preprocessor_base, kfold, ignoreheader, workers, hyperopt_rounds):
    if preprocessor_base is None:
        preprocessor_base = []
    else:
        preprocessor_base = preprocessor_base.to_options()

    if not metric:
        metric = DEFAULT_METRICS

    optimization_metric = metric[0]

    best_preprocessor_opts = None
    best_vw_options = None
    best_result_so_far = None
    already_done = {}

    preprocessor_variants = expand(args, only=Preprocessor.ALL_OPTIONS_DASHDASH)

    log('Trying preprocessor variants: %s', pprint.pformat(preprocessor_variants), importance=-1)

    best_result = {}
    if len(preprocessor_variants) > 1:
        best_result[MARKER_BEST] = (float('inf'), None)

    if not is_loss(optimization_metric):
        do_abs = abs
    else:
        do_abs = _id

    for my_args in preprocessor_variants:
        preprocessor = Preprocessor.from_options(preprocessor_base + my_args)
        preprocessor_opts = ' '.join(preprocessor.to_options() if preprocessor else [])

        previously_done = already_done.get(str(preprocessor))

        if previously_done:
            log('Same as %s', previously_done)
            continue

        already_done[str(preprocessor)] = preprocessor_opts

        to_cleanup = []

        try:
            weight_train = config.get('weight_train')

            if format == 'vw' and not weight_train and not preprocessor:
                vw_filename = filename
                vw_validation_filename = validation
                vw_test_filename = test
            else:
                vw_filename = get_temp_filename('vw_filename')
                to_cleanup.append(vw_filename)

                convert_args = dict(
                    format=format,
                    columnspec=config.get('columnspec'),
                    named_labels=config.get('named_labels'),
                    remap_label=config.get('remap_label'),
                    weights=weight_train,
                    preprocessor=preprocessor_opts,
                    ignoreheader=ignoreheader,
                    workers=workers)

                convert_any_to_vw(source=filename, output_filename=vw_filename, **convert_args)

                if validation:
                    vw_validation_filename = get_temp_filename('vw_validation')
                    to_cleanup.append(vw_validation_filename)
                    convert_any_to_vw(source=validation, output_filename=vw_validation_filename, **convert_args)
                else:
                    vw_validation_filename = None

                if test:
                    vw_test_filename = get_temp_filename('vw_test')
                    to_cleanup.append(vw_test_filename)
                    convert_any_to_vw(source=test, output_filename=vw_test_filename, **convert_args)
                else:
                    vw_test_filename = None

            vw_args = [x for x in my_args if getattr(x, 'opt', None) or str(x).split()[0] not in Preprocessor.ALL_OPTIONS_DASHDASH]

            if hyperopt_rounds:
                opt = vw_optimize_hyperopt
                extra = {
                    'rounds': hyperopt_rounds,
                }
            else:
                opt = vw_optimize
                extra = {}

            this_best_result, this_best_options = opt(
                vw_filename,
                vw_validation_filename,
                vw_test_filename,
                y_true,
                kfold,
                vw_args,
                metric,
                config,
                sample_weight=sample_weight,
                workers=workers,
                best_result=best_result,
                validation_holdout=validation_holdout,
                **extra)
        finally:
            unlink(*to_cleanup)

        is_best = ''
        if this_best_result is not None and (best_result_so_far is None or best_result_so_far > this_best_result):
            best_result_so_far = this_best_result
            best_vw_options = this_best_options
            best_preprocessor_opts = preprocessor_opts
            is_best = '*'

        if len(preprocessor_variants) > 1:
            if preprocessor_opts:
                log_always('Best options with %s = %s', preprocessor_opts or 'no preprocessing', this_best_options)
            log_always('Best %s with %r = %s%s', optimization_metric, preprocessor_opts or 'no preprocessing', _frmt_score(do_abs(this_best_result)), is_best)
        # print 'Improvement over no l1=%.4f. Improvement over initial guess=%.4f' % (no_l1_result - best_result[0], initial_l1_result - best_result[0])

    # XXX don't show this if preprocessor is not enabled and not tuned
    if len(preprocessor_variants) > 1:
        log_always('Best preprocessor options = %s', best_preprocessor_opts or '<none>')
    log_always('Best vw options = %s', best_vw_options)
    log_always('Best %s = %s', optimization_metric, _frmt_score(do_abs(best_result_so_far)))
    # print 'Improvement over no l1=%.4f. Improvement over initial guess=%.4f' % (no_l1_result - best_result[0], initial_l1_result - best_result[0])
    preprocessor = Preprocessor.from_options(best_preprocessor_opts)
    return best_vw_options, preprocessor


def format_item(counts, weight, hash):
    top_items = [(v, k) for (k, v) in counts[hash].items()]
    count = len(top_items)
    if not top_items:
        return None, count
    top_items.sort(reverse=True)
    top_items = ', '.join('%s %s' % (k, v) for (v, k) in top_items)
    return '%g %s' % (weight, top_items), count


def parseaudit(source, includezeros=False, oaa=None, top=None, bottom=None):
    weights_per_class = {}  # class -> hash -> weight
    counts_per_class = {}  # class -> hash -> text -> count
    line = None
    top = top or 0
    bottom = bottom or 0

    # TODO: for binary labels, count positive vs negative class

    current_class = 1

    while True:
        line = source.readline()
        if not line:
            break
        if not line.startswith('\t'):
            continue
        line = line.rstrip()

        example_features = {}

        for feature in line.strip().split():
            text, hash, _value, weight = feature.split(':')[:4]

            if hash in example_features:
                # Count the feature only once. This ignores collisions within the example (which is better than counting a particular hash twice).
                continue

            weight = weight.split('@')[0]
            weight = float(weight)

            if not weight and not includezeros:
                continue

            example_features[hash] = text
            weights_per_class.setdefault(current_class, {})[hash] = weight

        for hash, feature in example_features.iteritems():
            counts = counts_per_class.setdefault(current_class, {})
            c = counts.get(hash)
            if c is None:
                counts[hash] = {feature: 1}
            else:
                c[feature] = c.get(feature, 0) + 1

        if oaa is not None:
            current_class += 1
            if current_class > oaa:
                current_class = 1

    total_count = 0
    for klass in sorted(weights_per_class.keys()):
        if oaa is not None:
            print "\nclass: %s" % klass
        weights = weights_per_class[klass]
        weights = [(w, hash) for (hash, w) in weights.iteritems()]
        weights.sort(reverse=True)

        printing = True

        for index, (w, hash) in enumerate(weights):
            item, count = format_item(counts_per_class[klass], w, hash)
            total_count += count

            if top or bottom:
                if index >= top and index < len(weights) - bottom:
                    if index == top:
                        print '...'
                    continue

            if printing and item:
                try:
                    print item
                except IOError:
                    # likely because we're being piped into head or tail
                    printing = False
                    # not aborting loop so that total_count is good

    log("Unique%s features: %s", '' if includezeros else ' non-zero', total_count, importance=1)
    if oaa is not None:
        log("Unique%s features per class: %g", '' if includezeros else ' non-zero', total_count / float(oaa), importance=1)


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
        # %g would use scientific notation for big numbers
        # %f alone would add trailing zeros
        x = '%f' % x
        if '.' in x:
            x = x.rstrip('0').rstrip('.')
        return x + suffix
    if x is None:
        return 'nan'
    return str(x)


def read_argument(args, name, type=None):
    assert isinstance(args, list), args
    for item in args:
        if name is None:
            if type is not None:
                item = type(item)
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

    for yp, yt, example in zip(y_pred, y_true, open_anything(filename, format, ignoreheader=ignoreheader)):
        # add hash of the example as a second item so that we get a mix of false positives and false negatives for a given error level
        try:
            err = abs(yp - yt)
        except TypeError:
            # XXX for multiclass, fetch raw scores
            err = 1 if yp != yt else 0
        errors.append((err, hash(repr(example)), yp, example))

    errors.sort(reverse=True)

    if '.' in toperrors:
        min_error = float(toperrors)
        errors = [x for x in errors if x[0] >= min_error]
    else:
        count = int(toperrors)
        errors = errors[:count]

    output = csv.writer(sys.stdout)

    for err, _hash, yp, example in errors:
        row = [str(yp)]
        if isinstance(example, list):
            row.extend(example)
        else:
            row.append(str(example))
        output.writerow(row)


def print_top_differences(topdiff, y_true, y_pred, y_pred_text, y_pred2, y_pred_text2, filename, format, ignoreheader):
    assert y_true is not None
    assert y_pred is not None
    assert y_pred2 is not None
    assert filename is not None
    assert len(y_true) == len(y_pred), (len(y_true), len(y_pred))
    assert len(y_true) == len(y_pred2), (len(y_true), len(y_pred2))

    differences = []

    for yp, yp_text, yp2, yp_text2, yt, example in zip(y_pred, y_pred_text, y_pred2, y_pred_text2, y_true, open_anything(filename, format, ignoreheader=ignoreheader)):
        diff = abs(yp - yp2)
        # XXX for multiclass, fetch raw scores
        if yp2 * yp > 0:
            continue
        differences.append((diff, yp2 * yp < 0, hash(repr(example)), yp_text.strip(), yp_text2.strip(), example))

    differences.sort(reverse=True)

    if '.' in topdiff:
        min_diff = float(topdiff)
        differences = [x for x in differences if x[0] >= min_diff]
    else:
        count = int(topdiff)
        differences = differences[:count]

    output = csv.writer(sys.stdout)

    for _diff, _diffsign, _hash, yp_text, yp_text2, example in differences:
        row = [yp_text, yp_text2]
        if isinstance(example, list):
            row.extend(example)
        else:
            row.append(str(example).strip())
        output.writerow(row)


def cleanup_vw_train_options(vw_args):
    vw_args = vw_args.split()
    remove_option(vw_args, '--quiet', 0)
    remove_option(vw_args, '--progress', 1)
    remove_option(vw_args, '-P', 1)
    remove_option(vw_args, '--threads', 0)
    return ' '.join(vw_args)


def get_breakdown_group(breakdown_re, item):
    item = item.split(' ', 1)
    if len(item) >= 2:
        item = item[-1].strip()
    else:
        item = ''
    m = breakdown_re.search(item)
    if m is None:
        return 'nomatch'
    else:
        group = m.groups()
        if group == ():
            group = m.group(0)
        else:
            group = ','.join(group)
        return group


def log_report_one(prefix, metrics, y_true, y_pred, sample_weight, config, classification_report, outputs=None, mask=None, num_features=None):

    if mask is not None:
        y_true = np.ma.MaskedArray(y_true, mask=mask).compressed()
        y_pred = np.ma.MaskedArray(y_pred, mask=mask).compressed()
        sample_weight = np.ma.MaskedArray(sample_weight, mask=mask).compressed() if sample_weight is not None else None
        assert y_true.shape == y_pred.shape, (y_true.shape, y_pred.shape)

    for metric in metrics:
        log_always('%s%s = %s', prefix, metric, _frmt_score(calculate_or_extract_score(metric, y_true, y_pred, config, outputs=outputs, sample_weight=sample_weight, num_features=num_features)))

    if classification_report:
        assert y_true is not None
        assert y_pred is not None
        log_classification_report(prefix, y_true, y_pred, labels=config.get('named_labels'), threshold=config.get('threshold'))  # XXX sample_weight


def _parse_number_or_fraction(x):
    if x is None or x == '':
        return None
    if '%' in x:
        return float(x.rstrip('%')) / 100.0
    if '.' in x:
        return float(x)
    return int(x)


def parse_number_or_fraction(x, total=None):
    x = _parse_number_or_fraction(x)
    if total is not None and isinstance(x, float):
        return int(round(x * total))
    return x


def log_report(prefix, metrics, breakdown_re, breakdown_top, breakdown_min, y_true, y_pred, y_pred_text, sample_weight, config, classification_report, outputs=None, num_features=None):
    log_report_one(prefix, metrics, y_true, y_pred, sample_weight, config, classification_report, outputs=outputs, num_features=num_features)

    if breakdown_top and not breakdown_re:
        breakdown_re = re.compile('.+')

    if not breakdown_re or not y_pred_text:
        return

    calculated_metrics = [x for x in metrics if not x.startswith('vw')]

    breakdown_counts = {}

    for item in y_pred_text:
        group = get_breakdown_group(breakdown_re, item)
        breakdown_counts[group] = 1 + breakdown_counts.get(group, 0)

    breakdown_counts = breakdown_counts.items()
    breakdown_counts.sort(key=lambda (key, count): (-count, key == 'nomatch', key))

    total_count = len(y_pred_text)

    print_rest = False
    breakdown_top = parse_number_or_fraction(breakdown_top)
    breakdown_min = parse_number_or_fraction(breakdown_min, total_count)
    original_number_of_groups = len(breakdown_counts)

    if breakdown_min:
        breakdown_counts = [x for x in breakdown_counts if x[-1] >= breakdown_min]

    if breakdown_top and isinstance(breakdown_top, int):
        breakdown_counts = breakdown_counts[:breakdown_top]
    elif breakdown_top and isinstance(breakdown_top, float):
        max_count = round(breakdown_top * total_count)
        result = []
        top_count = 0
        for item in breakdown_counts:
            result.append(item)
            top_count += item[-1]
            if top_count >= max_count:
                break
        breakdown_counts = result

    if len(breakdown_counts) != original_number_of_groups:
        print_rest = True

    groups = [x[0] for x in breakdown_counts]

    indices = {}
    for group in groups:
        indices[group] = len(indices)

    rest_index = len(indices)
    breakdown_mask = []

    for item in y_pred_text:
        group = get_breakdown_group(breakdown_re, item)
        breakdown_mask.append(indices.get(group, rest_index))

    max_length = max(len(x) for x in groups)
    max_length = '%' + str(max_length) + 's'
    breakdown_mask = np.array(breakdown_mask)

    for group in groups:
        group_index = indices.get(group, rest_index)
        mask = breakdown_mask != group_index
        log_report_one(prefix + 'breakdown ' + (max_length % group) + ' ', calculated_metrics, y_true, y_pred, sample_weight, config, classification_report, mask=mask)

    if print_rest:
        mask = breakdown_mask != rest_index
        log_report_one(prefix + 'breakdown rest ', calculated_metrics, y_true, y_pred, sample_weight, config, classification_report, mask=mask)


def json_load_byteified(f):
    return _byteify(json.load(f, object_hook=_byteify))


def _byteify(data, ignore_dicts=False):
    # from http://stackoverflow.com/a/33571117
    if isinstance(data, unicode):
        return data.encode('utf-8')

    if isinstance(data, list):
        return [_byteify(item, ignore_dicts=True) for item in data]

    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True):
            _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }

    return data


def main(to_cleanup):
    if '--parseaudit' in sys.argv:
        parser = optparse.OptionParser()
        parser.add_option('--parseaudit', action='store_true')
        parser.add_option('--includezeros', action='store_true')
        parser.add_option('--oaa', type=int)
        parser.add_option('--top', type=int)
        parser.add_option('--bottom', type=int)
        options, args = parser.parse_args()
        if args:
            sys.exit('Unexpected arguments with --parseaudit: %r' % args)
        parseaudit(sys.stdin, includezeros=options.includezeros, oaa=options.oaa, top=options.top, bottom=options.bottom)
        sys.exit(0)

    if '--version' in sys.argv:
        log_always("%s %s", __file__, __version__)
        sys.exit(0)

    if '--report' in sys.argv or '--tovw' in sys.argv or '--tovw_simple' in sys.argv:
        parser = optparse.OptionParser()
        parser.add_option('--quiet', action='store_true')
    else:
        parser = PassThroughOptionParser()

    # cross-validation and parameter tuning options
    parser.add_option('--kfold', type=int)
    parser.add_option('--workers', type=int)
    parser.add_option('--metric', action='append')
    parser.add_option('--metricformat')
    parser.add_option('--validation')
    parser.add_option('--test')
    parser.add_option('--validation_holdout', type=float, default=0)

    # class weight option
    parser.add_option('--weight', action='append', help='Class weights to use in CLASS:WEIGHT format', default=[])
    parser.add_option('--weight_train', action='append', help='Class weight to use (during training only), in CLASS:WEIGHT format', default=[])
    parser.add_option('--weight_metric', action='append', help='Class weight to use (for weighted metrics only), in CLASS:WEIGHT format', default=[])

    # vowpal wabbit arguments (those that we care about. everything else is passed through)
    parser.add_option('-r', '--raw_predictions')
    parser.add_option('-p', '--predictions', action='append', default=[])
    parser.add_option('-f', '--final_regressor')
    parser.add_option('--readable_model')
    parser.add_option('-i', '--initial_regressor')
    parser.add_option('-d', '--data')
    parser.add_option('-a', '--audit', action='store_true')
    parser.add_option('--named_labels')

    parser.add_option('--remap_label', action='append')
    parser.add_option('--named_labels_file')

    parser.add_option('--readconfig')
    parser.add_option('--writeconfig')

    # preprocessing options:
    Preprocessor.init_option_parser(parser)
    parser.add_option('--columnspec')

    # should be 'count'
    parser.add_option('--ignoreheader', action='store_true')

    # using preprocessor standalone:
    parser.add_option('--tovw')
    parser.add_option('--tovw_simple')
    parser.add_option('--format', help='File format, one of vw|tsv|csv|tab. If not provided, will be guessed from file extension or from file contents')

    # using as perf
    parser.add_option('--report', action='store_true')
    parser.add_option('--toperrors')
    parser.add_option('--topdiffs')
    parser.add_option('--threshold', type=float)
    parser.add_option('--classification_report', action='store_true')
    parser.add_option('--breakdown')
    parser.add_option('--breakdown_top')
    parser.add_option('--breakdown_min')

    # logging and debugging and misc
    parser.add_option('--morelogs', action='count', default=0)
    parser.add_option('--lesslogs', action='count', default=0)
    parser.add_option('--keeptmp', action='store_true')
    parser.add_option('--linemode', action='store_true')

    # extra
    parser.add_option('--vw')

    parser.add_option('--tmpid')
    parser.add_option('--tmp', default='.vwoptimize /tmp/vwoptimize')
    parser.add_option('--foldscript')

    # enable hyperopt
    parser.add_option('--hyperopt', type=int)
    parser.add_option('--hyperopt_alg', default='tpe')
    parser.add_option('--hyperopt_hierarchy')

    tune_args = []
    args = sys.argv[1:]
    index = 0
    while index < len(args) - 1:
        arg = args[index]
        if arg.lstrip('-') in Preprocessor.ALL_OPTIONS_INT and args[index + 1].endswith('?'):
            tune_args.extend(args[index:index + 2])
            del args[index:index + 2]
        else:
            index += 1

    options, args = parser.parse_args(args)
    args += tune_args

    globals()['MINIMUM_LOG_IMPORTANCE'] += options.lesslogs - options.morelogs + int(getattr(options, 'quiet', None) or 0) + args.count('--quiet')
    globals()['KEEPTMP'] = options.keeptmp
    globals()['METRIC_FORMAT'] = options.metricformat or METRIC_FORMAT

    tmp_prefix = None
    tmp_options = options.tmp.split()

    for path in tmp_options:
        if os.path.exists(path):
            tmp_prefix = path
            break
        try:
            os.mkdir(path)
        except Exception, ex:
            sys.stderr.write('Failed to create %r: %s\n' % (path, ex))
        else:
            tmp_prefix = path
            break

    if not tmp_prefix:
        sys.exit('Please specify location for temp files with --tmp' % tmp_options)

    globals()['TMP_PREFIX'] = tmp_prefix

    if options.tmpid:
        globals()['TMPID'] = options.tmpid

    if options.foldscript:
        assert options.foldscript in ('perl', 'awk'), options.foldscript
        globals()['FOLDSCRIPT'] = options.foldscript

    if options.kfold is not None and options.kfold <= 1:
        sys.exit('kfold parameter must > 1')

    if options.breakdown:
        options.breakdown = re.compile(options.breakdown)

    globals()['options'] = options

    config = {
        'orig_commang': ' '.join(sys.argv)
    }

    if options.readconfig:
        config = json_load_byteified(open(options.readconfig))
        log('vwoptimize config = %s', options.readconfig, importance=1)

        if 'regressor' in config and options.initial_regressor is None:
            options.initial_regressor = os.path.normpath(os.path.join(os.path.dirname(options.readconfig), config['regressor']))
            if not os.path.exists(options.initial_regressor):
                sys.exit('Cannot find %r referenced from %r' % (options.initial_regressor, options.readconfig))

        globals()['VW_CMD'] = config.get('vw') or VW_CMD

    if options.vw:
        globals()['VW_CMD'] = options.vw

    if options.data is None and args:
        sys.exit('Must provide -d/--data. In order to read from stdin, pass "-d -".')

    used_stdin = False
    if options.data is None or options.data in STDIN_NAMES:
        used_stdin = True
        filename = None
    else:
        filename = options.data
        if not os.path.exists(filename):
            sys.exit('File not found: %s' % filename)

    named_labels = _make_proper_list(options.named_labels, proper_label)
    if options.named_labels_file:
        named_labels_file = [x.strip() for x in open(options.named_labels_file).readlines()]
        named_labels = named_labels_file + (named_labels or [])

    if named_labels is not None:
        config['named_labels'] = named_labels
        args += ['--named_labels', ','.join(named_labels)]

    weight = parse_weight(options.weight, config.get('named_labels'))
    weight_train = parse_weight(options.weight_train, config.get('named_labels')) or weight
    weight_metric = parse_weight(options.weight_metric, config.get('named_labels')) or weight

    if weight_train:
        config['weight_train'] = weight_train

    if weight_metric:
        config['weight_metric'] = weight_metric

    if options.remap_label:
        config['remap_label'] = parse_mapping(options.remap_label)

    preprocessor_from_options = Preprocessor.from_options(options.__dict__)

    if preprocessor_from_options:
        if config.get('preprocessor'):
            log_always('Preprocessor specified in config (%s) and on command line (%s), going with the latter.', config['preprocessor'], preprocessor_from_options)
        config['preprocessor'] = str(preprocessor_from_options)
        preprocessor = preprocessor_from_options
    elif config.get('preprocessor'):
        preprocessor = Preprocessor.from_options(config['preprocessor'])
    else:
        preprocessor = None

    format = options.format

    if not format and filename:
        format = get_format_from_filename(filename)

    if format and format not in ('vw', 'csv', 'tsv', 'tab'):
        sys.exit('--format must one of vw,csv,tsv,tab not %r' % format)

    format = format or config.get('format')

    if not format:
        if options.columnspec:
            format = 'csv'
        else:
            format = 'vw'

    if options.columnspec:
        config['columnspec'] = _make_proper_list(options.columnspec)
    elif 'columnspec' not in config and format != 'vw':
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
            config.get('remap_label'),
            ignoreheader=options.ignoreheader)
        sys.exit(0)

    if options.threshold is not None:
        config['threshold'] = options.threshold

    need_tuning = 0

    for arg in args:
        if arg.endswith('?'):
            need_tuning = 1
            break

    options.metric = _make_proper_list(options.metric) or []
    calculated_metrics, vw_metrics, show_num_features = split_metrics(options.metric)
    y_true = None
    sample_weight = None
    need_y_true_and_y_pred = calculated_metrics or options.toperrors or options.classification_report or options.topdiffs

    if need_y_true_and_y_pred or options.kfold or need_tuning:
        # cannot work with stdin, write it to a temp file
        if filename is None:
            filename = get_temp_filename(format)
            to_cleanup.append(filename)
            fobj = open(filename, 'wb')
            for line in sys.stdin:
                fobj.write(line)
            flush_and_close(fobj)

    if options.tovw:
        convert_any_to_vw(
            filename,
            format=format,
            output_filename=options.tovw,
            preprocessor=config.get('preprocessor'),
            columnspec=config.get('columnspec'),
            named_labels=config.get('named_labels'),
            remap_label=config.get('remap_label'),
            weights=config.get('weight_train'),
            ignoreheader=options.ignoreheader,
            workers=options.workers)
        sys.exit(0)

    is_multiclass = any([read_argument(args, '--' + x) for x in 'oaa ect csoaa log_multi recall_tree'.split()])

    examples = read_argument(args, '--examples', type=int)

    if need_y_true_and_y_pred:
        if options.validation:
            y_true, sample_weight = read_y_true(options.validation, format, config.get('columnspec'), options.ignoreheader, config.get('named_labels'), config.get('remap_label'))
            if not len(y_true):
                sys.exit('%s is empty' % options.validation)
        else:
            y_true, sample_weight = read_y_true(filename, format, config.get('columnspec'), options.ignoreheader, config.get('named_labels'), config.get('remap_label'), examples=examples)
            if not len(y_true):
                sys.exit('%s is empty' % filename)
        if not config.get('named_labels') and not is_multiclass:
            min_label = np.min(y_true)
            max_label = np.max(y_true)
            config.setdefault('min_label', min_label)
            config.setdefault('max_label', max_label)
            config.setdefault('threshold', (min_label + max_label) / 2.0)

    assert isinstance(options.predictions, list)

    if options.report or options.topdiffs:
        # XXX major source of confusion when report is done on multiclass, since it tries to calculate threshold for it rather than
        # treating it as multiclass. Perhaps refuse to calculate threshold if min_value/max_value is not 0/1 or -1/1 or if there more than 2 classes
        if not options.predictions:
            sys.exit('Must provide -p')

        list_y_pred = []

        for pred in options.predictions:
            if pred in STDIN_NAMES:
                if used_stdin:
                    sys.exit('Can only use stdin in one argument')
                predictions = sys.stdin
                used_stdin = True
            else:
                predictions = pred

            assert y_true is not None
            y_pred, y_pred_text = _load_predictions(predictions, len(y_true), with_text=True, named_labels=config.get('named_labels'))
            list_y_pred.append((y_pred, y_pred_text))

            log_report(prefix='',
                       # vw_* metrics not supported here, but pass them anyway to let the caller now
                       metrics=options.metric,
                       breakdown_re=options.breakdown,
                       breakdown_top=options.breakdown_top,
                       breakdown_min=options.breakdown_min,
                       y_true=y_true,
                       y_pred=y_pred,
                       y_pred_text=y_pred_text,
                       sample_weight=sample_weight,
                       config=config,
                       classification_report=options.classification_report)

            if options.toperrors:
                print_toperrors(options.toperrors, y_true, y_pred, y_pred_text, filename=filename, format=format, ignoreheader=options.ignoreheader)

        if options.topdiffs:
            if len(list_y_pred) <= 1:
                sys.exit('Must have two predictions files specified to compare (pass -p filename1 -p filename2)')
            y_pred, y_pred_text = list_y_pred[0]
            y_pred2, y_pred_text2 = list_y_pred[1]
            print_top_differences(options.topdiffs, y_true, y_pred, y_pred_text, y_pred2, y_pred_text2, filename, format, ignoreheader=options.ignoreheader)

        sys.exit(0)

    else:
        if options.predictions:
            options.predictions = options.predictions[0]
        else:
            options.predictions = None

    args = parse_tuning_args(args)

    if need_tuning:
        # QQQ --initial_regressor is not passed there
        vw_args, preprocessor = main_tune(
            metric=options.metric,
            config=config,
            filename=filename,
            validation=options.validation,
            test=options.test,
            validation_holdout=options.validation_holdout,
            y_true=y_true,
            sample_weight=sample_weight,
            format=format,
            args=args,
            preprocessor_base=preprocessor,
            kfold=options.kfold,
            ignoreheader=options.ignoreheader,
            workers=options.workers,
            hyperopt_rounds=options.hyperopt,
        )
        if vw_args is None:
            sys.exit('tuning failed')
        config['preprocessor'] = str(preprocessor) if preprocessor else None
        config['vw_train_options'] = cleanup_vw_train_options(vw_args)
    else:
        vw_args = ' '.join(args)
        if options.initial_regressor == '' and config.get('vw_train_options'):
            vw_args = vw_args + ' ' + config.get('vw_train_options')
        else:
            config['vw_train_options'] = cleanup_vw_train_options(vw_args)

    vw_filename = None

    weight_train = config.get('weight_train')

    if filename:
        if format == 'vw' and not weight_train and not preprocessor:
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
                remap_label=config.get('remap_label'),
                weights=weight_train,
                ignoreheader=options.ignoreheader,
                workers=options.workers)

    reported = False

    if options.kfold and not need_tuning:
        # QQQ --initial_regressor is not passed there

        assert vw_filename

        cv_pred, raw_cv_pred_text, num_features, outputs = vw_cross_validation(
            vw_filename,
            options.kfold,
            vw_args,
            vw_test_args=extract_test_args(vw_args),
            workers=options.workers,
            with_predictions=bool(calculated_metrics) or options.predictions or options.toperrors,
            with_raw_predictions=bool(options.raw_predictions),
            calc_num_features=show_num_features,
            capture_output=set([_get_stage(m) for m in (vw_metrics or DEFAULT_METRICS)]))

        log_report(prefix='%s-fold ' % options.kfold,
                   metrics=options.metric or DEFAULT_METRICS,
                   breakdown_re=options.breakdown,
                   breakdown_top=options.breakdown_top,
                   breakdown_min=options.breakdown_min,
                   y_true=y_true,
                   y_pred=cv_pred,
                   y_pred_text=None,
                   sample_weight=sample_weight,
                   config=config,
                   classification_report=options.classification_report,
                   outputs=outputs,
                   num_features=num_features)

        if options.predictions:
            write_file(options.predictions, cv_pred)

        if options.raw_predictions:
            write_file(options.raw_predictions, raw_cv_pred_text)

        if options.toperrors:
            print_toperrors(options.toperrors, y_true, cv_pred, y_pred_text=None, filename=filename, format=format, ignoreheader=options.ignoreheader)

        # all of these are related to --kfold if --kfold is enabled
        options.predictions = None
        options.raw_predictions = None
        options.toperror = None

        reported = True

    final_regressor = options.final_regressor

    config_tmp_filename = None
    if options.writeconfig:
        log('write config = %s', options.writeconfig, importance=1)
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

    if not reported or final_regressor_tmp:
        my_args = vw_args

        predictions_fname = options.predictions

        if need_y_true_and_y_pred and not options.validation:
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
            readable_model=readable_model)

        if len(vw_cmd) == 1 and vw_filename is None:
            vw_cmd = vw_cmd[-1]

            # don't want to capture stderr here, so vw_ metrics don't work there

            weight_train = config.get('weight_train')

            if format == 'vw' and not weight_train and not preprocessor:
                popen = Popen(vw_cmd, stdin=sys.stdin, importance=1)
            else:
                log('preprocessor = %s', preprocessor, importance=1 if preprocessor else 0)
                popen = Popen(vw_cmd, stdin=subprocess.PIPE, importance=1)
                for row in open_anything(sys.stdin, format, ignoreheader=options.ignoreheader, force_unbuffered=options.linemode):
                    line = convert_row_to_vw(
                        row,
                        columnspec=config.get('columnspec'),
                        preprocessor=preprocessor,
                        weights=weight_train,
                        named_labels=config.get('named_labels'),
                        remap_label=config.get('remap_label'))
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

        if predictions_fname is not None and y_true is not None:
            y_pred, y_pred_text = _load_predictions(predictions_fname, len(y_true), with_text=True, named_labels=config.get('named_labels'))

        # we don't support extracted metrics there because we don't capture stderr

        log_report(prefix='',
                   metrics=calculated_metrics,
                   breakdown_re=options.breakdown,
                   breakdown_top=options.breakdown_top,
                   breakdown_min=options.breakdown_min,
                   y_true=y_true,
                   y_pred=y_pred,
                   y_pred_text=y_pred_text,
                   sample_weight=sample_weight,
                   config=config,
                   classification_report=options.classification_report)

        if show_num_features and readable_model:
            log_always('num_features = %s', get_num_features(readable_model))

        if options.toperrors:
            print_toperrors(options.toperrors, y_true, y_pred, y_pred_text, filename=filename, format=format, ignoreheader=options.ignoreheader)

    if final_regressor_tmp:
        os.rename(final_regressor_tmp, final_regressor)

    if config_tmp_filename:
        os.rename(config_tmp_filename, options.writeconfig)


if __name__ == '__main__':
    TO_CLEANUP = []
    try:
        main(TO_CLEANUP)
    finally:
        unlink(*TO_CLEANUP)
