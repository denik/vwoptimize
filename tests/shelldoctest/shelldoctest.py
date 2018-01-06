#!/usr/bin/env python
"""
Shell Doctest module.

:Copyright: (c) 2009, the Shell Doctest Team All rights reserved.
:license: BSD, see LICENSE for more details.
"""

import commands
import doctest
import inspect
import re
import subprocess
import sys

master = None
_EXC_WRAPPER = 'system_command("%s")'

def system_command(cmd, shell="bash"):
    p = subprocess.Popen('%(shell)s -c "%(cmd)s"' % vars(),
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    status, stdout, stderr = p.wait(), p.stdout.read().strip(), p.stderr.read().strip()
    if status == 0 and stderr == "":
        format = "%(stdout)s"
    elif stdout != "":
        format = "(%(status)d)%(stderr)s\n%(stdout)s"
    else:
        format = "(%(status)d)%(stderr)s"
    result = format % vars()
    if sys.version_info < (2, 5):
        print result
        return
    print(result)

class ShellExample(doctest.Example):
    def __init__(self, source, want, exc_msg=None, lineno=0, indent=0,
                     label=None,
                     options=None):
        doctest.Example.__init__(self, source, want, exc_msg=None, lineno=lineno, indent=indent,
                     options=None)
        self.label = label

class ShellDocTestParser(doctest.DocTestParser):
    _PROMPT = "$"
    _EXC_WRAPPER = _EXC_WRAPPER
    _EXAMPLE_RE = re.compile(r'''
        # Source consists of a PS1 line followed by zero or more PS2 lines.
        (?P<source>
            (?:^  (?P<indent> [ ]*))                   # PS0 line: indent
            (?:   \[(?P<label>.+)\]\n)?                # PS0 line: label
            (?:   (?P<user>[\w]*)@(?P<host>[\w\.-]*)\n)? # PS0 line: user@host
            (?:   [ ]* \$ .*)                          # PS1 line
            (?:\n [ ]* \. [ ].*)*)                        # PS2 lines
        \n?
        # Want consists of any non-blank lines that do not start with PS1.
        (?P<want> (?:(?![ ]*$)    # Not a blank line
                     (?![ ]*\$)   # Not a line starting with PS1
                     .*$\n?       # But any other line
                  )*)
        ''', re.MULTILINE | re.VERBOSE)

    def parse(self, string, name='<string>'):
        string = string.expandtabs()
        min_indent = self._min_indent(string)
        if min_indent > 0:
            string = '\n'.join([l[min_indent:] for l in string.split('\n')])
        output = []
        charno, lineno = 0, 0
        for m in self._EXAMPLE_RE.finditer(string):
            output.append(string[charno:m.start()])
            lineno += string.count('\n', charno, m.start())
            (source, options, want, exc_msg) = \
                     self._parse_example(m, name, lineno)
            if not self._IS_BLANK_OR_COMMENT(source):
                source = source.replace("\n","; ")
                user = m.group('user')
                host = m.group('host')
                if host:
                    if user:
                        cmd_base = "ssh %(user)s@%(host)s '%(source)s'"
                    else:
                        cmd_base = "ssh %(host)s '%(source)s'"
                    source = cmd_base % vars()
                output.append( ShellExample(self._EXC_WRAPPER % source.replace("\n","; "),
                                    want, exc_msg, lineno=lineno,
                                    label=m.group('label'),
                                    indent=min_indent+len(m.group('indent')),
                                    options=options) )
            lineno += string.count('\n', m.start(), m.end())
            charno = m.end()
        output.append(string[charno:])
        return output

    def _parse_example(self, m, name, lineno):
        indent = len(m.group('indent'))
        source_lines = [sl for sl in m.group('source').split('\n') if sl.strip()[1] == " "]
        self._check_prompt_blank(source_lines, indent, name, lineno)
        self._check_prefix(source_lines[1:], ' '*indent + '.', name, lineno)
        source = '\n'.join([sl[indent+len(self._PROMPT)+1:] for sl in source_lines])
        want = m.group('want')
        want_lines = want.split('\n')
        if len(want_lines) > 1 and re.match(r' *$', want_lines[-1]):
            del want_lines[-1]
        self._check_prefix(want_lines, ' '*indent, name,
                           lineno + len(source_lines))
        want = '\n'.join([wl[indent:] for wl in want_lines])
        m = self._EXCEPTION_RE.match(want)
        if m:
            exc_msg = m.group('msg')
        else:
            exc_msg = None
        options = self._find_options(source, name, lineno)
        return source, options, want, exc_msg

    def _check_prompt_blank(self, lines, indent, name, lineno):
        for i, line in enumerate(lines):
            if len(line) >= indent+len(self._PROMPT)+1 and line[indent+len(self._PROMPT)] != ' ':
                raise ValueError('line %r of the docstring for %s '
                                 'lacks blank after %s: %r' %
                                 (lineno+i+1, name,
                                  line[indent:indent+len(self._PROMPT)], line))

class ShellDocTestRunner(doctest.DocTestRunner):
    _EXC_WRAPPER = _EXC_WRAPPER
    _BEFORE, _AFTER = [len(i) for i in _EXC_WRAPPER.split("%s")]

    def __init__(self, checker=None, verbose=None, verbose_level=None, optionflags=0):
        doctest.DocTestRunner.__init__(self, checker=checker, verbose=verbose, optionflags=optionflags)
        self._verbose_level = verbose_level

    def report_start(self, out, test, example):
        source = example.source[self._BEFORE:-(self._AFTER+1)] + "\n"
        if self._verbose_level > 1:
            out('Label:%s\n' % example.label)
        if self._verbose:
            if example.want:
                out('Trying:\n' + doctest._indent(source) +
                    'Expecting:\n' + doctest._indent(example.want))
            else:
                out('Trying:\n' + doctest._indent(source) +
                    'Expecting nothing\n')

    def _failure_header(self, test, example):
        out = [self.DIVIDER]
        if test.filename:
            if test.lineno is not None and example.lineno is not None:
                lineno = test.lineno + example.lineno + 1
            else:
                lineno = '?'
            out.append('File "%s", line %s, in %s' %
                       (test.filename, lineno, test.name))
        else:
            out.append('Line %s, in %s' % (example.lineno+1, test.name))
        out.append('Failed example:')
        source = example.source[self._BEFORE:-(self._AFTER+1)] + "\n"
        out.append(doctest._indent(source))
        return '\n'.join(out)

def testmod(m=None, name=None, globs=None, verbose=None,
            report=True, optionflags=doctest.ELLIPSIS, extraglobs=None,
            raise_on_error=False, exclude_empty=False,
            verbose_level=None, filters=None,
            ):
    if globs == None:
        globs = dict()
    globs.update({"system_command": system_command})
    global master
    if m is None:
        m = sys.modules.get('__main__')
    if not inspect.ismodule(m):
        raise TypeError("testmod: module required; %r" % (m,))
    if name is None:
        name = m.__name__
    finder = doctest.DocTestFinder(parser=ShellDocTestParser(), exclude_empty=exclude_empty)
    if raise_on_error:
        runner = doctest.DebugRunner(verbose=verbose, optionflags=optionflags)
    else:
        runner = ShellDocTestRunner(verbose=verbose, verbose_level=verbose_level, optionflags=optionflags)
    tests = finder.find(m, name, globs=globs, extraglobs=extraglobs)
    if filters:
        _tests = list()
        z = dict([(k,v) for v,k in enumerate(filters)])
        for test in tests:
            test.examples = sorted(filter(lambda x: x.label in filters, test.examples),
                                cmp=lambda x,y: cmp(z[x.label], z[y.label]))
            _tests.append(test)
        tests = _tests
    for test in tests:
        runner.run(test)
    if report:
        runner.summarize()
    if master is None:
        master = runner
    else:
        master.merge(runner)
    if sys.version_info < (2, 6):
        return runner.failures, runner.tries
    return doctest.TestResults(runner.failures, runner.tries)

if __name__ == "__main__":
    testmod()

