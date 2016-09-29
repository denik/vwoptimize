import os
import sys
import unittest


def system(cmd):
    cmd = cmd.replace('PYTHON', sys.executable)
    sys.stderr.write('+ %s\n' % cmd)
    if os.system(cmd):
        sys.exit('%r failed' % cmd)


def grab_output(cmd):
    cmd = cmd.replace('PYTHON', sys.executable)
    sys.stderr.write('+ %s\n' % cmd)
    popen = os.popen(cmd)
    out = popen.read()
    retcode = popen.close()
    if retcode is not None:
        sys.stderr.write(('\n%r failed with code %r:\n' % (cmd, retcode)) + out + '\n')
        sys.exit(1)
    return out


def cleanup_tmp():
    system('rm -fr tmp')


class TestCase(unittest.TestCase):

    maxDiff = 20000

    def tearDown(self):
        cleanup_tmp()

    def setUp(self):
        cleanup_tmp()
        os.mkdir('tmp')

    def assert_command_output(self, cmd, expected_output):
        output = grab_output(cmd + ' 2>&1')
        self.assertMultiLineEqual(expected_output, output)
