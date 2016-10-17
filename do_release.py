import sys
import os
import re

filename = 'vwoptimize.py'
__version__ = '0.4.0dev'
version_regex = r"__version__ = '([^']+)'"
versiondev_regex = r"(\.\d)dev(')"


def write(filename, data):
    fobj = open(filename, 'wb')
    fobj.write(data)
    fobj.flush()
    os.fsync(fobj.fileno())
    fobj.close()


def system(cmd):
    if os.system(cmd) != 0:
        sys.exit('%r failed' % cmd)


git_describe = os.popen('git describe --tags --dirty').read().strip()
print 'Version from git: %r' % git_describe
version_from_git = git_describe.split('-')[0]

data = open(filename, 'rb').read()
header, rest = data[:200], data[200:]
version_from_source = re.findall(version_regex, header)
assert len(version_from_source) == 1, version_from_source
version_from_source = version_from_source[0]
print 'Version from source: %r' % version_from_source

assert 'dev' in version_from_source, version_from_source
version_from_source = version_from_source.replace('dev', '')

if version_from_git != version_from_source:
    sys.exit('Version in the source (%r) is not the same as from git tag (%r), perhaps you forgot "git tag"' %
             (version_from_source, version_from_git))


system("cp %s %s.backup" % (filename, filename))
header, count = re.subn(versiondev_regex, r"\1\2", header)
assert count == 1, (count, version_regex, header)
header, count = re.subn(re.escape("GIT"), git_describe, header)

write(filename, header + rest)
os.system('diff -U 1 %s.backup %s' % (filename, filename))

system("%s setup.py sdist" % sys.executable)
system("cp %s.backup %s" % (filename, filename))

if 'dirty' in git_describe:
    sys.exit('dirty working copy, will not upload')
