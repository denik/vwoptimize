#!/usr/bin/env python
import sys
import os


def system(cmd):
    if os.system(cmd):
        sys.exit(1)


if not os.path.exists('bincache'):
    os.mkdir('bincache')

if not os.path.exists('bincache/vw'):
    system('git clone --depth 1 https://github.com/JohnLangford/vowpal_wabbit.git /tmp/vw')
    system('cd /tmp/vw && make')
    system('cp /tmp/vw/vowpalwabbit/vw bincache/')

system('cp bincache/vw /usr/local/bin/')
