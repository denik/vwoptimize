# coding: utf-8
import sys
import unicodedata

ideograph_range = []
hangul_range = []
hiragana_range = []
katakana_range = []
combined_range = []


def add_to_range(range, index):
    if not range:
        range.append([index])
        return
    if len(range[-1]) == 1 and range[-1][0] == index - 1:
        range[-1].append(index)
        return
    if range[-1][-1] == index - 1:
        range[-1][-1] = index
        return
    range.append([index])


def count_chars(range):
    result = 0
    for item in range:
        if len(item) == 1:
            result += 1
        else:
            result += item[1] - item[0] + 1
    return result


def print_range(range):
    for item in range:
        for char_index in xrange(item[0], item[1] if len(item) > 1 else item[0] + 1):
            char = unichr(char_index)
            print char_index, char.encode('utf8'), unicodedata.name(char)


for index in xrange(32, 10000000):
    try:
        char = unichr(index)
    except ValueError:
        break
    try:
        name = unicodedata.name(char)
    except Exception, ex:
        pass
    else:
        overlap = []

        for subname, lst in [
                ('IDEOGRAPH', ideograph_range),
                ('HANGUL', hangul_range),
                ('HIRAGANA', hiragana_range),
                ('KATAKANA', katakana_range)]:

            if subname in name:
                add_to_range(lst, index)
                overlap.append(subname)

        if len(overlap) > 1:
            sys.stderr.write('Character %r %s in %r\n' % (index, name, overlap))

        if overlap:
            add_to_range(combined_range, index)


def report(name, lst):
    print '# %s total chars=%s' % (name, count_chars(lst))
    print '%s = %s' % (name, lst)
    print


report('ideograph_range', ideograph_range)
report('hangul_range', hangul_range)
report('hiragana_range', hiragana_range)
report('katakana_range', katakana_range)
report('combined_range', combined_range)
