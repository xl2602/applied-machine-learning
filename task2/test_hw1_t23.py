# Task 2.3

import io


def cha_num(text):
    return len(text)


def test_cha():
    file = io.open('input.txt', encoding='utf-8')
    text = file.read().strip()
    assert cha_num(text) == 6



