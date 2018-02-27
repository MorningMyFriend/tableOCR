# -*- coding:utf-8-*-

# 判断一个unicode是否是汉字
def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fff':
        return True
    else:
        return False


# 判断一个unicode是否是数字
def is_number(uchar):
    if u'\u0030' <= uchar <= u'\u0039':
        return True
    else:
        return False


# 判断一个unicode是否是英文字母
def is_alphabet(uchar):
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    else:
        return False


# 判断是否非汉字，数字和英文字符
def is_other(uchar):
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False


if __name__ == "__main__":
    ustring = u'中国 人名ａ高频Ａ'
    # 判断是否有其他字符；
    for item in ustring:
        if (is_other(item)):
            break

    # 简单方法
    ord(ustring[0]) > 127  # 汉字
    ustring[0] > chr(127)  # 汉字