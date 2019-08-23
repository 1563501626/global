# temp = []
# for i in range(100):
#     temp.append(str(i))
# for i in "+-×÷":
#     temp.append(i)
# # {...0: '+', 1: '-', 2: '×', 3: '÷'}
# letter = dict(enumerate(temp))
# # {...'+': 0, '-': 1, '×': 2, '÷': 3}
# letter = dict(zip(letter.values(), letter.keys()))
# print(letter)
# import numpy as np
# label = np.array([[b'1\xc3\x973'],
# [b'2-1'],
# [b'64\xc3\xb78'],
# [b'2+8'],
# [b'60\xc3\xb76'],
# [b'15-3'],
# [b'5+11'],
# [b'6-0'],
# [b'5\xc3\xb71'],
# [b'27\xc3\xb79'],
# [b'8\xc3\x976'],
# [b'32\xc3\xb78'],
# [b'9\xc3\xb79'],
# [b'13-1'],
# [b'0-0'],
# [b'24\xc3\xb78'],
# [b'4\xc3\x974'],
# [b'6-3'],
# [b'7\xc3\x979'],
# [b'72\xc3\xb79'],
# [b'4+0'],
# [b'3-0'],
# [b'10\xc3\x975'],
# [b'6+2'],
# [b'9\xc3\xb73'],
# [b'45\xc3\xb75'],
# [b'9\xc3\x978'],
# [b'35\xc3\xb77'],
# [b'10\xc3\x971'],
# [b'7-5'],
# [b'8\xc3\x971'],
# [b'12\xc3\xb72'],
# [b'18+5'],
# [b'12-0'],
# [b'4+9'],
# [b'9\xc3\x970'],
# [b'6\xc3\x970'],
# [b'8\xc3\x974'],
# [b'30\xc3\xb76'],
# [b'36\xc3\xb76'],
# [b'72\xc3\xb78'],
# [b'6\xc3\xb71'],
# [b'15-7'],
# [b'4+18'],
# [b'1\xc3\x979'],
# [b'2\xc3\x977'],
# [b'32\xc3\xb78'],
# [b'80\xc3\xb78'],
# [b'10-6'],
# [b'30\xc3\xb75'],
# [b'8-1'],
# [b'5\xc3\x970'],
# [b'28\xc3\xb74'],
# [b'8-0'],
# [b'5+16'],
# [b'10-4'],
# [b'15+7'],
# [b'5+0'],
# [b'6+1'],
# [b'14-1'],
# [b'9\xc3\x976'],
# [b'12\xc3\xb76'],
# [b'14-2'],
# [b'48\xc3\xb76'],
# [b'1+5'],
# [b'7\xc3\x977'],
# [b'27\xc3\xb79'],
# [b'9+6'],
# [b'45\xc3\xb79'],
# [b'19-3'],
# [b'2\xc3\x978'],
# [b'9-7'],
# [b'4-2'],
# [b'8+15'],
# [b'12-4'],
# [b'3+10'],
# [b'7-7'],
# [b'9-6'],
# [b'11-0'],
# [b'2\xc3\x972'],
# [b'10+1'],
# [b'6\xc3\x972'],
# [b'6\xc3\xb71'],
# [b'16\xc3\xb72'],
# [b'36\xc3\xb74'],
# [b'80\xc3\xb78'],
# [b'5+13'],
# [b'4\xc3\x972']])
#
# import re
# label_letter = []
# for i in label:
#     string_i = i[0].decode()
#     num = re.search(r'(\d+)([\+\-×÷]+)(\d+)', string_i)
#     first = num.group(1)
#     second = num.group(2)
#     third = num.group(3)
#     label_letter.append([letter[first], letter[second], letter[third]])
#
# print(label_letter)

with open('./text', 'r', encoding='utf8') as f:
    string = f.read()
    string = string.split('\n')

f = open('./text02.csv', 'w', encoding='utf8')
for i in range(500):
    f.write(str(i)+','+string[i]+'\n')
f.close()