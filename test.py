import re

label_letter = []
labels = [[b"1+1"], [b"2+2"], [b"3+3"]]
# 数值化
for i in labels:
    string_i = i[0].decode()
    num = re.search(r'(\d+)([\+\-×÷]+)(\d+)', string_i)
    first = num.group(1)
    second = num.group(2)
    third = num.group(3)
    label_letter.append([first, second, third])
print(label_letter)