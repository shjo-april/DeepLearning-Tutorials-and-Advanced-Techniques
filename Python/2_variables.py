

##########################################
# String
##########################################

'''
# C/C++
char* a = "Test";
char b[10] = "Test2";

# Java
String a = "Test";
String b = "Test2";
a + b // "TestTest2"
a.equals(b);
a.---(b);
'''

a = "Test1"
b = "Test2"
print(a + b)

# in
# True: a < b
# False : b > a, X
a in b 

# C:/테스트/main.c
# C:/안드로이드/main.java
print(a in b)

print(a == b)

a = "Hello World"
a_array = ['H', 'e', 'l', 'l', 'o']

print(a_array[0]+a_array[1]+a_array[2])
print(a_array[:3])

a_array = ['H', 'e', 'l', 'l', 'o'] + ['1', '2', '3'] + ['4', '5']

# 1. 
length = len(a_array)
print(a_array[length-2:])

# 2. 
print(a_array[-2:])

# matlab (c + Python)

a = 'Korea'
print(a[::-1])

b = [1, 2, 3]
print(b[::-1])

a = 'Korea 3'

# 1. 
arr = a.split(' ')
print(arr[0]) # Korea
print(arr[1]) # 3

# 2. 
string, length = a.split(' ')
print('Korea 3 5 USA'.split(' '))

