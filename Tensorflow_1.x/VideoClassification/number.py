
length = 31

a = list(range(length))
size = len(a) // 5

print(a, size, [a[i] for i in range(0, length - 1, size)])