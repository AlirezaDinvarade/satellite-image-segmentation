import os
o = 1659
os.chdir('/home/hamtech/Desktop/ss/data_soleyman')
a = os.listdir('/home/hamtech/Desktop/ss/data_soleyman')
print(a)
for i in range(len(a)):
    os.rename(a[i] ,f'pol{o}')
    o = o + 1

