import random
file1 = open("A.txt","w")
file2 = open("data.txt","r")
r = 2  ** 256
M = (2 ** 255) - 19
R_square = 1444
b = 10
for a  in range (1,b):
    x = int(file2.readline())
    y = int(file2.readline())

    result_add = (x + y) % M
    result_sub = (x - y) % M

    r_inv = pow(r, -1, M)
    result_1 = (x * y * r_inv) % M
    result_mult = (result_1 * R_square * r_inv) % M

    mod_inverse = pow(x, -1, M)
    result_inv = (mod_inverse * r) % M

    file1.write("x = ")
    file1.write(str(x))
    file1.write("\n")
    file1.write("y = ")
    file1.write(str(y))
    file1.write("\n")
    file1.write("result_add = ")
    file1.write(str(result_add))
    file1.write("\n")
    file1.write("result_sub = ")
    file1.write(str(result_sub))
    file1.write("\n")
    file1.write("result_mult = ")
    file1.write(str(result_mult))
    file1.write("\n")
    file1.write("result_inv = ")
    file1.write(str(result_inv))
    file1.write("\n")