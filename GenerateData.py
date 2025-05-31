import numpy as np

# Let the user input p, q, m, a, b
p = int(input("Enter value for p: "))
q = int(input("Enter value for q: "))
m = int(input("Enter value for m: "))
a = float(input("Enter value for a: "))
b = float(input("Enter value for b: "))

# Let D be a matrix of size (p+q) x m
D = np.zeros((p + q, m))
print(a)
print(b)

# Fill D(i, j) = i * j for i=1:p, j=1:m
for j in range(m):
    for i in range(p):
        D[i, j] = i+2*j 
    D[p,j] =a*D[0,j] + b*D[1,j] # Python is 0-indexed
    print(a)
    print(b)
    print(D[0,j])
    print(D[1,j])
    print(D[p,j])

# Write D to a file data.txt in matrix format
with open("data.txt", "w") as f:
    for row in D:
        f.write(" ".join(str(x) for x in row) + "\n")