# https://codeforces.com/contest/479/A
# Writing test solutions to feed to a solution classifier
# https://github.com/aleloi/ast-classification (it's guessing 
# which out of 104 CF problems this solves)

a, b, c = [int(input()) for _ in range(3)]
best = 0
ops = [lambda x, y: x+y, lambda x, y: x*y]
for A, B in [((a, b), c), (a, (b, c))]:
    for op1 in ops:
        for op2 in ops:
            if isinstance(A, int): 
                val = op1(A, op2(*B))
            else:
                val = op1(op2(*A), B)
            best = max(best, val)
print(best)
