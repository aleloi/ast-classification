# https://codeforces.com/contest/1454/problem/D
# Writing test solutions to feed to a solution classifier
# https://github.com/aleloi/ast-classification (it's guessing 
# which out of 104 CF problems this solves)

def factor(x):
    res = []
    p = 2
    while p*p <= x:
        i = 0
        while x % p == 0:
            x, i = x // p, i+1
        if i > 0:
            res.append((i, p))
        p += 1
    if x != 1:
        res.append((1, x))
    return res
    
for t in range(int(input())):
    x = int(input())
    fct = sorted(factor(x), reverse=True)
    i, p = fct[0]
    res = [p for _ in range(i)]
    rest = x // p**i
    res[-1] *= rest
    print(len(res))
    print(*res)
    
