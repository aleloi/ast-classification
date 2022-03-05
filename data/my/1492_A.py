# https://codeforces.com/contest/1492/problem/A
# Writing test solutions to feed to a solution classifier
# https://github.com/aleloi/ast-classification (it's guessing 
# which out of 104 CF problems this solves)

for _ in range(int(input())):
    p, a, b, c = map(int, input().split())
    f = lambda md, x: 0 if md == 0 else x - md
    print(min([f(p % x, x) for x in [a, b, c]]))
