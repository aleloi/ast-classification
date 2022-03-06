# https://codeforces.com/contest/1491/problem/A
# Writing test solutions to feed to a solution classifier
# https://github.com/aleloi/ast-classification (it's guessing 
# which out of 104 CF problems this solves)

N, Q = map(int, input().split())
ARR = list(map(int, input().split()))
n = [ARR.count(0), ARR.count(1)]

def upd(x):
    x-=1
    d = ARR[x] if ARR[x] == 1 else -1
    n[0] += d
    n[1] -= d
    ARR[x] = 1-ARR[x]

for _ in range(Q):
    t, arg = map(int, input().split())
    [None,
     upd,
     lambda k: print(int(n[1] >= k))
     ] [t] ( arg )
