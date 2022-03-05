# https://codeforces.com/contest/1494/problem/A
# Writing test solutions to feed to a solution classifier
# https://github.com/aleloi/ast-classification (it's guessing 
# which out of 104 CF problems this solves)
import itertools

def corr(lst):
    depth = 0
    for x in lst:
        depth += (1 if x == '(' else -1)
        if depth < 0: return False
    return depth == 0

for _ in range(int(input())):
    s = input()
    for a, b, c in itertools.product('()', repeat=3):
        if corr(s.replace('A', a).replace('B', b).replace('C', c)):
            print('YES')
            break
    else:
        print('NO')
