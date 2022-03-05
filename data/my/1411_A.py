# https://codeforces.com/contest/1411/problem/A
# Writing test solutions to feed to a solution classifier
# https://github.com/aleloi/ast-classification (it's guessing 
# which out of 104 CF problems this solves)

for _ in range(int(input())):
    input()
    t = list(reversed(input()))
    i = 0
    while i < len(t) and t[i] == ')':
        i += 1
    print("YES" if i > len(t)-i else "NO")
