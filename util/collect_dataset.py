from typing import Union, Tuple, List
from  dataclasses import dataclass, field
import statistics
import itertools
import csv
import collections
import ast
import sys

# local imports below
from . import read

# This is the most common problems with py3.8-AST-parseable solutions
# for a part of the data.
MOST_COMMON = {('1480', 'A'), ('617', 'A'), ('1453', 'A'), ('200', 'B'), ('1454', 'B'), ('1462', 'A'), ('263', 'A'), ('705', 'A'), ('1465', 'A'), ('1462', 'C'), ('1451', 'A'), ('977', 'A'), ('271', 'A'), ('133', 'A'), ('791', 'A'), ('1491', 'A'), ('281', 'A'), ('118', 'A'), ('266', 'B'), ('1462', 'B'), ('160', 'A'), ('1485', 'A'), ('1473', 'B'), ('1466', 'A'), ('1487', 'A'), ('1452', 'A'), ('1328', 'A'), ('677', 'A'), ('1450', 'A'), ('266', 'A'), ('1490', 'A'), ('1451', 'B'), ('1473', 'A'), ('1466', 'C'), ('1469', 'A'), ('1486', 'A'), ('71', 'A'), ('443', 'A'), ('1475', 'A'), ('1472', 'D'), ('486', 'A'), ('50', 'A'), ('1471', 'A'), ('1490', 'C'), ('1466', 'B'), ('479', 'A'), ('58', 'A'), ('122', 'A'), ('1490', 'B'), ('1478', 'A'), ('231', 'A'), ('1455', 'A'), ('1469', 'B'), ('96', 'A'), ('1472', 'A'), ('1475', 'B'), ('1459', 'A'), ('1471', 'B'), ('1467', 'A'), ('41', 'A'), ('4', 'A'), ('158', 'A'), ('1455', 'C'), ('1', 'A'), ('344', 'A'), ('1476', 'A'), ('734', 'A'), ('546', 'A'), ('131', 'A'), ('110', 'A'), ('1481', 'A'), ('1478', 'B'), ('1463', 'B'), ('116', 'A'), ('1455', 'B'), ('1494', 'A'), ('208', 'A'), ('1454', 'D'), ('1472', 'B'), ('1474', 'A'), ('520', 'A'), ('1463', 'A'), ('1476', 'B'), ('318', 'A'), ('1461', 'A'), ('61', 'A'), ('1030', 'A'), ('112', 'A'), ('228', 'A'), ('1481', 'B'), ('236', 'A'), ('1411', 'A'), ('282', 'A'), ('469', 'A'), ('1454', 'A'), ('467', 'A'), ('1474', 'B'), ('1472', 'C'), ('339', 'A'), ('69', 'A'), ('136', 'A'), ('1492', 'A'), ('1454', 'C'), ('59', 'A')}

# 10 largest classes (with largest file size)
MOST_COMMON10 = {('71','A'),  ('1','A'), ('4','A'), ('158','A'), ('231','A'), ('263','A'), ('118','A'), ('112','A'), ('339','A'), ('282','A')}

# Processed token frequencies. Kept so that one can exclude some of the less common ones.
TOKEN_FREQUENCIES = {'Name': 9896564, 'Load': 8973363, 'Call': 3708524, 'Constant': 3247347, 'Store': 2848715, 'Assign': 1781855, 'BinOp': 1070073, 'Attribute': 985678, 'Expr': 897278, 'Compare': 885553, 'If': 655055, 'Subscript': 646700, 'Add': 591522, 'Module': 442424, 'Eq': 432617, 'For': 370783, 'AugAssign': 313649, 'Sub': 282382, 'Tuple': 178878, 'Mod': 165968, 'arguments': 164552, 'Mult': 160143, 'arg': 146846, 'alias': 137959, 'FunctionDef': 136885, 'BoolOp': 135689, 'Gt': 130665, 'Return': 128730, 'List': 107486, 'UnaryOp': 102230, 'FloorDiv': 93547, 'And': 87624, 'USub': 85157, 'While': 82609, 'NotEq': 81571, 'comprehension': 71816, 'Lt': 67986, 'Div': 65707, 'Slice': 63927, 'ListComp': 61136, 'GtE': 60733, 'Break': 56706, 'Import': 55563, 'LtE': 53641, 'Or': 48065, 'In': 48057, 'ImportFrom': 45774, 'keyword': 37507, 'Lambda': 27667, 'IfExp': 24145, 'Not': 16853, 'NotIn': 16622, 'Continue': 13766, 'Pow': 12080, 'GeneratorExp': 9578, 'Starred': 9039, 'ClassDef': 8923, 'BitAnd': 5865, 'Pass': 5846, 'Dict': 5201, 'FormattedValue': 4150, 'ExceptHandler': 3385, 'Try': 3347, 'RShift': 2752, 'Del': 2267, 'Delete': 2208, 'JoinedStr': 2154, 'Is': 2052, 'Set': 1725, 'BitOr': 1525, 'LShift': 1151, 'BitXor': 1080, 'Global': 1068, 'Raise': 755, 'Assert': 686, 'Yield': 548, 'IsNot': 461, 'AnnAssign': 406, 'DictComp': 365, 'SetComp': 300, 'With': 134, 'withitem': 134, 'UAdd': 117, 'Invert': 103, 'NamedExpr': 40, 'Nonlocal': 31, 'YieldFrom': 16}

def to_simple_tree(node: ast.AST) -> Union[tuple, str]:
    """Input is an AST of a program. Output is a tree with only the node
    names and no value information. E.g. when input is the AST of 

```
from math import ceil
n, m, a = map(int,input().split())
c = ceil(n/a)*ceil(m/a)
print(c)
```

    this function outputs

('Module', [('Expr', [('Call', [('Call', [('Name', ['Load']), ('Lambda', [('arguments', ['arg', 'arg', 'arg']), ('BinOp', [('BinOp', [('BinOp', [('BinOp', [('Name', ['Load']), 'Add', ('Name', ['Load'])]), 'Sub', 'Constant']), 'Div', ('Name', ['Load'])]), 'Mult', ('BinOp', [('BinOp', [('BinOp', [('Name', ['Load']), 'Add', ('Name', ['Load'])]), 'Sub', 'Constant']), 'Div', ('Name', ['Load'])])])])]), ('Starred', [('ListComp', [('Call', [('Name', ['Load']), ('Name', ['Load'])]), ('comprehension', [('Name', ['Store']), ('Call', [('Attribute', [('Call', [('Name', ['Load'])]), 'Load'])])])]), 'Load'])])])])

    """
    curr = type(node).__name__
    kids = list(ast.iter_child_nodes(node))
    if not kids:
        return curr
    return (curr, [to_simple_tree(kid) for kid in kids])

def depth(node: ast.AST):
    kids = list(ast.iter_child_nodes(node))
    return max([1]+[1+depth(kid) for kid in kids])

class NodeCounter(ast.NodeVisitor):
    """This class can go though all nodes of a tree, count them, and put
every node name in a Counter. Usage:

nc = NodeCounter(tokens_dict)
nc.visit(an_ast)
print(f"tree had {nc.num_tokens} tokens; tokens dict updated to {tokens_dict}")

    """
    def __init__(self, tokens_dict: collections.Counter):
        super(NodeCounter, self).__init__()
        self.num_tokens = 0
        self.tokens_dict = tokens_dict
    def generic_visit(self, node):
        self.num_tokens += 1
        self.tokens_dict.update([type(node).__name__])
        ast.NodeVisitor.generic_visit(self, node)

# Struct for keeping track of dataset statistics.
@dataclass
class Stats:
    # Will put every token in this dict
    token_dict : collections.Counter = field(
        default_factory=lambda : collections.Counter())
    
    # Every parsed problem will be put here
    parsed_problems : dict = field(
        default_factory = lambda: collections.defaultdict(list))
    
    # Count the number of solutions per problem
    problems_dict: collections.Counter = field(
        default_factory=lambda : collections.Counter())
    
    # These will be updated with one value for every parsed problem:
    num_tokens: List[int] = field(default_factory = list)
    depth: List[int] = field(default_factory = list)
    num_lines: List[int] = field(default_factory = list)
    
    num_failed: int = 0
    num_tried: int = 0

if __name__ == "__main__":
    stats = Stats()

    # reader = csv.reader(open('short_test_output.csv'))
    reader = csv.reader(open('output_concat.csv'))
    # Skip first line (it has the header).
    next(reader)

    for row in reader:
        submission_id = int(row[1])
        prob = row[2], row[3]
        if prob not in MOST_COMMON:
            continue
        code = read.read_code(submission_id)
        stats.num_tried += 1
        try:
            parsed = ast.parse(code)
        except Exception as e:
            stats.num_failed += 1
        else:
            nc = NodeCounter(stats.token_dict)
            nc.visit(parsed)
            stats.parsed_problems[prob].append((to_simple_tree(parsed), code, row))
            stats.problems_dict[prob] += 1
            stats.num_tokens.append(nc.num_tokens)
            stats.depth.append(depth(parsed))
            stats.num_lines.append(len(code.splitlines()))
            
        if stats.num_tried % 10000 == 0:
            print(f"tried: {stats.num_tried}, succeeded: {stats.num_tried-stats.num_failed}")
            print(f"max num lines: {max(stats.num_lines)}")
            print(f"max tokens: {max(stats.num_tokens)}")
            print(f"max depth: {max(stats.depth)}")
            print(stats.problems_dict.most_common(5))

    print("################")
    print(f"Done collecting data! Got {stats.num_tried-stats.num_failed} solutions. Token dict: ")
    print(stats.token_dict)
    print(f"num lines stats: {statistics.quantiles(stats.num_lines, n=10)}")
    print(f"num tokens stats: {statistics.quantiles(stats.num_tokens, n=10)}")
    print(f"depth stats: {statistics.quantiles(stats.depth, n=10)}")
    print("Most common problems:")
    print(stats.problems_dict.most_common(100))

    for (cont, prob) in stats.parsed_problems:
        f = open(f"cont{cont}_prob{prob}.txt", 'w')
        g = open(f"cont{cont}_prob{prob}_full.txt", 'w', encoding='utf-8')
        for (l, c, r) in stats.parsed_problems[(cont, prob)]:
            print(l, file=f)
            print(l, end='\n\n', file=g)
            print(c, end='\n\n', file=g)
            print(r, end='\n\n', file=g)
            print("\n\n#&$^=#### separator ####", file=g)
