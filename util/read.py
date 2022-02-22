import sys
import ast
from collections import Counter

def read_code(id):
    BLOCK_LEN = 2**10 - 2
    BLOCK_BYTES = BLOCK_LEN * 4 + 8
    fileId = (id - 1) // 10**6 + 1
    id = (id - 1) % 10**6
    blockId = id // BLOCK_LEN
    blockPos = id % BLOCK_LEN
    with open(str(fileId) + '.bin', 'rb') as f:
        f.seek(blockId * BLOCK_BYTES)
        buf = f.read(BLOCK_BYTES)
        offset = int.from_bytes(buf[:8], 'little')
        for i in range(blockPos):
            offset += int.from_bytes(buf[i * 4 + 8:i * 4 + 12], 'little')
        length = int.from_bytes(buf[blockPos * 4 + 8:blockPos * 4 + 12], 'little')
        f.seek(offset)
        return f.read(length).decode('utf-8')

if __name__ == '__main__':
    

    code = read_code(int(sys.argv[1]))
    parsed = ast.parse(code)


    C : dict = Counter()
    class MyVis(ast.NodeVisitor):
        def generic_visit(self, node):
            C.update([type(node).__name__])
            ast.NodeVisitor.generic_visit(self, node)
        
    MyVis().visit(parsed)
    print(ast.dump(parsed))
    print(C)
