'''
holoxiong
小孩分油问题，假设三个容器为A(10), B(7), C(3)
依题意需要将状态(A, B, C) = (10, 0, 0) --> (5, 5, 0)
由于B+C<=7，所以可以将A视为无限大小的油桶，只要BC未满就可以倒油
即状态(B, C) = (0, 0) --> (5, 5)
'''
# 定义状态类
class OilState:
    def __init__(self, B, C) -> None:
        self.B = B
        self.C = C

# 判断状态是否出现过
def is_visited(visited, state_new):
    for state in visited:
        if state.B == state_new.B and state.C == state_new.C:
            return True
    return False

# 倒油
def pour(stack, visited, state):
    # 状态未出现过则倒油
    if not is_visited(visited, state):
        visited.append(state)
        stack.append(state)

# 深度优先搜索
def dfs():
    # 列表作为栈
    stack = []
    # 定义访问数组
    visited = []
    # 初始化
    begin_state = OilState(0, 0)
    stack.append(begin_state)
    visited.append(begin_state)
    i = 0

    # 开始搜索
    while len(stack)>0:
        i+=1
        top_state = stack.pop()
        B = top_state.B
        C = top_state.C
        print("state {}:{}  {}  {}".format(i, 10-B-C , B, C))

        # 到达目标状态
        if B == 5 and C == 0:
            print("search finished! total steps: {}".format(i))
            return
        
        # 按照8种规则进行状态变换
        # (B, C) -->(7, C)
        if B < 7:
            pour(stack, visited, OilState(7, C))
        # (B, C) -->(B, 3)
        if C < 3:
            pour(stack, visited, OilState(B, 3))
        # (B, C) -->(0, C)
        if B > 0:
            pour(stack, visited, OilState(0, C))
        # (B, C) -->(B, 0)
        if C > 0:
            pour(stack, visited, OilState(B, 0))
        # (B, C) -->(0, B+C)
        if B > 0 and B+C <= 3:
            pour(stack, visited, OilState(0, B+C))
        # (B, C) -->(B+C, 0)
        if C > 0 and B+C <=7:
            pour(stack, visited, OilState(B+C, 0))
        # (B, C) -->(B+C-3, 3)
        if C < 3 and B+C >= 3:
            pour(stack, visited, OilState(B+C-3, 3))
        # (B, C) -->(7, B+C-7)
        if B < 7 and B+C >=7:
            pour(stack, visited, OilState(7, B+C-7))
    print("search failed!")

if __name__ == "__main__":
    dfs()

