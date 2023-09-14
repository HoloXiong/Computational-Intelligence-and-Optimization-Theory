'''
bfs解决小孩分油问题
'''
from collections import deque

# 定义分油状态类
class OilState:
    def __init__(self, B, C, pre) -> None:
        # 油量
        self.B = B
        self.C = C
        # 前驱节点
        self.pre = pre

# 判断状态是否出现过
def is_visited(visited, new_state):
    for state in visited:
        if state.B == new_state.B and state.C == new_state.C:
            return True
    return False

# 广度优先搜索
def bfs():
    # 创建队列
    queue = deque()
    # 初始状态
    source = OilState(0, 0, -1)
    # 访问列表
    visited = []

    # 初始状态入队
    queue.append(source)
    # 维护一个列表，用于路径输出
    path = []
    path.append(source)
    # 前驱节点索引
    index = -1
    
    while(len(queue) != 0):
        # 队首节点出队
        head = queue.popleft()
        index = index+1
        #print('{} {} {} pre:{}'.format(10-head.B-head.C, head.B, head.C, head.pre))

        # 到达目标状态
        if head.B == 5 and head.C == 0:
            optimal_path = []
            # 路径回溯
            while head.pre != -1:
                optimal_path.append(head)
                head = path[head.pre]
            optimal_path.append(source)

            # 路径输出
            steps = 0
            while len(optimal_path) > 0:
                p = optimal_path.pop()
                steps = steps+1
                print('state{}: {} {} {}'.format(steps, 10-p.B-p.C, p.B, p.C))
            print('search finished!total steps:{}'.format(steps))
            return
        
        # 标记已访问
        visited.append(head)
        B = head.B
        C = head.C

        # 根据规则进行倒油
        # (B, C) -->(7, C)
        if B < 7 and not is_visited(visited, OilState(7, C, index)):
            queue.append(OilState(7, C, index))
            path.append(OilState(7, C, index))
        # (B, C) -->(B, 3)
        if C < 3 and not is_visited(visited, OilState(B, 3, index)):
            queue.append(OilState(B, 3, index))
            path.append(OilState(B, 3, index))
        # (B, C) -->(0, C)
        if B > 0 and not is_visited(visited, OilState(0, C, index)):
            queue.append(OilState(0, C, index))
            path.append(OilState(0, C, index))
        # (B, C) -->(B, 0)
        if C > 0 and not is_visited(visited, OilState(B, 0, index)):
            queue.append(OilState(B, 0, index))
            path.append(OilState(B, 0, index))
        # (B, C) -->(0, B+C)
        if B > 0 and B+C <= 3 and not is_visited(visited, OilState(0, B+C, index)):
            queue.append(OilState(0, B+C, index))
            path.append(OilState(0, B+C, index))
        # (B, C) -->(B+C, 0)
        if C > 0 and B+C <= 7 and not is_visited(visited, OilState(B+C, 0, index)):
            queue.append(OilState(B+C, 0, index))
            path.append(OilState(B+C, 0, index))
        # (B, C) -->(B+C-3, 3)
        if C < 3 and B+C >= 3 and not is_visited(visited, OilState(B+C-3, 3, index)):
            queue.append(OilState(B+C-3, 3, index))
            path.append(OilState(B+C-3, 3, index))
        # (B, C) -->(7, B+C-7)
        if B < 7 and B+C >=7 and not is_visited(visited, OilState(7, B+C-7, index)):
            queue.append(OilState(7, B+C-7, index))
            path.append(OilState(7, B+C-7, index))
    print('search failed!')

if __name__ == "__main__":
    bfs()