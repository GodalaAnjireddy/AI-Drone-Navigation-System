import heapq

def astar(grid, start, goal):

    rows = len(grid)
    cols = len(grid[0])

    open_list = []
    heapq.heappush(open_list,(0,start))

    came_from = {}
    cost_so_far = {}

    came_from[start] = None
    cost_so_far[start] = 0

    while open_list:

        current = heapq.heappop(open_list)[1]

        if current == goal:
            break

        neighbors = [
            (current[0]+1,current[1]),
            (current[0]-1,current[1]),
            (current[0],current[1]+1),
            (current[0],current[1]-1)
        ]

        for next in neighbors:

            x,y = next

            if 0 <= x < rows and 0 <= y < cols and grid[x][y] == 0:

                new_cost = cost_so_far[current] + 1

                if next not in cost_so_far or new_cost < cost_so_far[next]:

                    cost_so_far[next] = new_cost
                    priority = new_cost

                    heapq.heappush(open_list,(priority,next))
                    came_from[next] = current

    path = []
    node = goal

    while node:

        path.append(node)
        node = came_from.get(node)

    path.reverse()

    return path