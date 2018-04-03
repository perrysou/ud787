## Project: 3D Motion Planning
Yang Su
---

## Project Summary
1. A 2.5D map of downtown San Francisco is loaded.
2. The environment is discretized into a grid/graph representation.
3. Start and goal locations are set.
4. Searches are performed using A*.
5. Unnecessary waypoints are culled with a collinearity test or ray tracing method (like Bresenham).
6. Waypoints are returned in local ECEF coordinates **[N, E, altitude, heading]**, where the drone’s start location corresponds to **[0, 0, 0, 0]**.

## [Rubric](https://review.udacity.com/#!/rubrics/1534/view) Points
Rubric points and related implementations will be described individually.

---

### Explain the Starter Code
1. `motion_planning.py` implements a drone with state transition and path planning.

2. `planning_utils.py` are planning functions that discretize the map data, perform grid/graph search and truncate waypoints.

### Implementing Path Planning Algorithm

#### 1. Set your global home position
The header of the file contains latitude and longitude information of the map center and is extraced using `numpy.genfromtxt`.
```python
_, lat0, _, lon0, _  = np.genfromtxt("colliders.csv", delimiter='0', max_rows=1)
self.set_home_position(lon0, lat0, 0)
```

#### 2. Set your current local position
`global_to_local` is used to identify the local position of the drone with respect to the global home position.
```python
local_north, local_east, local_down = global_to_local(self.global_position, self.global_home)
```

#### 3. Set grid start position from local position
If grid search is used, the start position has to be offset from the local position. 
If graph search is used, needs to find the nearest point to the local position.

```python
start = (local_north, local_east)
grid_start = (-north_offset + int(start[0]), -east_offset + int(start[1]))
if 'graph' in globals():
    graph_start = nearest_neighbor(graph, (start[0], start[1]))
```

#### 4. Set grid goal position from geodetic coords
Here, the latitude and longitude limites are defined with the corners of the map data and global home using `get_global_limits`.
Next, the goal position is randomized within the map limits and converted to grid/graph coordinates
```python
lon_min, lon_max, lat_min, lat_max = \
        get_global_limits(north_offset, east_offset, north_max, east_max, self.global_home)
# Randomize the goal position within the map
goal_global = (np.random.uniform(lon_min, lon_max), np.random.uniform(lat_min, lat_max), TARGET_ALTITUDE)
goal = global_to_local(goal_global, self.global_home)
grid_goal = (-north_offset + int(goal[0]), -east_offset + int(goal[1]))
grid_goal = (np.clip(grid_goal[0], 0, grid.shape[0] - 1), np.clip(grid_goal[1], 0, grid.shape[1] - 1))
# Make sure the goal position is not an obstacle
while grid[grid_goal[0], grid_goal[1]] == 1:
    goal_global = (np.random.uniform(lon_min, lon_max), np.random.uniform(lat_min, lat_max), TARGET_ALTITUDE)
    goal = global_to_local(goal_global, self.global_home)
    grid_goal = (-north_offset + int(goal[0]), -east_offset + int(goal[1]))
    grid_goal = (np.clip(grid_goal[0], 0, grid.shape[0] - 1), np.clip(grid_goal[1], 0, grid.shape[1] - 1))
if 'graph' in globals():
    graph_goal = nearest_neighbor(graph, (goal[0], goal[1]))
```

#### 5. Modify A* to include diagonal motion (or replace A* altogether)
Diagonal motions are included with the following code snippets. Next, a Voronoi graph search is implemented with `a_star_graph`.

```python
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    SOUTHWEST = (1, -1, np.sqrt(2))
    SOUTH = (1, 0, 1)
    SOUTHEAST = (1, 1, np.sqrt(2))
    EAST = (0, 1, 1)
    NORTHEAST = (-1, 1, np.sqrt(2))
    NORTH = (-1, 0, 1)
    NORTHWEST = (-1, -1, np.sqrt(2))

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])
        
               
def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle
    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x - 1 < 0 or y - 1 < 0 or grid[x - 1, y - 1] == 1:
        valid_actions.remove(Action.NORTHWEST)
    if x - 1 < 0 or y + 1 > m or grid[x - 1, y + 1] == 1:
        valid_actions.remove(Action.NORTHEAST)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if x + 1 > n or y - 1 < 0 or grid[x + 1, y - 1] == 1:
        valid_actions.remove(Action.SOUTHWEST)
    if x + 1 > n or y + 1 > m or grid[x + 1, y + 1] == 1:
        valid_actions.remove(Action.SOUTHEAST)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)

    return valid_actions 

       
def a_star_graph(graph, heuristic, start, goal):
    """Modified A* to work with NetworkX graphs."""

    path = []
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                new_cost = current_cost + cost + heuristic(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((new_cost, next_node))

                    branch[next_node] = (new_cost, current_node)

    path = []
    path_cost = 0
    if found:

        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])

    return path[::-1], path_cost
```

#### 6. Cull waypoints 
A collinearity test is used to remove redundant waypoints.

```python
def collinearity_check(p1, p2, p3, epsilon=1e-6):
    m = np.concatenate((p1, p2, p3), 0)
    det = np.linalg.det(m)
    return abs(det) < epsilon


def prune_path(path):
    pruned_path = [p for p in path]
    # TODO: prune the path!
    i = 0
    while i < len(pruned_path) - 2:
        p1 = np.array(pruned_path[i][:-1]).reshape(1, -1)
        p2 = np.array(pruned_path[i + 1][:-1]).reshape(1, -1)
        p3 = np.array(pruned_path[i + 2][:-1]).reshape(1, -1)

        # If the 3 points are in a line remove
        # the 2nd point.
        # The 3rd point now becomes and 2nd point
        # and the check is redone with a new third point
        # on the next iteration.
        if collinearity_check(p1, p2, p3):
            # Something subtle here but we can mutate
            # `pruned_path` freely because the length
            # of the list is check on every iteration.
            pruned_path.remove(pruned_path[i + 1])
        else:
            i += 1
    return pruned_path
```



### Execute the flight
#### 1. Does it work?
I think it does, although it sometimes fails to find a path given an arbitrary goal position.



