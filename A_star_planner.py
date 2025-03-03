import numpy as np
import math
from heapq import heappush, heappop
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class A_star_Planner():
    def __init__(self,grid,start_position,end_positions):
        self.start = start_position
        self.goal = self.goal_selection(end_positions)
        self.grid = self.compatible_grid(grid)

    def compatible_grid(self,grid):
        rows = len(grid)
        cols = len(grid[0])

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 2:
                    grid[i][j] = 0
        return grid

    def goal_selection(self, goal_positions):
        """
        Determine the goal position as the closest node to the centroid of the goal set.

        :return: Tuple (x, y) representing the selected goal position.
        """
        # Compute the centroid of the goal nodes
        goal_array = np.array(goal_positions)

        # Find the closest goal node to the centroid
        distances = [np.linalg.norm(np.array(node) - self.start) for node in goal_array]
        closest_index = np.argmin(distances)
        return goal_array[closest_index]

    def heuristic(self, a):
        # Minimum distance to any of the goal nodes
        return math.sqrt((a[0] - self.goal[0])**2 + (a[1] - self.goal[1])**2) 
    
    def a_star_search(self):
        # Initialize the open set as a priority queue and add the start node
        open_set = []
        heappush(open_set, (self.heuristic(self.start), 0, self.start))  # (f_cost, g_cost, position)

        # Initialize the came_from dictionary
        came_from = {}
        # Initialize g_costs dictionary with default value of infinity and set g_costs[self.start] = 0
        g_costs = {self.start: 0}
        # Initialize the explored set
        explored = set()
        operation_count = 0
        reached_goal = None

        while open_set:
            # Pop the node with the lowest f_cost from the open set
            current_f_cost, current_g_cost, current_pos = heappop(open_set)

            # Add the current node to the explored set
            explored.add(current_pos)

            # Check if the current node is in the goals set
            diff = current_pos-self.goal
            k = np.unique(diff)
            if len(k)==1:
                if k == 0:
                    reached_goal = current_pos
                    break

            # Get the neighbors of the current node (up, down, left, right)
            neighbors = [
                (current_pos[0] - 1, current_pos[1]),  # Up
                (current_pos[0] + 1, current_pos[1]),  # Down
                (current_pos[0], current_pos[1] - 1),  # Left
                (current_pos[0], current_pos[1] + 1),  # Right
                (current_pos[0] - 1, current_pos[1] + 1),  # Up-right
                (current_pos[0] - 1, current_pos[1] - 1),  # Up-left
                (current_pos[0] + 1, current_pos[1] + 1),  # Down-right
                (current_pos[0] + 1, current_pos[1] - 1)   # Down-left
            ]

            for neighbor in neighbors:
                # Check if neighbor is within bounds and not an obstacle
                if (0 <= neighbor[0] < self.grid.shape[0]) and (0 <= neighbor[1] < self.grid.shape[1]):
                    if self.grid[neighbor[0], neighbor[1]] != 1 and neighbor not in explored:
                        
                        # Determine if the move is diagonal
                        #is_diagonal = abs(neighbor[0] - current_pos[0]) == 1 and abs(neighbor[1] - current_pos[1]) == 1
                        # Use sqrt(2) for diagonal moves and 1 for horizontal/vertical moves
                        #movement_cost = math.sqrt(2) if is_diagonal else 1
                        movement_cost = 1

                        # Calculate tentative_g_cost
                        tentative_g_cost = current_g_cost + movement_cost + self.grid[neighbor[0], neighbor[1]]

                        # If this path to neighbor is better than any previous one
                        if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                            # Update came_from, g_costs, and f_cost
                            came_from[neighbor] = current_pos
                            g_costs[neighbor] = tentative_g_cost
                            f_cost = tentative_g_cost + self.heuristic(neighbor)

                            # Add neighbor to open set
                            heappush(open_set, (f_cost, tentative_g_cost, neighbor))
                            operation_count += 1

        else:
            # If no goal is found
            return None, explored, operation_count, None

        # Reconstruct path
        path = []
        if reached_goal:
            path.append(reached_goal)  # Append the reached goal to the path
        while current_pos in came_from:
            current_pos = came_from[current_pos]
            path.append(current_pos)
        path.append(self.start)

        
        return path[::-1]


