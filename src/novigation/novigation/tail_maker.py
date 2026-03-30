from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from tf_transformations import euler_from_quaternion
import numpy as np

# for debugging, not needed
def print_cells(cells, map: OccupancyGrid):

	grid = np.zeros((map.info.height, map.info.width))
	for cell in cells:
		grid[cell[1], cell[0]] = 5

	print(grid)

def bresenham(x0: int, y0: int, x1: int, y1: int, gridmap: OccupancyGrid):
	# shamelessly stolen from wikipedia B)
	dx = abs(x1 - x0)
	sx = 1 if x0 < x1 else -1
	dy = -abs(y1 - y0)
	sy = 1 if y0 < y1 else -1
	error = dx + dy

	cells = []
	cost = 0
	while True:
		cells.append((y0, x0)) # y=row, x=col
		t = gridmap.data[x0 + gridmap.info.width*y0]
		cost += t if t >= 0 else 0
		e2 = 2 * error
		if e2 >= dy:
			if x0 == x1: break
			error = error + dy
			x0 = x0 + sx
		if e2 <= dx:
			if y0 == y1: break
			error = error + dx
			y0 = y0 + sy

	return cells, cost

def find_tail(inflated_map: OccupancyGrid, goal_pose: PoseStamped, tail_length: float):

	# Given an inflated map, a goal pose and a tail lenght it will return the tail with the lowest cost

	def pose_to_grid_idx(x: float, y: float, grid_map: OccupancyGrid) -> tuple[int, int]:
		x_idx = int(np.floor((x - grid_map.info.origin.position.x)/grid_map.info.resolution))
		y_idx = int(np.floor((y - grid_map.info.origin.position.y)/grid_map.info.resolution))
		return x_idx, y_idx

	start_x, start_y = pose_to_grid_idx(goal_pose.pose.position.x, goal_pose.pose.position.y, inflated_map)
	
	angle_offsets = np.arange(0,2*np.pi, np.pi/8)

	q = [
		goal_pose.pose.orientation.x,
		goal_pose.pose.orientation.y,
		goal_pose.pose.orientation.z,
		goal_pose.pose.orientation.w
	]
	theta = euler_from_quaternion(q)[2] + np.pi 

	path = []
	min_cost = np.inf

	for angle in angle_offsets:
		end_x, end_y = pose_to_grid_idx(
			goal_pose.pose.position.x + tail_length * np.cos(theta + angle),
			goal_pose.pose.position.y + tail_length * np.sin(theta + angle),
			inflated_map
		)
		if end_x >= inflated_map.info.width or end_y >= inflated_map.info.height or end_y < 0 or end_x < 0: continue
		if inflated_map.data[end_x + inflated_map.info.width*end_y] == 100: continue
		
		cells, cost = bresenham(end_x, end_y, start_x, start_y, inflated_map) # we go from the edge to the goal
		if cost < min_cost:
			min_cost = cost/len(cells) # average cost should be a better measure
			path = cells
		#print_cells(cells, inflated_map) # for debugging

	return path

	

def main():
	
	map = OccupancyGrid()
	map.info.width = 11
	map.info.height = 11
	map.info.resolution = 0.1
	map.data = [0]*map.info.width*map.info.height

	goal = PoseStamped()
	goal.pose.position.x = 0.5
	goal.pose.position.y = 0.5
	goal.pose.orientation.w = 1.0

	path = find_tail(map, goal, 0.5)
	print(f"cheapest path: {path}")

if __name__=="__main__":
	main()