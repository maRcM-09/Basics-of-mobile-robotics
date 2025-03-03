MARKERS = [0,1,2,3,4]
# dimensions in mm
MAP_DIMENSIONS = (1189,841)
# number of cells in the map
NUM_CELLS = 20
# current waypoint was reached
MIN_DIST_TO_WAYPOINT = 25
# cruising speed
V = 1200
# Physical Thymio Parameters
WHEEL_RADIUS = 20
WHEEL_DISTANCE = 105
# P-Controller gain
K_P_RHO = 0.5
K_P_ALPHA = 2
# Obstacle Avoidance Speed
V_AVOID = 80
# Obstacle Avoidance Gains and Scales
SENSOR_SCALE = 200
WEIGHT_LEFT = [15, 8, -5, -8, -15, 2, -2]
WEIGHT_RIGHT = [-15, -8, -5, 8, 15, -2, 2]
SENSOR_THRESHOLD = 600
# Conversion scale from thymio to cm/s
SPEED_SCALE = 3.5
CONVERSION_FACTOR = 0.04
