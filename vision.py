import cv2 
import numpy as np
import constants
import A_star_planner

import matplotlib.pyplot as plt
class Vision_Module:

    def __init__(self, markers_list):

        """
        In this class, we assume that the markers 1 --> 4 will represent the corners of the map
        while the marker 0 will represent the position and orientation of the Thymio.
        """

        # define all parameters to the vision 
        self.markers = markers_list
        self.thymio_position = np.zeros(2) # initialize the position of the thymio
        self.thymio_xframe = np.array([0,0,0]) # used to compute the orientation of the thymio
        self.thymio_orientation = 0 # orientation of the robot
        # initialize the predefined marker dictionary, and the detector parameters
        self.marker_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.dictionary_parameters = cv2.aruco.DetectorParameters()
        # initialize our marker detectors 
        self.marker_detector = cv2.aruco.ArucoDetector(self.marker_dictionary,self.dictionary_parameters)

        self.obstacles = [] # list of detected obstacles
        self.goal = [] # goal coordinates

        self.marker_length = 600 # size of the marker
        self.axis = int(self.marker_length/2) # simply half the size

        self.camera_working = False # is true when we do not see the thymio on the camera
        self.map_centers = [] # save the centers of the markers delimiting the map

        self.focal_length = 1000 
        self.center = [constants.MAP_DIMENSIONS[0]//2,constants.MAP_DIMENSIONS[1]//2] # (w/2,h/2)

        self.camera_mtx = np.array([[self.focal_length, 0, self.center[0]],
                                     [0, self.focal_length, self.center[1]],
                                    [0, 0, 1]], dtype=np.float64)
        
        self.dist_coeffs = np.zeros(5)

        self.x_thymio = [0,0]
        self.real_world_path = []

        self.compute_path = True

        self.real_world_grid_points = []
        self.goals_location = []
        self.occupancy_grid = np.zeros((constants.NUM_CELLS, constants.NUM_CELLS), dtype=int)
        self.path = []

        self.start_position = []
        self.start_position_OcG = []
        

    def detect_markers_and_robot(self, frame):
        # QR detection (to be put in a function alone)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.marker_detector.detectMarkers(gray)
        map_centers = []

        if ids is not None and len(ids)>=1:
            # sort the corners and ids in increasing order of ID: 0,1,2,3,4.
            sorted_pairs = sorted(zip(ids.flatten(), corners), key=lambda x: x[0])
            ids, corners = zip(*sorted_pairs)

            if len(ids) == 5:
                self.camera_working = True
            else:
                self.camera_working = False

            for (marker, id) in zip(corners, ids):
                if id in [1,2,3,4]:
                    index = np.where(ids == id)[0][0]
                    corner = corners[index] # find corners of the marker
                    # get orientation and positions 
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corner, self.marker_length, self.camera_mtx, self.dist_coeffs)
                    # define points representing the frame of the markers
                    points = np.float32([[0, 0, 0], [self.axis, 0, 0], [0, self.axis, 0], [0, 0, -self.axis]]).reshape(-1, 3, 1)
                    # project these to the camera frame
                    axis_points, _ = cv2.projectPoints(points, rvecs[0],tvecs[0],self.camera_mtx,self.dist_coeffs)
                    # convert them to type int
                    axis_points = axis_points.astype(int)
                    cx, cy = axis_points[0].ravel() # get the coordinates of the center
                    map_centers.append([int(cx),int(cy)])
                    cv2.circle(frame, (int(cx), int(cy)), 4, (255, 255, 1), 2) # draw a circle around the center

                elif id == 0:
                    # find the position of the thymio
                    index_thymio = np.where(ids==id)[0][0]
                    # find the corner
                    corner_thymio = corners[index_thymio]
                    # same as above
                    rvecs_thymio,tvecs_thymio,_ = cv2.aruco.estimatePoseSingleMarkers(corner_thymio, self.marker_length, self.camera_mtx, self.dist_coeffs)
                    points_thymio = np.float32([[0, 0, 0], [self.axis, 0, 0], [0, self.axis, 0], [0, 0, -self.axis]]).reshape(-1, 3, 1)
                    axis_points_thymio, _ = cv2.projectPoints(points_thymio, rvecs_thymio[0],tvecs_thymio[0],self.camera_mtx,self.dist_coeffs)
                    axis_points_thymio = axis_points_thymio.astype(int)
                    self.thymio_position[0] = axis_points_thymio[0].ravel()[0]
                    self.thymio_position[1] = axis_points_thymio[0].ravel()[1]
                    self.thymio_xframe = axis_points_thymio[1].ravel()

        self.map_centers = map_centers

    def draw_centers(self, frame):
        if len(self.map_centers) == 4:
            for i in range(len(self.map_centers)-1):
                current_center = self.map_centers[i]
                next_center = self.map_centers[i+1]
                cv2.line(frame,current_center,next_center,(0,0,255),2)
            cv2.line(frame,self.map_centers[len(self.map_centers)-1],self.map_centers[0],(0,0,255),2)

    def warp_to_top_view(self, frame, dst_size=constants.MAP_DIMENSIONS): 
        # this computes the homography matrix and maps our frame to real world!
        """
        Warps the camera's perspective to align with a top-down view.
        
        Args:
            frame: The original camera frame.
            src_points: Detected 4 ArUco marker points in the image.
            dst_size: The desired size of the output top-down view.
            
        Returns:
            warped_frame: The top-down view of the frame.
            homography_matrix: The homography matrix used for the transformation.
        """
        # Define destination points for the top-down view
        dst_points = np.array([
            [0, 0],  # Top-left
            [dst_size[0] - 1, 0],  # Top-right
            [dst_size[0] - 1, dst_size[1] - 1],  # Bottom-right
            [0, dst_size[1] - 1]  # Bottom-left
        ], dtype=np.float32)

        # Compute the homography matrix
        src_points = np.array(self.map_centers,dtype=np.float32)
        homography_matrix, _ = cv2.findHomography(src_points, dst_points)
        # Warp the frame to the top-down view
        warped_frame = cv2.warpPerspective(frame, homography_matrix, dst_size)

        return warped_frame, homography_matrix 
    
    def detect_obstacle(self, frame):
        """
        Detects a red obstacle in the given frame.
        """
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define HSV range for detecting dark blue
        lower_red = np.array([160, 100, 100])  # Lower bound of red in HSV
        upper_red = np.array([180, 255, 255])  # Upper bound of red in HSV

        # Create a mask for the blue object
        mask = cv2.inRange(hsv, lower_red, upper_red)
        # Apply morphological operations to clean the mask
        kernel = np.ones((4, 4), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacles = []

        for contour in contours:
            # Filter small contours based on area
            if cv2.contourArea(contour) > 500:  # Adjust threshold as needed
                # Compute the bounding box of the contour
                x, y, w, h = cv2.boundingRect(contour)
                # Store information about the obstacle
                if h >60 and w>60 :
                    offset = 0
                    obstacles.append((x-offset,y-offset,w+offset,h+offset)) # add bounding box information
        self.obstacles = obstacles

    def detect_goal(self,frame): # find a blue goal

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define HSV range for detecting dark blue
        lower_green = np.array([36, 60, 60])  # Lower bound of green in HSV
        upper_green = np.array([89, 255, 255])  # Upper bound of green in HSV

        # Create a mask for the blue object
        mask = cv2.inRange(hsv, lower_green, upper_green)
        # Apply morphological operations to clean the mask
        kernel = np.ones((4, 4), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        for contour in contours:
            # Filter small contours based on area
            if cv2.contourArea(contour) > 500:  # Adjust threshold as needed
                # Compute the bounding box of the contour
                x, y, w, h = cv2.boundingRect(contour)
                # Store information about the obstacle
                if h >30 and w>30 :
                    self.goal = [x,y,w,h] # add bounding box information

    def grid_points(self,frame,n_cell):
        if len(self.map_centers) == 4:
            rows, cols = n_cell, n_cell
            grid_points = []
            col_total, row_total = constants.MAP_DIMENSIONS
            for row in range(1,rows):
                row_points = []
                for col in range(1,cols):
                    x = col * (col_total/cols)
                    y = row * (row_total/rows)
                    # Draw the points on the original frame
                    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        cv2.circle(frame, (int(x), int(y)), 4, (255, 0, 0), -1)
                        row_points.append((int(x), int(y)))
                grid_points.append(row_points)
        self.real_world_grid_points = grid_points
    
    def find_occupancy_grid(self, grid_points, frame):
        """
        Updates the occupancy grid based on the grid points, obstacles, and goal.

        Args:
            grid_points: List of grid points.
            obstacles: List of obstacles (bounding boxes).
            goal: The goal bounding box.
            frame: The original frame for visualization.

        Returns:
            occupancy_grid: A 2D array representing the grid map.
        """
        rows = len(self.real_world_grid_points)
        cols = len(self.real_world_grid_points[0])

        self.occupancy_grid = np.zeros((constants.NUM_CELLS,constants.NUM_CELLS))
        self.goals_location = []

        # Mark obstacles in the occupancy grid
        enlargement = 60
        for obstacle in self.obstacles:
            x, y, w, h = obstacle
            for i in range(rows):
                for j in range(cols):
                    px, py = self.real_world_grid_points[i][j]
                    if x-enlargement <= px <= (x + w+ enlargement) and (-enlargement+y) <= py <= (y + h + enlargement):
                        self.occupancy_grid[i, j] = 1  # Obstacle

            # Mark the goal in the occupancy grid
            if self.goal:
                gx, gy, gw, gh = self.goal
                for i in range(rows):
                    for j in range(cols):
                        px, py = self.real_world_grid_points[i][j]
                        if gx <= px <= gx + gw and gy <= py <= gy + gh:
                            self.occupancy_grid[i, j] = 2  # Goal
                            self.goals_location.append([px,py])    

    def draw_occupancy_grid(self,frame):
        rows = len(self.real_world_grid_points)
        cols = len(self.real_world_grid_points[0])
        # Visualization of the occupancy grid
        for i in range(rows):
            for j in range(cols):
                px, py = self.real_world_grid_points[i][j]
                if self.occupancy_grid[i, j] == 1:  # Obstacle
                    cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)  # Red
                elif self.occupancy_grid[i, j] == 2:  # Goal
                    cv2.circle(frame, (px, py), 4, (255, 0, 0), -1)  # Blue
                else:  # Free
                    cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)  # Green
        
    def draw_obstacles_and_goal(self, frame):
        
        # Visualize detected obstacles on the frame
        for obstacle in self.obstacles:
            # Draw the bounding box
            x, y, w, h = obstacle # add condition not to focus on objects outside of the map!
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if len(self.goal)==4:
            cv2.rectangle(frame, (self.goal[0],self.goal[1]), (self.goal[0]+self.goal[2], self.goal[1]+self.goal[3]),(255,0,0),2)

    def find_start_position(self,grid_points, thymio_position):
        """
        Find the closest grid point to the Thymio's position.

        Args:
            grid_points: List of grid points (2D list with tuples as points).
            thymio_position: Current position of the Thymio as a tuple or array (x, y).

        Returns:
            start_position: The closest grid point to the Thymio.
        """
        min_distance = float('inf')
        start_position = None

        # Iterate over each grid point to find the closest one
        for row in grid_points:
            for point in row:
                # Calculate Euclidean distance
                distance = np.sqrt((point[0] - thymio_position[0]) ** 2 + (point[1] - thymio_position[1]) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    start_position = point

        return start_position
    
    def get_grid_location(self,start_position, grid_size, map_dimensions):
        """
        Maps a real-world start position to its corresponding occupancy grid location.

        Args:
            start_position: Tuple (x, y) representing real-world coordinates (in pixels).
            grid_size: Tuple (rows, cols) representing the dimensions of the occupancy grid.
            map_dimensions: Tuple (width, height) representing the size of the real-world map (in pixels).

        Returns:
            grid_location: Tuple (row, col) representing the grid indices of the start position.
        """
        x_real, y_real = start_position
        rows, cols = grid_size
        width, height = map_dimensions

        # Calculate cell dimensions
        cell_width = width / cols
        cell_height = height / rows

        # Map real-world coordinates to grid indices
        row = int(y_real / cell_height)
        col = int(x_real / cell_width)

        # Clamp values to ensure they are within grid bounds
        row = max(0, min(row, rows - 1))
        col = max(0, min(col, cols - 1))

        return (row, col)
    
    def visualize_path_on_frame(self,warped_frame, path, grid_points):
        """
        Visualizes the path on the warped frame by mapping grid indices to pixel coordinates.

        Args:
            warped_frame: The top-down view of the map (warped frame).
            path: List of indices representing the path found by the planner.
            grid_points: 2D list of pixel coordinates corresponding to grid points.
        """
        if path is None:
            print("No path found to visualize.")
            return

        # Convert grid indices to pixel coordinates
        pixel_path = [grid_points[idx[0]][idx[1]] for idx in path]
        real_world_path = []
        # Draw the path on the warped frame
        for i in range(len(pixel_path) - 1):
            # Draw lines between consecutive points
            start = pixel_path[i]
            end = pixel_path[i + 1]
            cv2.line(warped_frame, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255, 0, 255), 2)
            

        # Highlight the start and goal positions
        cv2.circle(warped_frame, (int(pixel_path[0][0]), int(pixel_path[0][1])), 6, (0, 0, 255), -1)  # Start (red)
        cv2.circle(warped_frame, (int(pixel_path[-1][0]), int(pixel_path[-1][1])), 6, (255, 0, 0), -1)  # Goal (blue)

        return pixel_path
        


    

# here we define the main function of the vision module which will run in an individual thread

def main_vision(vision=Vision_Module(markers_list=[])):

    cap = cv2.VideoCapture(1,cv2.CAP_DSHOW) 
    if not cap.isOpened():
        print("Cannot open camera")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Width (1080p for your webcam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)  # Height
    cap.set(cv2.CAP_PROP_FPS, 30)
    patience = 0
    never_worked_before = True
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # initialize the vision module with a list of markers
        # detect the markers of the vision module
        vision.detect_markers_and_robot(frame)
        # draw the center of the current markers
        vision.draw_centers(frame)
        if len(vision.map_centers)==4:
            warped, H = vision.warp_to_top_view(frame,constants.MAP_DIMENSIONS)
            if vision.compute_path == True and vision.camera_working == True:
                vision.grid_points(warped,constants.NUM_CELLS) # find the grid
                vision.detect_obstacle(warped) # detect the obstacles
                vision.detect_goal(warped) # detect the goal
                vision.find_occupancy_grid(vision.grid_points,warped)
                
                if patience > 30: # give the stream one second to compute everything then no need to unless asked again below
                    vision.compute_path = False
                else: 
                    patience += 1

            if vision.camera_working == True:
                vision.draw_obstacles_and_goal(warped)
                vision.draw_occupancy_grid(warped)
                x_thymio = H@[vision.thymio_position[0], vision.thymio_position[1], 1]
                x_thymio = x_thymio / x_thymio[2]
                x_thymio = x_thymio[:2].ravel()
                thymio_second_point = vision.thymio_xframe
                thymio_second_point = H @[thymio_second_point[0],thymio_second_point[1],1]
                thymio_second_point = thymio_second_point/thymio_second_point[2]
                thymio_second_point = thymio_second_point[:2].ravel()
                cv2.line(warped,(int(x_thymio[0]),int(x_thymio[1])),(int(thymio_second_point[0]),int(thymio_second_point[1])),(0,255,0),2)
                # This angle is computed in radiants between -pi and pi
                vision.thymio_orientation = (np.arctan2(x_thymio[1]-thymio_second_point[1],x_thymio[0]-thymio_second_point[0]))+np.pi
                if vision.thymio_orientation > np.pi:
                    vision.thymio_orientation -= 2*np.pi
                vision.x_thymio = x_thymio # save this to use in planning
                vision.thymio_orientation = np.rad2deg(vision.thymio_orientation)
                # print(f"[x,y] : {vision.x_thymio} theta: {vision.thymio_orientation}") # UNCOMMENT TO GET MEASUREMENT DATA

                if len(vision.goals_location) > 0 and (vision.compute_path == True or never_worked_before == True):
                    start_position = vision.find_start_position(vision.real_world_grid_points,x_thymio)
                    cv2.circle(warped,start_position,4,(255,0,255),2)
                    start_position_grid = vision.get_grid_location(start_position,(constants.NUM_CELLS,constants.NUM_CELLS),constants.MAP_DIMENSIONS)
                    goals_grid = []
                    for goal in vision.goals_location: 
                        goals_grid.append(vision.get_grid_location(goal,(constants.NUM_CELLS,constants.NUM_CELLS),constants.MAP_DIMENSIONS))
                    global_planner = A_star_planner.A_star_Planner(vision.occupancy_grid, start_position_grid, goals_grid)  # should output a path
                    vision.path = global_planner.a_star_search()
                    
                    never_worked_before = False
            else: 
                vision.compute_path = True
                patience = 0
                never_worked_before = True
            if len(vision.path)!=0:
                vision.real_world_path = vision.visualize_path_on_frame(warped,vision.path, vision.real_world_grid_points)
                    
            cv2.imshow("Warped frame",warped)
        cv2.imshow("Normal frame: ",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_vision(vision=Vision_Module([0,1,2,3,4]))    