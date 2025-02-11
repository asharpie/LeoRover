from operator import truediv

import pybullet as p
import pybullet_data as pd
import random
import time
import keyboard
import numpy as np
import cv2
import threading
import math
from stable_baselines3 import PPO
from custom_env import CustomEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import robot as robot_module

# Connect to PyBullet (GUI mode for visualization)
p.connect(p.GUI)

# Set additional search paths for URDFs and other resources
p.setAdditionalSearchPath(pd.getDataPath())



# Configure terrain parameters
heightPerturbationRange = 0.025
maxSpeed = 10  # Maximum speed for wheels
maxForce = 500  # Maximum force for motors
wheel_joint_indices = []
left_wheel_indices = []
right_wheel_indices = []
numHeightfieldRows = 512
numHeightfieldColumns = 512
point_coordinates = []
touched_points = [False,False,False]

# Camera settings
camera_distance = 3  # Distance from the camera to the target
camera_yaw = 90  # Angle around the target on the horizontal plane
camera_pitch = -30  # Angle above/below the horizontal plane
up_axis_index = 2  # Z-axis is the up-direction


# Projection settings (Perspective)
fov = 60  # Field of view
aspect = 1.0  # Aspect ratio (1:1)
near_plane = 0.1  # Near clipping plane
far_plane = 1000  # Far clipping plane

# Programmatic heightfield data generation
heightfieldData = [0] * numHeightfieldRows * numHeightfieldColumns
for j in range(int(numHeightfieldColumns / 2)):
    for i in range(int(numHeightfieldRows / 2)):
        height = random.uniform(0, heightPerturbationRange)
        heightfieldData[2 * i + 2 * j * numHeightfieldRows] = height
        heightfieldData[2 * i + 1 + 2 * j * numHeightfieldRows] = height
        heightfieldData[2 * i + (2 * j + 1) * numHeightfieldRows] = height
        heightfieldData[2 * i + 1 + (2 * j + 1) * numHeightfieldRows] = height

# Create the heightfield terrain
terrainShape = p.createCollisionShape(
    shapeType=p.GEOM_HEIGHTFIELD,
    meshScale=[.05, .05, 1],
    heightfieldTextureScaling=(numHeightfieldRows - 1) / 2,
    heightfieldData=heightfieldData,
    numHeightfieldRows=numHeightfieldRows,
    numHeightfieldColumns=numHeightfieldColumns
)
terrain = p.createMultiBody(0, terrainShape)
p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])

# Load your robot URDF
robot = p.loadURDF("leo_robot_1_ros2_shared.urdf", basePosition=[0, 0, .5], useFixedBase=False)
joint_count = p.getNumJoints(robot)
for joint_index in range(joint_count):
    joint_info = p.getJointInfo(robot, joint_index)
    joint_name = joint_info[1].decode("utf-8")
    if "wheel" in joint_name:
        wheel_joint_indices.append(joint_index)
        if "wheel_FL_joint" in joint_name:
            left_wheel_indices.append(joint_index)
        elif "wheel_RL_joint" in joint_name:
            left_wheel_indices.append(joint_index)
        elif "wheel_FR_joint" in joint_name:
            right_wheel_indices.append(joint_index)
        elif "wheel_RR_joint" in joint_name:
            right_wheel_indices.append(joint_index)

# Set gravity for the simulation (9.8 m/s^2 downward)
p.setGravity(0, 0, -9.8)

# Set the simulation timestep (e.g., 1/240 seconds per step)
p.setTimeStep(1.0 / 240.0)

# Configure real-time simulation
p.setRealTimeSimulation(1)
# Driving functions
def drive_forward(speed):
    speed = min(max(-maxSpeed, speed), maxSpeed)
    for wheel_index in left_wheel_indices + right_wheel_indices:
        p.setJointMotorControl2(

            bodyUniqueId=robot,
            jointIndex=wheel_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=speed,
            force=maxForce
        )

def turn_in_place(speed):
    speed = min(max(-maxSpeed, speed), maxSpeed)
    for left_wheel in left_wheel_indices:
        p.setJointMotorControl2(
            bodyUniqueId=robot,
            jointIndex=left_wheel,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=speed,
            force=maxForce
        )
    for right_wheel in right_wheel_indices:
        p.setJointMotorControl2(
            bodyUniqueId=robot,
            jointIndex=right_wheel,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=-speed,
            force=maxForce
        )

# Driving arc implementation
def drive_arc(speed, rotation_speed):
    speed = min(max(-maxSpeed, speed), maxSpeed)
    rotation_speed = min(max(-maxSpeed, rotation_speed), maxSpeed)
    for left_wheel in left_wheel_indices:
        p.setJointMotorControl2(
            bodyUniqueId=robot,
            jointIndex=left_wheel,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=speed - rotation_speed,
            force=maxForce
        )
    for right_wheel in right_wheel_indices:
        p.setJointMotorControl2(
            bodyUniqueId=robot,
            jointIndex=right_wheel,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=speed + rotation_speed,
            force=maxForce
        )
start_time = time.time()

def create_non_collision_sphere(position, radius, color=[1, 0, 0, 1]):

    # Create a visual-only sphere (no collision shape)
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,  # Sphere geometry
        radius=radius,  # Radius of the sphere
        rgbaColor=color  # RGBA color for visualization
    )

    # Create a multi-body without any collision shape
    sphere_id = p.createMultiBody(
        baseMass=0,  # Mass of 0 for a static object
        baseCollisionShapeIndex=-1,  # No collision shape
        baseVisualShapeIndex=visual_shape_id,  # Attach the visual shape
        basePosition=position  # Position of the sphere in the environment
    )

    return sphere_id

def get_height(x, y):
    """Calculate the height of the terrain at (x, y)."""
    start_position = [x, y, 10]  # Start the ray 10 units above
    end_position = [x, y, -10]  # End the ray 10 units below
    ray_result = p.rayTest(start_position, end_position)

    # Debug output for ray test
    print(f"Ray test start: {start_position}, end: {end_position}")
    print(f"Ray test result: {ray_result}")

    # Visualize the ray to debug
    #p.addUserDebugLine(start_position, end_position, lineColorRGB=[1, 0, 0], lineWidth=2)

    # Validate if the ray hit the terrain
    terrain_hit = ray_result[0]
    if terrain_hit[0] == terrain:  # Check if the hit object is the terrain
        return terrain_hit[3][2]  # Return the Z (height) of the hit point
    else:
        print(f"No hit detected at ({x}, {y}). Returning default height 0.")
        return 0.0  # Fallback height if no hit

def create_points():
    point_coordinates.append((0, 0, 0))
    for i in range(3):

        xCord = random.random()*10*random.choice([-1,1])
        yCord = random.random()*10*random.choice([-1,1])
        zCord = get_height(xCord, yCord)
        create_non_collision_sphere(position=(xCord, yCord, zCord+.05), radius=0.1, color=[1, 0, 0, 1])
        point_coordinates.append((xCord, yCord, zCord+.05))
    return point_coordinates

def create_path():
    print(point_coordinates[0],point_coordinates[1],point_coordinates[2],point_coordinates[3])
    p.addUserDebugLine(point_coordinates[0], point_coordinates[1], lineColorRGB=[1, 0, 0], lineWidth=2)
    p.addUserDebugLine(point_coordinates[1], point_coordinates[2], lineColorRGB=[1, 0, 0], lineWidth=2)
    p.addUserDebugLine(point_coordinates[2], point_coordinates[3], lineColorRGB=[1, 0, 0], lineWidth=2)

def get_robot_position():
    """
    Get the current position of the robot.
    :return: Tuple (x, y, z)
    """
    position, _ = p.getBasePositionAndOrientation(robot)
    return position


def get_robot_orientation():
    """
    Get the current orientation of the robot as a quaternion.
    :return: Tuple (x, y, z, w)
    """
    _, orientation = p.getBasePositionAndOrientation(robot)
    return orientation


def get_robot_linear_velocity():
    """
    Get the linear velocity of the robot.
    :return: Tuple (vx, vy, vz)
    """
    linear_velocity, _ = p.getBaseVelocity(robot)
    return linear_velocity


def get_robot_angular_velocity():
    """
    Get the angular velocity of the robot.
    :return: Tuple (wx, wy, wz)
    """
    _, angular_velocity = p.getBaseVelocity(robot)
    return angular_velocity

def get_robot_yaw():
    """
    Get the yaw of the robot (rotation around the Z-axis).
    :return: Yaw angle in radians
    """
    _, orientation = p.getBasePositionAndOrientation(robot)
    _, _, yaw = p.getEulerFromQuaternion(orientation)
    return yaw


def get_robot_pitch():
    """
    Get the pitch of the robot (rotation around the Y-axis).
    :return: Pitch angle in radians
    """
    _, orientation = p.getBasePositionAndOrientation(robot)
    _, pitch, _ = p.getEulerFromQuaternion(orientation)
    return pitch


def get_robot_roll():
    """
    Get the roll of the robot (rotation around the X-axis).
    :return: Roll angle in radians
    """
    _, orientation = p.getBasePositionAndOrientation(robot)
    roll, _, _ = p.getEulerFromQuaternion(orientation)
    return roll


def get_robot_state_array():
    """
    Get the robot state including position, orientation, linear velocity, angular velocity,
    yaw, pitch, and roll.
    :return: List containing position, orientation, linear velocity, angular velocity, yaw, pitch, and roll
    """
    # Core states
    position = get_robot_position()
    orientation = get_robot_orientation()
    linear_velocity = get_robot_linear_velocity()
    angular_velocity = get_robot_angular_velocity()

    # Additional states
    yaw = get_robot_yaw()
    pitch = get_robot_pitch()
    roll = get_robot_roll()

    # Combine everything into an array (list)
    return [position, orientation, linear_velocity, angular_velocity, yaw, pitch, roll]


#robot position=x0,y0
#start point of line segment=x1,y1
#end point of line segment=x2,y2
def shortest_distance_to_segment(x0, y0, x1, y1, x2, y2):
    # Vector from A to B
    ABx = x2 - x1
    ABy = y2 - y1

    # Vector from A to P
    APx = x0 - x1
    APy = y0 - y1

    # Dot product of AB and AP
    dotProduct = ABx * APx + ABy * APy

    # Magnitude squared of AB
    magnitudeABSquared = ABx * ABx + ABy * ABy

    # Projection scalar t
    t = dotProduct / magnitudeABSquared

    # Closest point on the line segment
    if t < 0:
        closestPointX, closestPointY = x1, y1
    elif t > 1:
        closestPointX, closestPointY = x2, y2
    else:
        closestPointX = x1 + t * ABx
        closestPointY = y1 + t * ABy

    # Distance from P to the closest point on the line segment
    distance = math.sqrt((x0 - closestPointX) ** 2 + (y0 - closestPointY) ** 2)

    return distance

#robot position=x0,y0
#cos(yaw) (x element of rover orientation)=dx
#sin(yaw) (y element of rover orientation)=dx
#start point of line segment=x1,y1
#end point of line segment=x2,y2
def shortest_angle_to_segment(x0, y0, dx, dy, x1, y1, x2, y2):
    """
    Calculate the smallest angle between the rover's facing direction and the vector
    representing the line segment joining two points.

    x0, y0: Rover's current position
    dx, dy: Rover's normalized facing direction vector
    x1, y1: Start point of the line segment
    x2, y2: End point of the line segment
    """
    # Vector representing the robot's orientation
    robotDirX = dx
    robotDirY = dy

    # Vector representing the line segment
    lineVectorX = x2 - x1
    lineVectorY = y2 - y1

    # Normalize the line segment vector
    lineMag = math.sqrt(lineVectorX ** 2 + lineVectorY ** 2)
    if lineMag == 0:
        raise ValueError("The line segment is a point, not a valid segment.")

    lineVectorX /= lineMag  # Normalize line vector
    lineVectorY /= lineMag

    # Normalize the robot's direction vector
    robotMag = math.sqrt(robotDirX ** 2 + robotDirY ** 2)
    if robotMag == 0:
        raise ValueError("The robot's direction vector cannot be zero.")

    robotDirX /= robotMag  # Normalize robot direction
    robotDirY /= robotMag

    # Compute the dot product
    dotProduct = robotDirX * lineVectorX + robotDirY * lineVectorY

    # Clamp dotProduct to avoid floating point issues
    clampedDotProduct = max(-1.0, min(1.0, dotProduct))

    # Compute the angle (in radians)
    angle = math.acos(clampedDotProduct)

    # Get the sign of the angle using the cross product
    crossProduct = robotDirX * lineVectorY - robotDirY * lineVectorX
    if crossProduct < 0:
        angle = -angle  # Negative angle indicates clockwise direction

    # Convert the angle to degrees
    angle_degrees = math.degrees(angle)

    return angle_degrees  # Return the angle in degrees


def calculate_distance_to_point(point_coordinates):
    """
    Calculate the Euclidean distance from the robot's current position to a specific point.

    :param point_coordinates: Tuple (x, y, z) of the target point's coordinates
    :return: Euclidean distance to the point
    """
    # Get the robot's current position
    robot_position = get_robot_position()

    # Calculate Euclidean distance
    distance = math.sqrt(
        (robot_position[0] - point_coordinates[0]) ** 2 +
        (robot_position[1] - point_coordinates[1]) ** 2 +
        (robot_position[2] - point_coordinates[2]) ** 2
    )

    return distance


def capture_camera_data():
    """
    Captures camera images with the camera mounted at the front of the robot
    and ensures it correctly follows the robot's orientation.
    """
    # Get the robot's current position and orientation
    robot_pos, robot_orientation = p.getBasePositionAndOrientation(robot)

    # Compute the forward vector from the robot's orientation quaternion
    orientation_matrix = p.getMatrixFromQuaternion(robot_orientation)
    forward_vector = [orientation_matrix[0], orientation_matrix[3],
                      orientation_matrix[6]]  # Forward is column 0 (X-axis)

    # Set the camera position in front of the robot
    camera_offset = [0.5, 0, 0.2]  # Offset: Forward (0.5), sideways (0), upward (0.2)
    camera_position = [
        robot_pos[0] + camera_offset[0] * forward_vector[0],  # Move camera forward (front of robot)
        robot_pos[1] + camera_offset[0] * forward_vector[1],
        robot_pos[2] + camera_offset[2]  # Slightly above robot's base
    ]
    downward_adjustment = 0.4
    # Camera looks straight ahead from the front of the robot
    camera_target = [
        camera_position[0] + forward_vector[0],  # Extend target further in the forward direction
        camera_position[1] + forward_vector[1],
        camera_position[2] - downward_adjustment # Same height as the camera position
    ]

    # Up vector remains constant (Z-axis is "up")
    up_vector = [0, 0, 1]

    # Create the view matrix
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_position,  # Camera at the front of robot
        cameraTargetPosition=camera_target,  # Looking forward
        cameraUpVector=up_vector  # Maintain correct "up" direction
    )

    # Create the projection matrix
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=aspect,
        nearVal=near_plane,
        farVal=far_plane,
    )

    # Capture the camera image
    width, height, rgb_img, depth_img, seg_mask = p.getCameraImage(
        width=640,
        height=480,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
    )

    # Convert and display the RGB image using OpenCV (Optional)
    rgb_array = np.array(rgb_img)[:, :, :3]  # Strip the Alpha channel
    rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    #cv2.imshow("Robot Camera Feed", rgb_array)
    #cv2.waitKey(1)

def capture_camera_data_thread():
    """This function will execute in a separate thread."""
    counter = 0
    while p.isConnected():
        if counter % 25 == 0:
            capture_camera_data()  # Existing function
        counter += 1
        time.sleep(1. / 240.0)  # Match simulation rate in separate thread


camera_thread = threading.Thread(target=capture_camera_data_thread)
#camera_thread.daemon = True  # Daemonize the thread to ensure it exits with the main program
camera_thread.start()




create_points()
create_path()




#env = CustomEnv()
#model = PPO("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=10000)
distance_counter = 0
print(f"p.isConnected(): {p.isConnected()}")
keyboard.block_key('w')
keyboard.block_key('a')
keyboard.block_key('s')
counter = 0
shortest_distances = [0,0,0]
relative_angles = [0,0,0]
point_distances = [0,0,0]
print(point_coordinates[0],point_coordinates[1],point_coordinates[2],point_coordinates[3])
point_state = 1
#robot position=x0,y0
#cos(yaw) (x element of rover orientation)=dx
#sin(yaw) (y element of rover orientation)=dx
#start point of line segment=x1,y1
#end point of line segment=x2,y2
while p.isConnected():
    dx=math.cos(get_robot_yaw())
    dy=math.sin(get_robot_yaw())
    robot_x = get_robot_position()[0]
    robot_y = get_robot_position()[1]
    relative_angles[0]=shortest_angle_to_segment(robot_x, robot_y, dx, dy, point_coordinates[0][0], point_coordinates[0][1],point_coordinates[1][0], point_coordinates[1][1])
    relative_angles[1]=shortest_angle_to_segment(robot_x, robot_y, dx, dy, point_coordinates[1][0],point_coordinates[1][1], point_coordinates[2][0],point_coordinates[2][1])
    relative_angles[2]=(robot_x, robot_y, point_coordinates[2][0], point_coordinates[2][1],point_coordinates[3][0], point_coordinates[3][1])
    shortest_distances[0]=shortest_distance_to_segment(robot_x, robot_y, point_coordinates[0][0], point_coordinates[0][1],point_coordinates[1][0], point_coordinates[1][1])
    shortest_distances[1]=shortest_distance_to_segment(robot_x, robot_y, point_coordinates[1][0], point_coordinates[1][1],point_coordinates[2][0], point_coordinates[2][1])
    shortest_distances[2]=(robot_x, robot_y, dx, dy, point_coordinates[2][0], point_coordinates[2][1],point_coordinates[3][0], point_coordinates[3][1])
    point_distances[0]=calculate_distance_to_point(point_coordinates[1])
    point_distances[1]=calculate_distance_to_point(point_coordinates[2])
    point_distances[2]=calculate_distance_to_point(point_coordinates[3])
    if abs(robot_x - point_coordinates[1][0]) <= 0.1 and abs(robot_y - point_coordinates[1][1]) <= 0.1:
        touched_points[0] = True
        point_state = 2
    if touched_points[0]:
        if abs(robot_x - point_coordinates[2][0]) <= 0.1 and abs(robot_y - point_coordinates[2][1]) <= 0.1:
            touched_points[1] = True
            point_state = 3
    if touched_points[1]:
        if abs(robot_x - point_coordinates[3][0]) <= 0.1 and abs(robot_y - point_coordinates[3][1]) <= 0.1:
            touched_points[2] = True
            point_state = 0
    if distance_counter % 40 == 0:
        if touched_points[0]:
            if touched_points[1]:
                if not touched_points[2]:
                    print("Angle to segment 3: " + str(relative_angles[2]))
                    print("Distance to segment 3: " + str(shortest_distances[2]))
                    print("Distance to point 3: " + str(point_distances[2]))
            else:
                print("Angle to segment 2: " + str(relative_angles[1]))
                print("Distance to segment 2: " + str(shortest_distances[1]))
                print("Distance to point 2: " + str(point_distances[1]))
        else:
            print("Angle to segment 1: " + str(relative_angles[0]))
            print("Distance to segment 1: " + str(shortest_distances[0]))
            print("Distance to point 1: " + str(point_distances[0]))


    distance_counter += 1
    if keyboard.is_pressed('w'):
        drive_forward(5)
    elif keyboard.is_pressed('s'):
        drive_forward(-5)
    elif keyboard.is_pressed('a'):
        turn_in_place(-5)
    elif keyboard.is_pressed('d'):
        turn_in_place(5)
    elif keyboard.is_pressed('e'):
        drive_arc(5, -2)
    elif keyboard.is_pressed('q'):
        drive_arc(5, 2)
    else:
        drive_forward(0)  # Stop all wheels
    p.stepSimulation()

    # Sleep to match the simulation rate
    time.sleep(1. / 240.0)


    # Break after 10 seconds of simulation

# Disconnect from the simulation
p.disconnect()


