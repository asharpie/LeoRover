import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
import random


class CustomEnv(gym.Env):
    """
    A custom Gym-compatible environment for the PyBullet robot.
    """

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Connect to PyBullet
        p.connect(p.DIRECT)  # Use DIRECT for headless simulation (no GUI for faster training)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load terrain and robot
        self.terrain = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[.05, .05, 1],
            heightfieldTextureScaling=128,
            heightfieldData=[random.uniform(0, 0.03) for _ in range(512 * 512)],
            numHeightfieldRows=512,
            numHeightfieldColumns=512
        )
        self.terrain_body = p.createMultiBody(0, self.terrain)
        p.resetBasePositionAndOrientation(self.terrain_body, [0, 0, 0], [0, 0, 0, 1])

        self.robot = p.loadURDF("leo_robot_1_ros2_shared.urdf", basePosition=[0, 0, 0.5])

        # Simulation settings
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1.0 / 240.0)

        # Action space: control wheel velocities (left and right)
        self.action_space = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)

        # Observation space: position (x, y, z), orientation (yaw, pitch, roll), velocities
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # Initialize variables
        self.target_position = [5, 5, 0]
        self.reset()

    def step(self, action):
        """
        Step the simulation forward by applying the action and returning an observation,
        reward, done, and info.
        """
        # Apply action to wheels
        left_wheel_velocity, right_wheel_velocity = action
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=[2, 3],  # Example wheel joint indices (update based on your URDF)
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[left_wheel_velocity, right_wheel_velocity],
            forces=[500, 500]
        )

        # Step simulation
        p.stepSimulation()

        # Get current state
        observation = self._get_observation()

        # Compute reward: example reward is negative distance to target
        robot_position = observation[:3]
        distance_to_target = np.linalg.norm(np.array(robot_position) - np.array(self.target_position))
        reward = -distance_to_target

        # Check if episode is done (e.g., near target or timeout)
        done = distance_to_target < 0.5  # Near the target

        # Optional additional info
        info = {"distance_to_target": distance_to_target}

        return observation, reward, done, info

    def reset(self):
        """
        Reset the simulation and return the initial observation.
        """
        # Reset robot position and orientation
        p.resetBasePositionAndOrientation(self.robot, [0, 0, 0.5], [0, 0, 0, 1])
        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])

        # Initialize target
        self.target_position = [random.uniform(-5, 5), random.uniform(-5, 5), 0]

        return self._get_observation()

    def render(self, mode="human"):
        """
        Render the environment using PyBullet's GUI.
        """
        if mode == "human":
            p.connect(p.GUI)

    def close(self):
        """
        Close the simulation.
        """
        p.disconnect()

    def _get_observation(self):
        """
        Return the current state of the robot.
        Observation: [x, y, z, yaw, pitch, roll, vx, vy, vz, angular_velocity]
        """
        position, orientation = p.getBasePositionAndOrientation(self.robot)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.robot)
        yaw, pitch, roll = p.getEulerFromQuaternion(orientation)

        # Combine into a single observation array
        observation = np.array([
            *position,  # x, y, z
            yaw, pitch, roll,  # Orientation
            *linear_velocity,  # Velocities
            *angular_velocity  # Angular velocities
        ])
        return observation
