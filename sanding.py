"""
A 2D sanding environment.

Copyright Joni Pajarinen 2023.

"""

from typing import Optional
import numpy as np
from gymnasium import core, spaces
import pygame
from pygame import gfxdraw



"""
A 2D sanding environment.

Copyright Joni Pajarinen 2023.

"""

from typing import Optional
import numpy as np
from gymnasium import core, spaces
from gymnasium.envs.registration import register
from gymnasium.envs.classic_control import utils
import ipdb

#
# Environment where a robot should sand sanding areas but avoid
# no-sanding areas.
#
# Both sanding and no-sanding areas are circles. The robot is also a
# circle. When the robot circle collides with a sanding or/and
# no-sanding circle the robot gets a reward.  The robot gets a
# positive reward for sanding circles and a negative reward for
# no-sanding circles. You can collide with multiple circles, each
# circle yielding a reward. Sanding and no-sanding circles can
# overlap.
#
# You can only collide once with a circle: the sanding or/and
# no-sanding circles causing a collision are removed after the
# collision.
#
# The agent specifies target robot xy-coordinates at each time step. A
# PD-controller tries to move the robot towards the target
# xy-coordinates by control of x and y acceleration.
#


class SandingEnv(core.Env):
    # Rendering parameters
    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 20,
        }
    SCREEN_DIM = 500

    # Min and max of x, y coordinates in the 2D operating area
    MIN_SIZE = -50
    MAX_SIZE = 50

    # Radius of sanding and no-sanding circles and the robot circle
    RADIUS = 20

    REW_SAND = 1.0  # Positive reward for sanding a sanding circle
    REW_NOSAND = -1.0  # Negative reward for sanding a no-sanding circle

    # Number of times to execute the PD-controller in one time step
    PD_ITERATIONS = 5

    # PD-controller output (acceleration) limits
    PD_U_MINMAX = np.array([-10, 10])

    def __init__(self, 
                 render_mode: Optional[str] = None, 
                 max_episode_steps=20,
                 n_sanding=1, n_no_sanding=1, radius=10, is_PD_control=False,
                 Kp=0.95, Kd=0.5):

        self.n_sanding = n_sanding  # Number of sanding circles
        self.n_no_sanding = n_no_sanding  # Number of no-sanding circles
        self.RADIUS = radius
        # Do we use PD-control, or, transition directly to target coordinates
        self.is_PD_control = is_PD_control

        if is_PD_control:
            self.metadata['render_fps'] = 30
        else:
            self.metadata['render_fps'] = 3

        self.Kp = Kp  # PD-controller K_p parameter
        self.Kd = Kd  # PD-controller K_d parameter
       
        # Squared distance inside which the robot circle causes a
        # collision between sanding/no-sanding circle centers.
        self.MAX_SQUARED_DIST = (self.RADIUS * 2) * (self.RADIUS * 2)

        self.n_state_dim = 2 * (1 + n_sanding + n_no_sanding)

        # Observation (and state) space consists of robot
        # xy-coordinates, sanding circle center xy-coordinates and
        # no-sanding circle center xy-coordinates.
        low = np.tile(np.ones(2) * self.MIN_SIZE, 1 + self.n_sanding +
                      self.n_no_sanding)
        high = np.tile(np.ones(2) * self.MAX_SIZE, 1 + self.n_sanding
                       + self.n_no_sanding)
        self.observation_space = spaces.Box(low=low, high=high,
                                            dtype=np.float32)

        # Action space consists of x,y target and PD-controller
        # parameters Kp, Kd
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32)
        self.state = None
        self.prev_state = None

        self.clock = None
        self.screen = None
        self.render_mode = render_mode

        self.max_episode_steps = max_episode_steps
        self._counter = 0

        self.robot_positions = []
        self.state_trajs = []


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Randomly position the agent, the sanding, and no-sanding circles.
        # All positions are x,y coordinates of circles with fixed radius.
        self.state = self.np_random.uniform(low=self.observation_space.low,
                                            high=self.observation_space.high)
        self.prev_state = np.copy(self.state)

        self.robot_positions = [self.prev_state[[0, 1]]]
        self.state_trajs = [self.prev_state]
        self.target_xy = self.prev_state[[0, 1]]
        if self.render_mode == "human":
            for pd_id, pd_pos in enumerate(self.robot_positions):
                self._render_frame(pd_pos, self.state_trajs[pd_id])


        self._counter = 0
        return self.state, {}

    def get_coords(self, prev_state, offset, n):
        x_indices = 2 * np.array(range(n)) + offset
        y_indices = x_indices + 1
        return prev_state[x_indices], prev_state[y_indices]

    def collisions_sand(self, prev_state):
        # Robot's current xy-coordinates
        x, y = prev_state[[0, 1]]

        # Collisions with sanding circles
        x_sand, y_sand = self.get_coords(prev_state, offset=2, n=self.n_sanding)
        x_dist = x_sand - x
        y_dist = y_sand - y
        squared_dist = x_dist * x_dist + y_dist * y_dist
        return squared_dist < self.MAX_SQUARED_DIST

    def collisions_no_sand(self, prev_state):
        # Robot's current xy-coordinates
        x, y = prev_state[[0, 1]]

        # Collisions with no-sanding circles
        x_nosand, y_nosand = self.get_coords(prev_state, offset=2 + 2 * self.n_sanding, n=self.n_no_sanding)
        x_dist = x_nosand - x
        y_dist = y_nosand - y
        squared_dist = x_dist * x_dist + y_dist * y_dist
        return squared_dist < self.MAX_SQUARED_DIST



    def pd_control(self, Kp, Kd, dt, 
                         initial_pos, target_pos, initial_velocity, num_steps,
                         prev_state):
        """
        Perform PD control to move an object from its initial position to a target position.

        :param Kp: Proportional gain
        :param Kd: Derivative gain
        :param dt: Time step
        :param initial_pos: Initial position of the object as a numpy array [x, y]
        :param target_pos: Target position of the object as a numpy array [x, y]
        :param initial_velocity: Initial velocity of the object as a numpy array [vx, vy]
        :param num_steps: Number of steps to simulate
        :return: A list of states, where each state is a dictionary with position, velocity, and error
        """
        positions = []
        states = []
        pos = np.array(initial_pos, dtype=float)
        velocity = np.array(initial_velocity, dtype=float)
        prev_error = target_pos - pos

        next_state = np.copy(prev_state)
        
        reward = 0

        for _ in range(num_steps):
            # Calculate error and error derivative
            error = target_pos - pos
            error_diff = (error - prev_error) / dt
            
            # Calculate control output (acceleration)
            control = Kp * error + Kd * error_diff
            
            # Update velocity and position
            velocity += control * dt
            pos += velocity * dt

            # Enforce control limits
            #pos[0] =
            
            pos[pos < self.MIN_SIZE] = self.MIN_SIZE
            pos[pos > self.MAX_SIZE] = self.MAX_SIZE

            positions.append(pos.copy())
            # Save state
            #states.append({
            #    'position': pos.copy(),
            #    'velocity': velocity.copy(),
            #   'error': error.copy()
            #})
            
            # Update previous error
            prev_error = error

            next_state[[0, 1]] = pos

            # Collisions with sanding circles
            cols = self.collisions_sand(prev_state)
            reward = reward + np.sum(cols) * self.REW_SAND  # Reward depends on number of collisions with sanding circles

            # Move collided sanding circles outside the operating area
            next_state[2 + np.flatnonzero(np.repeat(cols, repeats=2))] = self.MIN_SIZE - 2 * self.RADIUS

            # Collisions with no-sanding circles
            cols = self.collisions_no_sand(prev_state)
            reward = reward + np.sum(cols) * self.REW_NOSAND  # Reward depends on number of collisions with no-sanding circles

            # Move collided no-sanding circles outside the operating area
            next_state[2 + 2 * self.n_sanding + np.flatnonzero(np.repeat(cols, repeats=2))] = self.MIN_SIZE - 2 * self.RADIUS

            prev_state = np.copy(next_state)
            states.append(np.copy(next_state))

        return positions, reward, next_state, states

      

    def get_reward_next_state(self, prev_state, new_xy):
        next_state = np.copy(prev_state)
        next_state[[0, 1]] = new_xy

        # Collisions with sanding circles
        cols = self.collisions_sand(prev_state)
        reward = np.sum(cols) * self.REW_SAND  # Reward depends on number of collisions with sanding circles

        # Move collided sanding circles outside the operating area
        next_state[2 + np.flatnonzero(np.repeat(cols, repeats=2))] = self.MIN_SIZE - 2 * self.RADIUS

        # Collisions with no-sanding circles
        cols = self.collisions_no_sand(prev_state)
        reward = reward + np.sum(cols) * self.REW_NOSAND  # Reward depends on number of collisions with no-sanding circles

        # Move collided no-sanding circles outside the operating area
        next_state[2 + 2 * self.n_sanding + np.flatnonzero(np.repeat(cols, repeats=2))] = self.MIN_SIZE - 2 * self.RADIUS

        return reward, next_state

    def step(self, a):
        try:
            a = a.numpy()
        except:
            a = np.array(a)
        a = self.MAX_SIZE * a
        self._counter += 1
        prev_xy = self.prev_state[[0, 1]]
        self.prev_state = np.copy(self.state)

        # Current robot xy-coordinates
        xy = np.copy(self.state)[[0, 1]]

        # Target robot xy-coordinates
        target_xy = a[[0, 1]]
        self.target_xy = a[[0, 1]]


        # Limit target xy-coordinates to the operating area
        target_xy[target_xy < self.MIN_SIZE] = self.MIN_SIZE
        target_xy[target_xy > self.MAX_SIZE] = self.MAX_SIZE
            
        # create a buffer to record the trajectory
        self.robot_positions = [xy]
        self.state_trajs = [np.copy(self.state)]

        if self.is_PD_control:
            # Use self.PD_ITERATIONS of PD-control to go to the target position
            # Note: we do not check for collisions during the move using PD-control, only
            # after the move at the new coordinates.
            velocity = (target_xy - prev_xy)*0
            pd_iter = np.random.randint(self.PD_ITERATIONS-4, self.PD_ITERATIONS+2)
      

            # Amount of noise to add (standard deviation of the Gaussian noise)
            noise_scale = 1.0

            # Generate random Gaussian noise
            noise = np.random.normal(scale=noise_scale, size=target_xy.shape)

            # Add noise to the vector
            target_xy = target_xy + noise

            #xy_changes = pd_control(self.Kp, self.Kd, 1., xy, target_xy, xy*0, pd_iter*2)
            xy_changes, reward, next_state, state_traj = self.pd_control(0.08/4.5, 0.1/4.5, 1.2, xy, target_xy, xy*0, pd_iter*7, self.prev_state)

            self.robot_positions = self.robot_positions + xy_changes
            self.state_trajs = self.state_trajs + state_traj
            
            xy = xy_changes[-1]
            #print('compare', xy, self.target_xy, len(self.robot_positions))
            #print(self.robot_positions)
        else:
            # Go directly to the target position
            xy = target_xy

            # Limit xy-coordinates to the operating area
            xy[xy < self.MIN_SIZE] = self.MIN_SIZE
            xy[xy > self.MAX_SIZE] = self.MAX_SIZE

            # Compute the reward and next state
            reward, next_state = self.get_reward_next_state(self.prev_state, xy)

        self.state = next_state

        if self.render_mode == "human":
            for pd_id, pd_pos in enumerate(self.robot_positions):
                self._render_frame(pd_pos, self.state_trajs[pd_id])

        # Maximum number of time steps
        terminated = self._counter >= self.max_episode_steps
        return self.state, reward, terminated, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            fs = []
            
            for pd_id, pd_pos in enumerate(self.robot_positions):
                f = self._render_frame(pd_pos, self.state_trajs[pd_id])
                #f = self._render_frame(pd_pos)
                fs.append(f)
            #print(self.robot_positions)
            #print(len(fs), len(self.robot_positions))
            return fs
    
    def _render_frame(self, pd_pos, pd_state):
        assert self.render_mode in self.metadata['render_modes']
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.SCREEN_DIM, self.SCREEN_DIM)
                )
            else:  # mode in {"rgb_array", "single_rgb_array"}
                self.screen = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        surf.fill((255, 255, 255))
        #s = self.state
        s = pd_state
        if s is None:
            return None

        # Circle radius in pixels
        size = self.MAX_SIZE - self.MIN_SIZE
        int_radius = int(self.SCREEN_DIM * self.RADIUS / size)

        # Draw PD control movements
  
        x, y = pd_pos
        x = int(self.SCREEN_DIM * (x - self.MIN_SIZE) / size)
        y = int(self.SCREEN_DIM * (y - self.MIN_SIZE) / size)
        # (0, 178, 238) corresponds to the deepskyblue2 color
        gfxdraw.aacircle(surf, x, y, int_radius, (120, 0, 238))
        gfxdraw.filled_circle(surf, x, y, int_radius, (120, 0, 238))

        #Draw target circle
        target_x, target_y = self.target_xy
        target_x = int(self.SCREEN_DIM * (target_x - self.MIN_SIZE) / size)
        target_y = int(self.SCREEN_DIM * (target_y - self.MIN_SIZE) / size)
        gfxdraw.aacircle(surf, int(target_x), int(target_y), int_radius, (34, 39, 134))

        '''
        # Draw a blue robot circle
        x, y = s[0], s[1]
        x = int(self.SCREEN_DIM * (x - self.MIN_SIZE) / size)
        y = int(self.SCREEN_DIM * (y - self.MIN_SIZE) / size)
        # (0, 178, 238) corresponds to the deepskyblue2 color
        gfxdraw.aacircle(surf, x, y, int_radius, (0, 178, 238))
        gfxdraw.filled_circle(surf, x, y, int_radius, (0, 178, 238))
        '''

        # Draw green sanding circles which are inside the operating area
        for i in range(self.n_sanding):
            x, y = s[2 + 2 * i], s[3 + 2 * i]
            x = int(self.SCREEN_DIM * (x - self.MIN_SIZE) / size)
            y = int(self.SCREEN_DIM * (y - self.MIN_SIZE) / size)
            if x > 0 and y > 0:
                # (34, 139, 34) corresponds to the forestgreen color
                gfxdraw.aacircle(surf, x, y, int_radius, (34, 139, 34))
                gfxdraw.filled_circle(surf, x, y, int_radius, (34, 139, 34))

        # Draw red no-sanding circles which are inside the operating area
        for i in range(self.n_sanding):
            x, y = s[2 + 2 * self.n_sanding + 2 * i], s[3 + 2 * self.n_sanding + 2 * i]
            x = int(self.SCREEN_DIM * (x - self.MIN_SIZE) / size)
            y = int(self.SCREEN_DIM * (y - self.MIN_SIZE) / size)
            if x > 0 and y > 0:
                # (178, 34, 34) corresponds to the firebrick color
                gfxdraw.aacircle(surf, int(x), int(y), int_radius, (178, 34, 34))
                gfxdraw.filled_circle(surf, int(x), int(y),
                                      int_radius, (178, 34, 34))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))
        
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode in {"rgb_array", "single_rgb_array"}:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.screen = None