import argparse
# from planning_utils import a_star, heuristic, create_grid
import time
from enum import auto

import msgpack
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.frame_utils import global_to_local
from udacidrone.messaging import MsgID

from planning.planning_utils import *


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) \
                    < max(2.0, 0.5 * np.linalg.norm(self.local_velocity[0:2])):
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2],
                          self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 10
        SAFETY_DISTANCE = 8

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
        _, lat0, _, lon0, _ = np.genfromtxt("colliders.csv", delimiter='0', max_rows=1)
        # TODO: set home position to (lon0, lat0, 0)
        self.set_home_position(lon0, lat0, 0)

        # TODO: retrieve current global position
        print(self.global_position)

        # TODO: convert to current local position using global_to_local()
        local_north, local_east, local_down = global_to_local(self.global_position, self.global_home)
        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt("colliders.csv", delimiter=',', dtype='Float64', skiprows=2)

        # Define a grid for a particular altitude and safety margin around obstacles
        t0 = time.time()

        # 2D grid
        # grid, north_offset, east_offset, north_max, east_max = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)

        # 2D Voronoi graph
        grid, graph, north_offset, east_offset, north_max, east_max = \
            create_grid_and_edges(data, TARGET_ALTITUDE, SAFETY_DISTANCE)

        # Probabilistic graph
        # sampler = Sampler(data)
        # graph = create_graph(sampler.polygons, sampler.sample(300), 10)
        print('grid/graph took {0} seconds to build'.format(time.time() - t0))

        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
        # Define starting point on the grid (this is just grid center)
        grid_start = (-north_offset, -east_offset)
        # TODO: convert start position to current position rather than map center
        start = (local_north, local_east)
        grid_start = (-north_offset + int(start[0]), -east_offset + int(start[1]))
        if 'graph' in globals():
            # graph_start = nearest_neighbor(graph, (local_north, local_east, TARGET_ALTITUDE))
            graph_start = nearest_neighbor(graph, (start[0], start[1]))

        # Set goal as some arbitrary position on the grid
        # TODO: adapt to set goal as latitude / longitude position and convert
        # Identify the map limits
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

        # Run A* to find a path from start to goal
        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)
        print('Local Start and Goal: ', grid_start, grid_goal)
        t0 = time.time()
        if 'graph' not in globals():
            path, cost = a_star(grid, heuristic, grid_start, grid_goal)
        else:
            path, cost = a_star_graph(graph, heuristic, graph_start, graph_goal)

        print('path took {0} seconds to plan'.format(time.time() - t0))

        # TODO: prune path to minimize number of waypoints
        # TODO (if you're feeling ambitious): Try a different approach altogether!
        # Convert path to waypoints
        if 'graph' not in globals():
            waypoints = [[p[0] + north_offset,
                          p[1] + east_offset,
                          TARGET_ALTITUDE,
                          0] for p in path]
        else:
            waypoints = [[p[0], p[1], TARGET_ALTITUDE, 0] for p in path]
        waypoints = prune_path(waypoints)
        if 'graph' in globals():
            waypoints.insert(0, [graph_start[0], graph_start[1], TARGET_ALTITUDE, 0])
            waypoints.append([graph_goal[0], graph_goal[1], TARGET_ALTITUDE, 0])

        print(waypoints)
        # Set self.waypoints
        self.waypoints = waypoints
        # TODO: send waypoints to sim
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
