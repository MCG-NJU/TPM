

from __future__ import print_function

import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from franka_gripper.msg import MoveActionGoal, GraspActionGoal

from utils import all_close


class MoveGroupPythonInterface(object):

    def __init__(self, config):
        super(MoveGroupPythonInterface, self).__init__()
        self.config = config

        ## BEGIN_SUB_TUTORIAL setup
        ##
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node(self.config['node_name'], anonymous=True)

        ## Instantiate a `RobotCommander` object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        scene = moveit_commander.PlanningSceneInterface()

        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this tutorial the group is the primary
        ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.
        ## This interface can be used to plan and execute motions:
        group_name = self.config['group_name']
        move_group = moveit_commander.MoveGroupCommander(group_name)

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher(
            self.config['topics']['display_trajectory'],
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )
        grasp_publisher = rospy.Publisher(self.config['topics']['grasp_goal'], GraspActionGoal, queue_size=10)
        move_publisher = rospy.Publisher(self.config['topics']['move_goal'], MoveActionGoal, queue_size=10)

        ## END_SUB_TUTORIAL

        ## BEGIN_SUB_TUTORIAL basic_info
        ##
        ## Getting Basic Information
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the name of the reference frame for this robot:
        planning_frame = move_group.get_planning_frame()
        # print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        # print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        # print("============ Available Planning Groups:", robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        # print(robot.get_current_state())
        print("")
        ## END_SUB_TUTORIAL

        # Misc variables
        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.grasp_publisher = grasp_publisher
        self.move_publisher = move_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

        self.initial_pose = move_group.get_current_pose().pose
        self.image_received = False
        self.z_min = -1e9

    def _check_trajectory(self, points, delta_rotation_threshold=0.6):
        src_pose = points[0].positions
        tar_pose = points[-1].positions

        # # if the first joint rotate with a large angle, and the second joint rotate BACK with a large angle
        # if abs(src_pose[0] - tar_pose[0]) > delta_rotation_threshold and src_pose[1] - tar_pose[1] > delta_rotation_threshold:
        #     raise RuntimeError('Planned joints movement not secure!')
        
        for i in range(len(src_pose)):
            if abs(src_pose[i] - tar_pose[i]) > delta_rotation_threshold:
                return False
        
        return True
        
    """ 
    Make a reletive movement according to /deltax, /deltay, /deltaz
    """
    def _delta_pose(self, delta_x, delta_y, delta_z, only_xy=False):
        move_group = self.move_group

        # get current pose
        wpose = move_group.get_current_pose().pose

        # get target pose
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation = wpose.orientation
        pose_goal.position.x = wpose.position.x + delta_x
        pose_goal.position.y = wpose.position.y + delta_y

        if only_xy:
            # keep the target z position and the orientation the same as intial state
            pose_goal.position.z = self.initial_pose.position.z
            pose_goal.orientation = self.initial_pose.orientation
        else:
            pose_goal.position.z = self._modify_absolute_z(wpose.position.z + delta_z, self.z_min)
            pose_goal.orientation = wpose.orientation

        if not self._check_absolute_xy(pose_goal.position.x, pose_goal.position.y):
            return False
        
        plan = move_group.plan(pose_goal)
        plan_success, plan_trajectory = plan[0], plan[1]
        if not plan_success:
            print('warning: motion planning failed')
            return False

        safe_trajectory = self._check_trajectory(plan_trajectory.joint_trajectory.points)
        if not safe_trajectory:
            print('warning: motion dangerous!')
            return False

        move_group.execute(plan_trajectory, wait=True)
        # move_group.set_pose_target(pose_goal)

        # ## Now, we call the planner to compute the plan and execute it.
        # # `go()` returns a boolean indicating whether the planning and execution was successful.
        # success = move_group.go(wait=True)
        # # Calling `stop()` ensures that there is no residual movement
        # move_group.stop()
        # # It is always good to clear your targets after planning with poses.
        # # Note: there is no equivalent function for clear_joint_value_targets().
        # move_group.clear_pose_targets()
        # ## END_SUB_TUTORIAL

        # Test whether whether the robot state is physically close to the target
        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

    """
    Move the end effector's position to an absolute position
    WARNING: This function should be called carefully
    """
    def _go_to_pose_goal(self, x, y, z):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL plan_to_pose

        # Get the current pose
        wpose = move_group.get_current_pose().pose

        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation = wpose.orientation
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z

        if not self._check_absolute_xy(x, y):
            raise RuntimeError(f'({x}, {y}) is not a valid position')

        move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        # `go()` returns a boolean indicating whether the planning and execution was successful.
        success = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        move_group.clear_pose_targets()

        ## END_SUB_TUTORIAL

        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

    """
    Move to the start pose as start_pose.yaml
    """
    def _move_to_start_pose(self, pose=None, pose_config_key='robot_start_pose'):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL plan_to_joint_state
        ##
        ## Planning to a Joint Goal
        # The values are aligned to start_post.yaml
        joint_goal = move_group.get_current_joint_values()
        if pose == None:
            joint_goal[0: 7] = self.config[pose_config_key]
        else:
            joint_goal[0: 7] = pose

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()

        ## END_SUB_TUTORIAL

        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)
    
    """
    Publish grasp goal to grasp objects
    WARNING: the values should be examined
    """
    def _pick(self):
        move_goal = GraspActionGoal()
        move_goal.goal.width = 0.01
        move_goal.goal.epsilon.inner = 0.01
        move_goal.goal.epsilon.outer = 0.06 # 0.06
        move_goal.goal.speed = 0.05
        move_goal.goal.force = 0.5

        self.grasp_publisher.publish(move_goal)

    """extend the gripper to place an object"""
    def _place(self, width=0.08):
        move_goal = MoveActionGoal()
        move_goal.goal.width = width
        move_goal.goal.speed = 0.1
        self.move_publisher.publish(move_goal)
    
    """check whether point (x,y) is availabel"""
    def _check_absolute_xy(self, x, y):
        return True
    
    @staticmethod
    def _modify_absolute_z(absolute_z, z_min):
        return max(absolute_z, z_min)

    """Run the interface, this is an abstract method for the basic class"""
    def run_collect_data(self):
        raise NotImplementedError

