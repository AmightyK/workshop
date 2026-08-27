#!/usr/bin/env python3
"""Launch Nav2 for the custom AMR in the official warehouse world."""

from pathlib import Path
import tempfile

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, Substitution
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


DEMO_DIR = Path(__file__).resolve().parents[1]
NAV2_SHARE = Path(get_package_share_directory("nav2_bringup"))
WAREHOUSE_MAP = DEMO_DIR / "maps" / "warehouse_lidar.yaml"
NAV_TO_POSE_BT = DEMO_DIR / "config" / "navigate_to_pose_no_spin.xml"
NAV_THROUGH_POSES_BT = DEMO_DIR / "config" / "navigate_through_poses_no_spin.xml"


class WarehouseSpeedParams(Substitution):
    """Apply one physically consistent, camera-friendly motion profile."""

    def __init__(self, source_file, max_linear_speed) -> None:
        super().__init__()
        self.source_file = normalize_to_list_of_substitutions(source_file)
        self.max_linear_speed = normalize_to_list_of_substitutions(max_linear_speed)

    def perform(self, context) -> str:
        source = perform_substitutions(context, self.source_file)
        speed = float(perform_substitutions(context, self.max_linear_speed))
        if not 0.0 < speed <= 1.5:
            raise ValueError("max_linear_speed must be in (0.0, 1.5] m/s")
        with Path(source).open(encoding="utf-8") as stream:
            params = yaml.safe_load(stream)
        controller_server = params["controller_server"]["ros__parameters"]
        controller = controller_server["FollowPath"]
        # Retain MPPI, which completes the multi-checkpoint mission reliably,
        # but make it follow the smoothed route farther ahead with much less
        # random yaw authority. This removes rapid counter-steering without
        # sacrificing dynamic-obstacle avoidance.
        controller.update({
            # Global smoothing already sees the entire route and bends the
            # blue line before each corner. Keep MPPI's local horizon compact
            # so it can run at cruise speed instead of optimizing distant
            # corners into an overly cautious near-zero command.
            "time_steps": 60,
            "model_dt": 0.05,
            "batch_size": 2000,
            "vx_max": speed,
            "vx_min": -0.10,
            # Match Gazebo Teleop / the four-wheel DiffDrive system's physical
            # angular limit. Path critics and low sampling noise prevent
            # over-steer; the controller must retain full authority to recover
            # cross-track error as quickly as manual control can.
            "wz_max": 1.80,
            "ax_max": 2.50,
            "ax_min": -2.50,
            "az_max": 4.00,
            "vx_std": 0.55,
            "wz_std": 0.15,
            "motion_model": "DiffDrive",
            "regenerate_noises": False,
            "temperature": 0.30,
            "gamma": 0.015,
            "visualize": False,
        })
        controller["PathAlignCritic"].update({
            "cost_weight": 18.0,
            # A crossing worker is handled by the stop gate. Never disable
            # cross-track alignment merely because that retained path is
            # temporarily occupied.
            "max_path_occupancy_ratio": 1.0,
            "trajectory_point_step": 3,
            "offset_from_furthest": 20,
            "use_path_orientations": False,
        })
        controller["PathFollowCritic"].update({
            # Target roughly 3 m along the 5 cm NavFn path. A one-metre target
            # made MPPI settle around 0.4–0.5 m/s even when vx_max was higher.
            "cost_weight": 12.0,
            "offset_from_furthest": 60,
        })
        controller["PathAngleCritic"].update({
            "cost_weight": 2.0,
            "offset_from_furthest": 35,
            "mode": 0,
        })
        controller["PreferForwardCritic"]["cost_weight"] = 6.0
        # NavFn paths are not kinematically constrained. If a dynamic event or
        # a tight corner leaves the differential base facing far away from the
        # path tangent, a short acceleration-limited shim realigns it before
        # MPPI resumes. This prevents MPPI settling below the smoother's
        # deadband while preserving MPPI for normal smooth path tracking.
        mppi = dict(controller)
        controller_server.pop("FollowPath")

        # Emit flat dotted parameter names. Nav2 Jazzy configures the internal
        # MPPI instance in the same FollowPath namespace as the shim, while
        # `primary_controller` remains the scalar plugin type.
        shim = {
            "plugin": "nav2_rotation_shim_controller::RotationShimController",
            "primary_controller": "nav2_mppi_controller::MPPIController",
            # Shim only aligns the initial route heading. Every later corner
            # stays under the long-horizon MPPI controller so the AGV drives a
            # continuous arc rather than stopping to rotate in place.
            "angular_dist_threshold": 0.48,
            "angular_disengage_threshold": 0.24,
            "forward_sampling_distance": 0.65,
            "rotate_to_heading_angular_vel": 1.20,
            "max_angular_accel": 4.00,
            "simulate_ahead_time": 0.80,
            "rotate_to_goal_heading": False,
            "rotate_to_heading_once": True,
            "closed_loop": True,
        }
        for key, value in shim.items():
            controller_server[f"FollowPath.{key}"] = value

        def flatten_parameter(prefix: str, value) -> None:
            if isinstance(value, dict):
                for child, child_value in value.items():
                    flatten_parameter(f"{prefix}.{child}", child_value)
            else:
                controller_server[prefix] = value

        for key, value in mppi.items():
            if key != "plugin":
                flatten_parameter(f"FollowPath.{key}", value)
        # A stopped dynamic-obstacle encounter is not navigation failure.
        # Give the worker time to clear without launching the aggressive
        # spin/backup recovery sequence visible in the camera feed.
        controller_server["progress_checker"].update({
            "required_movement_radius": 0.25,
            # Waiting for a crossing worker is intentional, not a navigation
            # failure. Keep the same plan while the stop zone is occupied.
            "movement_time_allowance": 120.0,
        })
        # If a person occupies every feasible trajectory, MPPI reports a
        # temporary control failure. Hold the action/path instead of allowing
        # the behavior tree to clear costmaps and compute another route.
        controller_server["failure_tolerance"] = 120.0
        controller_server["general_goal_checker"].update({
            # Coarse route goals hand off to camera servoing. Avoid a
            # 120-second deadband stall for the final few centimetres.
            "xy_goal_tolerance": 0.36,
            # Coarse Nav2 hands final shelf heading to the camera servo. Do not
            # spend 10–15 seconds rotating after distance_remaining is zero.
            "yaw_goal_tolerance": 0.60,
        })

        bt_navigator = params["bt_navigator"]["ros__parameters"]
        bt_navigator.update({
            "default_nav_to_pose_bt_xml": str(NAV_TO_POSE_BT),
            "default_nav_through_poses_bt_xml": str(NAV_THROUGH_POSES_BT),
        })

        smoother = params["velocity_smoother"]["ros__parameters"]
        smoother.update({
            "smoothing_frequency": 40.0,
            # Gazebo publishes odometry at 30 Hz, so closed-loop smoothing can
            # use measured chassis motion instead of assuming every command
            # was executed immediately.
            "feedback": "CLOSED_LOOP",
            "scale_velocities": True,
            "max_velocity": [speed, 0.0, 1.80],
            "min_velocity": [-0.10, 0.0, -1.80],
            "max_accel": [2.50, 0.0, 4.00],
            "max_decel": [-2.50, 0.0, -4.00],
            "deadband_velocity": [0.015, 0.0, 0.025],
            "velocity_timeout": 0.65,
            "odom_duration": 0.20,
        })
        path_smoother = params["smoother_server"]["ros__parameters"][
            "simple_smoother"
        ]
        path_smoother.update({
            "w_data": 0.20,
            "w_smooth": 0.55,
            "tolerance": 1.0e-8,
            "max_its": 1000,
            "do_refinement": True,
            "refinement_num": 2,
        })
        # The five worker trajectories are known by the dedicated Gazebo-pose
        # safety gate. Exclude their transient LiDAR marks from global A* so a
        # crossing person can never bend/recompute the blue route. The local
        # costmap and worker gate remain active for immediate collision safety.
        global_costmap = params["global_costmap"]["global_costmap"][
            "ros__parameters"
        ]
        global_costmap["obstacle_layer"]["enabled"] = False
        collision_monitor = params["collision_monitor"]["ros__parameters"]
        collision_monitor["cmd_vel_in_topic"] = "/cmd_vel_safety_input"
        # LiDAR obstacles are already represented in both Nav2 costmaps and
        # evaluated by MPPI's collision critic. The velocity-level footprint
        # predictor repeatedly alternated approach/normal beside static boxes,
        # which reset MPPI and produced the visible loop near Storage A.
        # Keep this lifecycle node as a transparent command bridge; only the
        # dedicated worker monitor may stop/resume the unchanged A* path.
        # Nav2 Jazzy cannot infer a parameter type from empty plugin/source
        # arrays. Retain the installed entries but disable the action polygon;
        # with no enabled polygon the node forwards every gated command.
        collision_monitor["polygons"] = ["FootprintApproach"]
        collision_monitor["observation_sources"] = ["scan"]
        collision_monitor["FootprintApproach"]["enabled"] = False
        collision_monitor["scan"]["enabled"] = False
        collision_monitor.pop("DynamicObstacleStop", None)
        output = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        yaml.safe_dump(params, output)
        output.close()
        return output.name


def generate_launch_description() -> LaunchDescription:
    use_rviz = LaunchConfiguration("use_rviz")
    localization_source = LaunchConfiguration("localization_source")
    max_linear_speed = LaunchConfiguration("max_linear_speed")
    source_params = LaunchConfiguration("params_file")
    base_params = RewrittenYaml(
        source_file=source_params,
        root_key="",
        param_rewrites={
            "use_sim_time": "True",
            # The warehouse AMR is wider than the TurtleBot in Nav2 defaults.
            "robot_radius": "0.26",
            # Navfn uses Dijkstra by default. This makes the global path an
            # explicit grid A* path while preserving Nav2's dynamic replanning.
            "use_astar": "True",
            # Clear departed workers from the global obstacle layer at 2 Hz
            # instead of waiting for the default one-second update.
            "global_costmap.global_costmap.ros__parameters.update_frequency": "2.0",
        },
        convert_types=True,
    )
    # RewrittenYaml cannot safely replace an individual list element in Jazzy,
    # so apply the scalar controller and list-valued smoother limits in one
    # small, validated second pass.
    configured_params = WarehouseSpeedParams(base_params, max_linear_speed)

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_rviz", default_value="True",
            description="Show map, LiDAR costmaps and the live Nav2 plan",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=str(NAV2_SHARE / "params" / "nav2_params.yaml"),
        ),
        DeclareLaunchArgument(
            "localization_source",
            default_value="vjepa",
            description="map->odom source: vjepa (demo) or ground_truth (reference test)",
        ),
        DeclareLaunchArgument(
            "max_linear_speed",
            default_value="1.0",
            description="Maximum forward Nav2 speed in m/s (model hard limit: 1.5)",
        ),
        Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="base_footprint_tf",
            arguments=[
                "--x", "0", "--y", "0", "--z", "0",
                "--yaw", "0", "--pitch", "0", "--roll", "0",
                "--frame-id", "base_link", "--child-frame-id", "base_footprint",
            ],
        ),
        Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="front_lidar_tf",
            arguments=[
                "--x", "-0.2063125", "--y", "0", "--z", "0.49994",
                "--yaw", "0", "--pitch", "0", "--roll", "0",
                "--frame-id", "base_link",
                "--child-frame-id", "warehouse_agv/lidar_link/lidar",
            ],
        ),
        Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="front_camera_tf",
            arguments=[
                "--x", "-0.1915625", "--y", "0", "--z", "0.47594",
                "--yaw", "0", "--pitch", "0", "--roll", "0",
                "--frame-id", "base_link",
                "--child-frame-id", "warehouse_agv/camera_link/front_camera",
            ],
        ),
        ExecuteProcess(
            cmd=[
                "python3", "-u",
                str(DEMO_DIR / "scripts" / "ground_truth_localizer.py"),
                "--ros-args", "-p", "use_sim_time:=true",
            ],
            condition=IfCondition(PythonExpression([
                "'", localization_source, "' == 'ground_truth'"
            ])),
            output="screen",
        ),
        ExecuteProcess(
            cmd=[
                "python3", "-u",
                str(DEMO_DIR / "scripts" / "vjepa_nav_localizer.py"),
                "--ros-args", "-p", "use_sim_time:=true",
            ],
            condition=IfCondition(PythonExpression([
                "'", localization_source, "' == 'vjepa'"
            ])),
            output="screen",
        ),
        Node(
            package="nav2_map_server", executable="map_server", name="map_server",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "yaml_filename": str(WAREHOUSE_MAP),
            }],
        ),
        # LifecycleServiceClient has a fixed 2 s response timeout. Under the
        # initial Gazebo/RViz/V-JEPA CPU burst, map_server can take almost that
        # long merely to create its services, causing configuration to finish
        # just after the client has disappeared. Start its manager only after
        # the server has had time to initialize; the transient-local /map is
        # then available before any mission readiness gate can pass.
        TimerAction(
            period=3.0,
            actions=[Node(
                package="nav2_lifecycle_manager", executable="lifecycle_manager",
                name="lifecycle_manager_map", output="screen",
                parameters=[{
                    "use_sim_time": True,
                    "autostart": True,
                    "node_names": ["map_server"],
                }],
            )],
        ),
        ExecuteProcess(
            cmd=[
                "python3", "-u",
                str(DEMO_DIR / "scripts" / "person_safety_monitor.py"),
                "--ros-args", "-p", "use_sim_time:=true",
            ],
            output="screen",
        ),
        ExecuteProcess(
            cmd=["python3", "-u", str(DEMO_DIR / "scripts" / "keyboard_cmd_mux.py")],
            output="screen",
        ),
        # Planner configuration subscribes to the transient-local map and can
        # block its lifecycle response until /map exists. Starting navigation
        # concurrently with the delayed map manager made planner_server miss
        # LifecycleServiceClient's fixed timeout, leaving bt_navigator forever
        # unconfigured. Bring the map up first, then launch navigation.
        TimerAction(
            period=6.0,
            actions=[IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(NAV2_SHARE / "launch" / "navigation_launch.py")
                ),
                launch_arguments={
                    "use_sim_time": "True",
                    "autostart": "True",
                    "params_file": configured_params,
                    "use_composition": "False",
                }.items(),
            )],
        ),
        Node(
            package="rviz2", executable="rviz2", name="rviz2", output="screen",
            condition=IfCondition(use_rviz),
            arguments=["-d", str(NAV2_SHARE / "rviz" / "nav2_default_view.rviz")],
            parameters=[{"use_sim_time": True}],
        ),
    ])
