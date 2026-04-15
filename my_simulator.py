"""Drake-based robot simulator: scene builder and simulation API."""

import atexit
import multiprocessing
import os
import time
import uuid

import numpy as np

from pydrake.geometry import (
    AddCompliantHydroelasticProperties,
    AddContactMaterial,
    AddRigidHydroelasticProperties,
    Box,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    ProximityProperties,
    Rgba,
    Sphere,
)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
)
from pydrake.multibody.tree import (
    PdControllerGains,
    SpatialInertia,
    UnitInertia,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource
from pydrake.common.eigen_geometry import Quaternion

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------
TABLE_WIDTH = 1.0
TABLE_DEPTH = 1.0
TABLE_HEIGHT = 1.0

ROBOT_BASE_POSITION = np.array([0.0, 0.0, TABLE_HEIGHT])

BLOCK_SIZE = 0.06
BLOCK_MASS = 0.1

BLOCK_COLORS = {
    "red": Rgba(1.0, 0.0, 0.0, 1.0),
    "blue": Rgba(0.0, 0.0, 1.0, 1.0),
}

BLOCK_NOMINAL_POSITIONS = [
    [0.4, -0.10],
    [0.4,  0.10],
]
BLOCK_XY_NOISE = 0.02
BLOCK_DROP_HEIGHT = 0.05

CAMERA_POSITION = np.array([0.5, 0.0, 1.2])
CAMERA_QUATERNION_XYZW = np.array([-0.3420, 0.0, 0.0, 0.9397])

ARM_KP, ARM_KD = 10000.0, 500.0
GRIPPER_KP, GRIPPER_KD = 500.0, 100.0

ARM_MOVE_DURATION = 2.0
GRIPPER_MOVE_DURATION = 0.5
GRIPPER_OPEN_WIDTH = 0.04
GRIPPER_CLOSED_WIDTH = 0.0

TCP_VISUAL_OFFSET_X = 0.145

TABLE_HYDRO_MODULUS = 1e7
BLOCK_HYDRO_MODULUS = 5e6
HC_DISSIPATION = 2.0
FRICTION_COEFF = 0.9

SIM_STEP = 0.05
PLANT_TIME_STEP = 0.002
COMMAND_TIMEOUT = 10.0
JOINT_CONVERGENCE_TOLERANCE = 0.02
SETTLING_TIMEOUT = 2.0


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------

def _build_scene(meshcat):
    """Build the Drake scene with UR3e, WSG-50, table, and colored blocks.

    Parameters
    ----------
    meshcat : Meshcat
        A Meshcat instance used for visualisation.

    Returns
    -------
    diagram : Diagram
    plant : MultibodyPlant
    ur_model : ModelInstanceIndex
    wsg_model : ModelInstanceIndex
    block_models : list of (ModelInstanceIndex, RigidBody, str)
        Each entry is (model_instance, body, color_name).
    """
    builder = DiagramBuilder()

    # Create plant + scene graph
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=PLANT_TIME_STEP
    )
    parser = Parser(plant)
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    # ------------------------------------------------------------------
    # Load UR3e
    # ------------------------------------------------------------------
    ur_models = parser.AddModels(
        os.path.join(models_dir, "ur_description", "urdf", "ur3e_cylinders_collision.urdf")
    )
    ur_model = ur_models[0]

    # Weld UR3e base to world at table height
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("ur_base_link", ur_model),
        RigidTransform(ROBOT_BASE_POSITION),
    )

    # ------------------------------------------------------------------
    # Load WSG-50 and weld to UR3e end-effector
    # ------------------------------------------------------------------
    wsg_models = parser.AddModels(
        os.path.join(models_dir, "wsg_50_description", "urdf", "schunk_wsg_50_with_tip.urdf")
    )
    wsg_model = wsg_models[0]

    plant.WeldFrames(
        plant.GetFrameByName("ur_ee_link", ur_model),
        plant.GetFrameByName("body", wsg_model),
        RigidTransform(RotationMatrix.MakeZRotation(-np.pi / 2), [0.036, 0, 0]),
    )

    # ------------------------------------------------------------------
    # Ground plane — registered on the world body
    # ------------------------------------------------------------------
    world_body = plant.world_body()
    ground_shape = Box(10.0, 10.0, 0.1)
    ground_pose = RigidTransform([0.0, 0.0, -0.05])

    plant.RegisterVisualGeometry(
        world_body,
        ground_pose,
        ground_shape,
        "ground_visual",
        np.array([0.35, 0.35, 0.35, 1.0]),
    )

    ground_contact = ProximityProperties()
    AddContactMaterial(
        ground_contact, dissipation=HC_DISSIPATION, friction=CoulombFriction(FRICTION_COEFF, FRICTION_COEFF)
    )
    AddRigidHydroelasticProperties(0.1, ground_contact)
    plant.RegisterCollisionGeometry(
        world_body, ground_pose, ground_shape, "ground_collision", ground_contact
    )

    # ------------------------------------------------------------------
    # Static table — registered on the world body
    # ------------------------------------------------------------------
    table_shape = Box(TABLE_WIDTH, TABLE_DEPTH, TABLE_HEIGHT)
    table_pose = RigidTransform([0.0, 0.0, TABLE_HEIGHT/ 2])

    plant.RegisterVisualGeometry(
        world_body,
        table_pose,
        table_shape,
        "table_visual",
        np.array([0.5, 0.4, 0.3, 1.0]),
    )

    table_contact = ProximityProperties()
    AddContactMaterial(
        table_contact, dissipation=HC_DISSIPATION, friction=CoulombFriction(FRICTION_COEFF, FRICTION_COEFF)
    )
    AddRigidHydroelasticProperties(0.05, table_contact)
    plant.RegisterCollisionGeometry(
        world_body, table_pose, table_shape, "table_collision", table_contact
    )

    # ------------------------------------------------------------------
    # Colored free blocks
    # ------------------------------------------------------------------
    block_shape = Box(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    block_spatial_inertia = SpatialInertia(
        mass=BLOCK_MASS,
        p_PScm_E=np.zeros(3),
        G_SP_E=UnitInertia.SolidBox(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
    )
    block_contact = ProximityProperties()
    AddContactMaterial(block_contact, dissipation=HC_DISSIPATION, friction=CoulombFriction(FRICTION_COEFF, FRICTION_COEFF))
    AddCompliantHydroelasticProperties(0.005, BLOCK_HYDRO_MODULUS, block_contact)

    block_models = []
    color_items = list(BLOCK_COLORS.items())
    for i, (color_name, color_rgba) in enumerate(color_items):
        model_instance = plant.AddModelInstance(f"block_{color_name}")
        body = plant.AddRigidBody(
            f"block_{color_name}", model_instance, block_spatial_inertia
        )

        # Visual geometry
        plant.RegisterVisualGeometry(
            body,
            RigidTransform(),
            block_shape,
            f"block_{color_name}_visual",
            np.array([color_rgba.r(), color_rgba.g(), color_rgba.b(), color_rgba.a()]),
        )

        # Collision geometry (shared ProximityProperties is fine — Drake copies it)
        plant.RegisterCollisionGeometry(
            body,
            RigidTransform(),
            block_shape,
            f"block_{color_name}_collision",
            block_contact,
        )

        # Randomized initial pose: jittered XY, dropped from above, random orientation
        rng = np.random.default_rng()
        nominal = BLOCK_NOMINAL_POSITIONS[i]
        x = nominal[0] + rng.uniform(-BLOCK_XY_NOISE, BLOCK_XY_NOISE)
        y = nominal[1] + rng.uniform(-BLOCK_XY_NOISE, BLOCK_XY_NOISE)
        z = TABLE_HEIGHT + BLOCK_SIZE / 2 + BLOCK_DROP_HEIGHT
        random_quat = rng.standard_normal(4)
        random_quat /= np.linalg.norm(random_quat)
        q = Quaternion(w=random_quat[3], x=random_quat[0], y=random_quat[1], z=random_quat[2])
        plant.SetDefaultFloatingBaseBodyPose(
            body, RigidTransform(RotationMatrix(q), [x, y, z])
        )

        block_models.append((model_instance, body, color_name))

    # ------------------------------------------------------------------
    # TCP visual indicator — visual-only sphere on ur_ee_link x-axis
    # ------------------------------------------------------------------
    ur_ee_body = plant.GetBodyByName("ur_ee_link", ur_model)
    plant.RegisterVisualGeometry(
        ur_ee_body,
        RigidTransform([TCP_VISUAL_OFFSET_X, 0.0, 0.0]),
        Sphere(0.001),
        "tcp_visual",
        np.array([0.0, 1.0, 1.0, 0]),  # cyan, transparent. Make it visible if you need to debug TCP point
    )

    # ------------------------------------------------------------------
    # Finalize plant
    # ------------------------------------------------------------------
    plant.Finalize()

    # ------------------------------------------------------------------
    # PD controller gains (must be set after Finalize)
    # ------------------------------------------------------------------
    for idx in plant.GetJointActuatorIndices(ur_model):
        act = plant.get_joint_actuator(idx)
        act.set_controller_gains(PdControllerGains(p=ARM_KP, d=ARM_KD))

    for idx in plant.GetJointActuatorIndices(wsg_model):
        act = plant.get_joint_actuator(idx)
        act.set_controller_gains(PdControllerGains(p=GRIPPER_KP, d=GRIPPER_KD))

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(show_hydroelastic=False),
    )

    # Meshcat cosmetics
    meshcat.SetProperty("/Background", "top_color", [0.1, 0.1, 0.15])
    meshcat.SetProperty("/Background", "bottom_color", [0.25, 0.25, 0.3])
    meshcat.SetProperty("/Grid", "visible", False)
    meshcat.SetProperty("/Axes", "visible", False)

    # ------------------------------------------------------------------
    # Desired-state input sources (ConstantVectorSource as placeholders)
    # ------------------------------------------------------------------
    # UR: 6 joints × 2 (position + velocity) = 12
    ur_desired_state_src = builder.AddSystem(
        ConstantVectorSource(np.zeros(12))
    )
    ur_desired_state_src.set_name("ur_desired_state")
    builder.Connect(
        ur_desired_state_src.get_output_port(),
        plant.get_desired_state_input_port(ur_model),
    )

    # WSG: 2 joints × 2 (position + velocity) = 4
    wsg_desired_state_src = builder.AddSystem(
        ConstantVectorSource(np.zeros(4))
    )
    wsg_desired_state_src.set_name("wsg_desired_state")
    builder.Connect(
        wsg_desired_state_src.get_output_port(),
        plant.get_desired_state_input_port(wsg_model),
    )

    # ------------------------------------------------------------------
    # Build diagram
    # ------------------------------------------------------------------
    diagram = builder.Build()

    return diagram, plant, ur_model, wsg_model, block_models


# ---------------------------------------------------------------------------
# Desired-state setters
# ---------------------------------------------------------------------------

def _update_gripper_action(gripper_action, sim_time, diagram, context, result_queue):
    """Advance gripper action. Returns updated action (or None if done)."""
    target = gripper_action["target_width"]
    _set_wsg_desired_state(diagram, context, np.array([-target, target]))
    if sim_time >= gripper_action["end_time"]:
        result_queue.put((gripper_action["request_id"], "done"))
        return None
    return gripper_action


def _update_arm_trajectory(trajectory, sim_time, plant, plant_context, ur_model, diagram, context, result_queue):
    """Advance arm trajectory interpolation. Returns updated trajectory (or None if done)."""
    alpha = min(1.0, (sim_time - trajectory["start_time"]) / trajectory["duration"])
    desired = (1 - alpha) * trajectory["start_positions"] + alpha * trajectory["end_positions"]
    _set_ur_desired_state(diagram, context, desired)
    if alpha >= 1.0:
        current = plant.GetPositions(plant_context, ur_model)
        error = np.max(np.abs(current - trajectory["end_positions"]))
        settling_elapsed = sim_time - trajectory["end_time"]
        if error < JOINT_CONVERGENCE_TOLERANCE or settling_elapsed > SETTLING_TIMEOUT:
            result_queue.put((trajectory["request_id"], "done"))
            return None
    return trajectory


def _set_ur_desired_state(diagram, root_context, desired_positions):
    """Set the UR arm desired state (positions + zero velocities)."""
    source = diagram.GetSubsystemByName("ur_desired_state")
    source_context = diagram.GetMutableSubsystemContext(source, root_context)
    desired_state = np.concatenate([desired_positions, np.zeros_like(desired_positions)])
    source.get_mutable_source_value(source_context).set_value(desired_state)


def _set_wsg_desired_state(diagram, root_context, desired_positions):
    """Set the WSG gripper desired state (positions + zero velocities)."""
    source = diagram.GetSubsystemByName("wsg_desired_state")
    source_context = diagram.GetMutableSubsystemContext(source, root_context)
    desired_state = np.concatenate([desired_positions, np.zeros_like(desired_positions)])
    source.get_mutable_source_value(source_context).set_value(desired_state)


# ---------------------------------------------------------------------------
# Block pose queries
# ---------------------------------------------------------------------------

def _query_block_poses(plant, plant_context, block_models):
    """Return block poses in world frame."""
    poses = []
    for model_instance, body, color_name in block_models:
        X_WB = plant.GetFreeBodyPose(plant_context, body)
        pos = X_WB.translation().tolist()
        q_wxyz = X_WB.rotation().ToQuaternion()
        # Drake returns w,x,y,z — convert to x,y,z,w
        quat_xyzw = [q_wxyz.x(), q_wxyz.y(), q_wxyz.z(), q_wxyz.w()]
        poses.append({
            "name": f"{color_name}_block",
            "color": color_name,
            "position": pos,
            "quaternion": quat_xyzw,
        })
    return poses


def _query_block_poses_in_camera_frame(plant, plant_context, block_models):
    """Return block poses expressed in camera frame."""
    # Build camera transform from constants
    xyzw = CAMERA_QUATERNION_XYZW
    norm = np.linalg.norm(xyzw)
    xyzw = xyzw / norm
    cam_quat = Quaternion(w=float(xyzw[3]), x=float(xyzw[0]),
                          y=float(xyzw[1]), z=float(xyzw[2]))
    X_WC = RigidTransform(RotationMatrix(cam_quat), CAMERA_POSITION)
    X_CW = X_WC.inverse()

    poses = []
    for model_instance, body, color_name in block_models:
        X_WB = plant.GetFreeBodyPose(plant_context, body)
        X_CB = X_CW @ X_WB
        pos = X_CB.translation().tolist()
        q_wxyz = X_CB.rotation().ToQuaternion()
        quat_xyzw = [q_wxyz.x(), q_wxyz.y(), q_wxyz.z(), q_wxyz.w()]
        poses.append({
            "name": f"block_{color_name}",
            "color": color_name,
            "position": pos,
            "quaternion": quat_xyzw,
        })
    return poses


# ---------------------------------------------------------------------------
# Main simulation entry point (child process target)
# ---------------------------------------------------------------------------

def _run_simulation(cmd_queue, result_queue, visualizer_port):
    """Target function for the simulation child process."""
    import queue as _queue_mod

    try:
        meshcat = Meshcat(visualizer_port)
        diagram, plant, ur_model, wsg_model, block_models = _build_scene(meshcat)

        simulator = Simulator(diagram)
        context = simulator.get_mutable_context()
        plant_context = plant.GetMyMutableContextFromRoot(context)

        # Set initial arm positions to a home pose (arm pointing upward, clear of table)
        initial_arm_positions = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
        plant.SetPositions(plant_context, ur_model, initial_arm_positions)

        # Initialize desired states to match initial positions
        _set_ur_desired_state(diagram, context, initial_arm_positions)
        _set_wsg_desired_state(diagram, context, np.zeros(plant.num_positions(wsg_model)))

        simulator.Initialize()

        # Active motion state
        trajectory = None
        gripper_action = None

        sim_time = 0.0
        running = True

        # Main sim loop: process commands, update trajectories, advance sim, repeat
        while running:
            # Poll command queue (non-blocking)
            while True:
                try:
                    cmd_name, args, request_id = cmd_queue.get_nowait()
                except _queue_mod.Empty:
                    break

                if cmd_name == "stop":
                    running = False
                    result_queue.put((request_id, "stopped"))
                    break
                elif cmd_name == "move_arm":
                    current = plant.GetPositions(plant_context, ur_model)
                    target = np.array(args["positions"], dtype=float)
                    move_duration = args.get("duration", ARM_MOVE_DURATION)
                    trajectory = {
                        "start_positions": current.copy(),
                        "end_positions": target,
                        "start_time": sim_time,
                        "end_time": sim_time + move_duration,
                        "duration": move_duration,
                        "request_id": request_id,
                    }
                elif cmd_name == "open_gripper":
                    gripper_action = {
                        "target_width": GRIPPER_OPEN_WIDTH,
                        "start_time": sim_time,
                        "end_time": sim_time + GRIPPER_MOVE_DURATION,
                        "request_id": request_id,
                    }
                elif cmd_name == "close_gripper":
                    gripper_action = {
                        "target_width": GRIPPER_CLOSED_WIDTH,
                        "start_time": sim_time,
                        "end_time": sim_time + GRIPPER_MOVE_DURATION,
                        "request_id": request_id,
                    }
                elif cmd_name == "get_gripper_open_status":
                    wsg_positions = plant.GetPositions(plant_context, wsg_model)
                    total_width = wsg_positions[1] - wsg_positions[0]
                    is_open = total_width > GRIPPER_OPEN_WIDTH * 0.1
                    result_queue.put((request_id, is_open))
                elif cmd_name == "get_arm_joint_positions":
                    positions = plant.GetPositions(plant_context, ur_model).tolist()
                    result_queue.put((request_id, positions))
                elif cmd_name == "get_block_poses":
                    result_queue.put((request_id, _query_block_poses(plant, plant_context, block_models)))
                elif cmd_name == "get_block_poses_camera_frame":
                    result_queue.put((request_id, _query_block_poses_in_camera_frame(plant, plant_context, block_models)))
                elif cmd_name == "reset":
                    # Reset arm to home
                    plant.SetPositions(plant_context, ur_model, initial_arm_positions)
                    plant.SetVelocities(plant_context, ur_model, np.zeros(plant.num_velocities(ur_model)))
                    _set_ur_desired_state(diagram, context, initial_arm_positions)
                    # Reset gripper to closed
                    plant.SetPositions(plant_context, wsg_model, np.zeros(plant.num_positions(wsg_model)))
                    plant.SetVelocities(plant_context, wsg_model, np.zeros(plant.num_velocities(wsg_model)))
                    _set_wsg_desired_state(diagram, context, np.zeros(plant.num_positions(wsg_model)))
                    # Cancel active motions
                    trajectory = None
                    gripper_action = None
                    # Re-sample block poses
                    rng = np.random.default_rng()
                    for i, (blk_model, blk_body, _color) in enumerate(block_models):
                        nominal = BLOCK_NOMINAL_POSITIONS[i]
                        bx = nominal[0] + rng.uniform(-BLOCK_XY_NOISE, BLOCK_XY_NOISE)
                        by = nominal[1] + rng.uniform(-BLOCK_XY_NOISE, BLOCK_XY_NOISE)
                        bz = TABLE_HEIGHT + BLOCK_SIZE / 2 + BLOCK_DROP_HEIGHT
                        rq = rng.standard_normal(4)
                        rq /= np.linalg.norm(rq)
                        bq = Quaternion(w=float(rq[3]), x=float(rq[0]), y=float(rq[1]), z=float(rq[2]))
                        plant.SetFreeBodyPose(
                            plant_context, blk_body,
                            RigidTransform(RotationMatrix(bq), [bx, by, bz]),
                        )
                        plant.SetVelocities(plant_context, blk_model,
                                            np.zeros(plant.num_velocities(blk_model)))
                    result_queue.put((request_id, "done"))
                else:
                    result_queue.put((request_id, {"error": f"Unknown command: {cmd_name}"}))

            if not running:
                break

            # Update arm trajectory
            if trajectory is not None:
                trajectory = _update_arm_trajectory(
                    trajectory, sim_time, plant, plant_context, ur_model,
                    diagram, context, result_queue,
                )

            # Update gripper
            if gripper_action is not None:
                gripper_action = _update_gripper_action(
                    gripper_action, sim_time, diagram, context, result_queue,
                )

            # Advance sim (real-time throttle)
            wall_start = time.monotonic()
            sim_time += SIM_STEP
            simulator.AdvanceTo(sim_time)
            elapsed = time.monotonic() - wall_start
            sleep_time = SIM_STEP - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        # Send error back so the notebook gets a useful message instead of a timeout
        result_queue.put(("__fatal__", {"error": str(e)}))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MySimulator:
    """Drake-based robot simulator with multiprocessing IPC.

    Only one instance may exist at a time. Creating a second instance
    raises RuntimeError.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is not None:
            raise RuntimeError(
                "Only one MySimulator instance is allowed. "
                "Call stop_simulation() and del the existing instance first."
            )
        instance = super().__new__(cls)
        cls._instance = instance
        return instance

    def __init__(self, visualizer_port=7000):
        """Initialize the simulator (does not start the child process).

        Parameters
        ----------
        visualizer_port : int, optional
            TCP port on which the Meshcat web visualizer will listen.
            Default is 7000; visit ``http://localhost:<port>`` in a browser
            after calling :meth:`start_simulation`.

        Raises
        ------
        RuntimeError
            If a ``MySimulator`` instance already exists (singleton guard).
        """
        self._visualizer_port = visualizer_port
        self._process = None
        self._cmd_queue = None
        self._result_queue = None
        atexit.register(self.stop_simulation)

    def __del__(self):
        if MySimulator._instance is self:
            MySimulator._instance = None

    def _send_command(self, cmd_name, args=None, timeout=COMMAND_TIMEOUT):
        """Send a command to the sim process and wait for result."""
        if self._process is None or not self._process.is_alive():
            raise RuntimeError("Simulation is not running.")
        request_id = uuid.uuid4().hex
        self._cmd_queue.put((cmd_name, args, request_id))
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"Command '{cmd_name}' timed out after {timeout}s")
            result_id, result = self._result_queue.get(timeout=remaining)
            if result_id == request_id:
                return result

    def start_simulation(self):
        """Start the Drake simulation in a child process.

        Spawns a subprocess that builds the scene (UR3e + WSG-50 + table +
        colored blocks), initialises the Drake ``Simulator``, and runs the
        real-time control loop.  A Meshcat visualizer is also started; open
        the returned URL in a browser to watch the simulation.
        """
        if self._process is not None and self._process.is_alive():
            raise RuntimeError("Simulation is already running.")
        self._cmd_queue = multiprocessing.Queue()
        self._result_queue = multiprocessing.Queue()
        self._process = multiprocessing.Process(
            target=_run_simulation,
            args=(self._cmd_queue, self._result_queue, self._visualizer_port),
            daemon=True,
        )
        self._process.start()
        time.sleep(1.0)  # let Meshcat start
        # Check if the child process crashed during startup
        try:
            result_id, result = self._result_queue.get_nowait()
            if result_id == "__fatal__":
                self._process.join(timeout=2.0)
                raise RuntimeError(f"Simulation failed to start: {result.get('error', result)}")
        except Exception as e:
            if "Simulation failed to start" in str(e):
                raise
        return f"http://localhost:{self._visualizer_port}"

    def stop_simulation(self):
        """Stop the simulation process."""
        if self._process is None:
            return
        if self._process.is_alive():
            try:
                self._send_command("stop", timeout=5.0)
            except (TimeoutError, RuntimeError):
                pass
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
        self._process = None
        self._cmd_queue = None
        self._result_queue = None
        MySimulator._instance = None

    def get_block_poses_in_camera_frame(self):
        """Get all block poses in virtual camera frame.
        Returns list of dicts: name, color, position [x,y,z], quaternion [x,y,z,w]."""
        return self._send_command("get_block_poses_camera_frame")

    def get_camera_extrinsics(self):
        """Get virtual camera pose in world frame (constants).
        Returns dict: position [x,y,z], quaternion [x,y,z,w]."""
        quat = CAMERA_QUATERNION_XYZW / np.linalg.norm(CAMERA_QUATERNION_XYZW)
        return {
            "position": CAMERA_POSITION.tolist(),
            "quaternion": quat.tolist(),
        }

    def get_robot_base_pose_in_world_frame(self):
        """Get UR3e base pose in world frame (constants).
        Returns dict: position [x,y,z], quaternion [x,y,z,w]."""
        return {
            "position": ROBOT_BASE_POSITION.tolist(),
            "quaternion": [0.0, 0.0, 0.0, 1.0],
        }

    def get_arm_joint_positions(self):
        """Get current arm joint positions in radians. Returns list of 6 floats."""
        return self._send_command("get_arm_joint_positions")

    def move_arm_to_joint_positions(self, positions, duration=ARM_MOVE_DURATION):
        """Move arm to target joint positions via linear interpolation.
        Blocks until motion completes.
        Args:
            positions — list of 6 joint angles in radians.
            duration — movement duration in seconds (default 2.0)."""
        if len(positions) != 6:
            raise ValueError(f"Expected 6 joint positions, got {len(positions)}")
        return self._send_command("move_arm", {"positions": positions, "duration": duration}, timeout=duration + COMMAND_TIMEOUT)

    def open_gripper(self):
        """Open the WSG-50 gripper. Blocks until done (~0.5s)."""
        return self._send_command("open_gripper", timeout=GRIPPER_MOVE_DURATION + COMMAND_TIMEOUT)

    def close_gripper(self):
        """Close the WSG-50 gripper. Blocks until done (~0.5s)."""
        return self._send_command("close_gripper", timeout=GRIPPER_MOVE_DURATION + COMMAND_TIMEOUT)

    def get_gripper_open_status(self):
        """Check whether the gripper is open.
        Returns True if open, False if closed."""
        return self._send_command("get_gripper_open_status")

    def reset(self):
        """Reset the simulation without restarting the process.
        Arm returns to home pose, gripper closes, and blocks are re-sampled
        at randomized positions. Blocks until complete."""
        return self._send_command("reset", timeout=COMMAND_TIMEOUT)

    def get_gripper_wrist_offset(self):
        """Get the pose of the gripper center relative to the wrist tool frame (ur_ee_link).
        The gripper center is the midpoint between the two fingers when closed.
        Returns dict: position [x,y,z] in metres, quaternion [x,y,z,w].
        This is a fixed constant derived from the WSG-50 SDF and weld transform."""
        return {
            "position": [0.145, 0.0, 0.0],
            "quaternion": [0.0, 0.7071, 0.0, 0.7071],  # Ry(+90°) in [x,y,z,w]
        }