"""Mujoco Client: This node is used to represent simulated robot, it can be used to read virtual positions, or can be controlled."""

import argparse
import os
import platform
import sys
import time
from pathlib import Path


def _reexec_under_mjpython() -> None:
    """
    On macOS, mujoco.viewer.launch_passive must run under `mjpython` (it needs
    the GUI event loop on the main thread). dora launches this node with plain
    `python`, so re-exec into mjpython. os.execv keeps the same PID, fds and env
    vars, so dora's node connection is preserved.
    """
    if platform.system() != "Darwin" or os.environ.get("_MJP_REEXEC"):
        return
    # must be the mjpython from THIS interpreter's env (not one found on PATH),
    # otherwise we'd jump out of the node's isolated uv env into another venv
    mjpython = os.path.join(os.path.dirname(sys.executable), "mjpython")
    if not os.path.exists(mjpython):
        return
    # uv's standalone python ships libpython in LIBDIR but not on mjpython's
    # @rpath, so mjpython can't dlopen it. Add LIBDIR to the dyld search path.
    import sysconfig

    libdir = sysconfig.get_config_var("LIBDIR")
    if libdir:
        existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
            f"{libdir}:{existing}" if existing else libdir
        )
    os.environ["_MJP_REEXEC"] = "1"
    os.execv(mjpython, [mjpython, *sys.argv])


_reexec_under_mjpython()

import mink
import mujoco
import mujoco.viewer
import numpy as np
import pyarrow as pa
from dora import Node
from loop_rate_limiters import RateLimiter

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent


class Client:
    """TODO: Add docstring."""

    def __init__(self, mode="pos"):
        """TODO: Add docstring."""

        self.model = mujoco.MjModel.from_xml_path(
            f"{ROOT_PATH}/AHSimulation/AH_Right/mjcf/scene.xml"
        )
        # self.data=mujoco.MjData(self.model)

        self.configuration = mink.Configuration(self.model)

        self.posture_task = mink.PostureTask(self.model, cost=1e-2)

        if mode == "pos":
            self.task1 = mink.FrameTask(
                frame_name="tip1",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=0.0,
                lm_damping=1.0,
            )

            self.task2 = mink.FrameTask(
                frame_name="tip2",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=0.0,
                lm_damping=1.0,
            )

            self.task3 = mink.FrameTask(
                frame_name="tip3",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=0.0,
                lm_damping=1.0,
            )

            self.task4 = mink.FrameTask(
                frame_name="tip4",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=0.0,
                lm_damping=1.0,
            )
        elif mode == "quat":  # we control the orientation
            self.task1 = mink.FrameTask(
                frame_name="tip1",
                frame_type="site",
                position_cost=0.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            )

            self.task2 = mink.FrameTask(
                frame_name="tip2",
                frame_type="site",
                position_cost=0.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            )

            self.task3 = mink.FrameTask(
                frame_name="tip3",
                frame_type="site",
                position_cost=0.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            )

            self.task4 = mink.FrameTask(
                frame_name="tip4",
                frame_type="site",
                position_cost=0.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            )
        else:
            print(f"Error, unknown mode: {mode}")
            return -1
        # Regulate all equality constraints with the same cost.
        eq_task = mink.EqualityConstraintTask(self.model, cost=1000.0)

        self.tasks = [
            eq_task,
            self.posture_task,
            self.task1,
            self.task2,
            self.task3,
            self.task4,
        ]

        self.model = self.configuration.model
        self.data = self.configuration.data
        # the repo env only ships some qpsolvers backends; prefer quadprog (what
        # this demo was written for) but fall back to whatever is installed
        # (e.g. daqp) so a missing solver doesn't crash the node.
        import qpsolvers

        available = qpsolvers.available_solvers
        self.solver = next(
            (s for s in ("quadprog", "daqp", "osqp") if s in available),
            available[0],
        )

        self.motor_pos = []
        self.metadata = []
        self.node = Node()

    def run(self):
        """TODO: Add docstring."""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            rate = RateLimiter(frequency=1000.0)
            # dt = rate.dt
            # t = 0
            self.configuration.update_from_keyframe("zero")

            # Initialize mocap bodies at their respective sites.
            self.posture_task.set_target_from_configuration(self.configuration)

            mink.move_mocap_to_frame(
                self.model, self.data, "finger1_target", "tip1", "site"
            )
            mink.move_mocap_to_frame(
                self.model, self.data, "finger2_target", "tip2", "site"
            )
            mink.move_mocap_to_frame(
                self.model, self.data, "finger3_target", "tip3", "site"
            )
            mink.move_mocap_to_frame(
                self.model, self.data, "finger4_target", "tip4", "site"
            )

            for event in self.node:
                event_type = event["type"]

                if event_type == "INPUT":
                    event_id = event["id"]

                    if event_id == "tick":
                        # self.node.send_output("tick", pa.array([]), event["metadata"])

                        if not viewer.is_running():
                            break

                        step_start = time.time()

                        self.task1.set_target(
                            mink.SE3.from_mocap_name(
                                self.model, self.data, "finger1_target"
                            )
                        )
                        self.task2.set_target(
                            mink.SE3.from_mocap_name(
                                self.model, self.data, "finger2_target"
                            )
                        )
                        self.task3.set_target(
                            mink.SE3.from_mocap_name(
                                self.model, self.data, "finger3_target"
                            )
                        )
                        self.task4.set_target(
                            mink.SE3.from_mocap_name(
                                self.model, self.data, "finger4_target"
                            )
                        )

                        # vel = mink.solve_ik(self.configuration, self.tasks, self.model.opt.timestep, self.solver, 1e-5)
                        # self.configuration.integrate_inplace(vel, self.model.opt.timestep)
                        # When the hand isn't in a valid pose (e.g. not visible),
                        # the IK targets are degenerate and the QP goes NaN or the
                        # solver fails. Skip those frames entirely so the hand just
                        # holds its last pose instead of crashing / jerking. The
                        # errstate silences the divide-by-zero warning spam.
                        vel = None
                        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                            try:
                                vel = mink.solve_ik(
                                    self.configuration,
                                    self.tasks,
                                    rate.dt,
                                    self.solver,
                                    1e-5,
                                )
                            except Exception:
                                vel = None
                        if vel is not None and np.all(np.isfinite(vel)):
                            self.configuration.integrate_inplace(vel, rate.dt)

                        # Step the simulation forward
                        # mujoco.mj_step(self.m, self.data)

                        # mujoco.mj_step(self.model, self.data)

                        # get the motors position and send for real motor control

                        f1_motor1 = mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger1_motor1"
                        )
                        f1_motor2 = mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger1_motor2"
                        )
                        f2_motor1 = mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger2_motor1"
                        )
                        f2_motor2 = mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger2_motor2"
                        )
                        f3_motor1 = mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger3_motor1"
                        )
                        f3_motor2 = mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger3_motor2"
                        )
                        f4_motor1 = mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger4_motor1"
                        )
                        f4_motor2 = mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger4_motor2"
                        )
                        # print(f"motor1: {self.data.joint(f1_motor1).qpos} motor2: {self.data.joint(f1_motor2).qpos}")
                        self.metadata = event["metadata"]
                        self.metadata["r_finger1"] = [0, 1]
                        self.metadata["r_finger2"] = [2, 3]
                        self.metadata["r_finger3"] = [4, 5]
                        self.metadata["r_finger4"] = [6, 7]

                        self.motor_pos = np.zeros(8)
                        self.motor_pos[self.metadata["r_finger1"]] = np.array(
                            [
                                self.data.joint(f1_motor1).qpos[0],
                                self.data.joint(f1_motor2).qpos[0],
                            ]
                        )
                        self.motor_pos[self.metadata["r_finger2"]] = np.array(
                            [
                                self.data.joint(f2_motor1).qpos[0],
                                self.data.joint(f2_motor2).qpos[0],
                            ]
                        )
                        self.motor_pos[self.metadata["r_finger3"]] = np.array(
                            [
                                self.data.joint(f3_motor1).qpos[0],
                                self.data.joint(f3_motor2).qpos[0],
                            ]
                        )
                        self.motor_pos[self.metadata["r_finger4"]] = np.array(
                            [
                                self.data.joint(f4_motor1).qpos[0],
                                self.data.joint(f4_motor2).qpos[0],
                            ]
                        )

                        viewer.sync()

                        # Rudimentary time keeping, will drift relative to wall clock.
                        time_until_next_step = self.model.opt.timestep - (
                            time.time() - step_start
                        )
                        if time_until_next_step > 0:
                            time.sleep(time_until_next_step)

                    elif event_id == "pull_position":
                        self.pull_position(self.node, event["metadata"])

                    elif event_id == "tick_ctrl":
                        if len(self.metadata) > 0:
                            self.node.send_output(
                                "mj_r_joints_pos",
                                pa.array(self.motor_pos),
                                self.metadata,
                            )
                        # self.pull_position(self.node, event["metadata"])

                    elif event_id == "pull_velocity":
                        self.pull_velocity(self.node, event["metadata"])
                    elif event_id == "pull_current":
                        self.pull_current(self.node, event["metadata"])
                    elif event_id == "write_goal_position":
                        self.write_goal_position(event["value"])
                    elif event_id == "r_hand_pos":
                        self.write_mocap_pos(event["value"])
                    elif event_id == "r_hand_quat":
                        self.write_mocap_quaT(event["value"])

                    elif event_id == "end":
                        break

                elif event_type == "ERROR":
                    raise ValueError(
                        "An error occurred in the dataflow: " + event["error"],
                    )

            self.node.send_output("end", pa.array([]))

    def pull_position(self, node, metadata):
        """TODO: Add docstring."""

    def pull_velocity(self, node, metadata):
        """TODO: Add docstring."""

    def pull_current(self, node, metadata):
        """TODO: Add docstring."""

    def write_goal_position(self, goal_position_with_joints):
        """TODO: Add docstring."""
        joints = goal_position_with_joints.field("joints")
        goal_position = goal_position_with_joints.field("values")

        for i, joint in enumerate(joints):
            self.data.joint(joint.as_py()).qpos[0] = goal_position[i].as_py()

    def write_mocap_pos(self, hand):

        # please, a method to access the mocap objects by name...

        if "r_tip1" in hand[0]:
            [x, y, z] = hand[0]["r_tip1"].values
            self.data.mocap_pos[0] = [
                x.as_py() * 1.5 - 0.025,
                y.as_py() * 1.5 + 0.022,
                z.as_py() * 1.5 + 0.098,
            ]
        if "r_tip2" in hand[0]:
            [x, y, z] = hand[0]["r_tip2"].values
            self.data.mocap_pos[1] = [
                x.as_py() * 1.5 - 0.025,
                y.as_py() * 1.5 - 0.009,
                z.as_py() * 1.5 + 0.092,
            ]
        if "r_tip3" in hand[0]:
            [x, y, z] = hand[0]["r_tip3"].values
            self.data.mocap_pos[2] = [
                x.as_py() * 1.5 - 0.025,
                y.as_py() * 1.5 - 0.040,
                z.as_py() * 1.5 + 0.082,
            ]
        if "r_tip4" in hand[0]:
            [x, y, z] = hand[0]["r_tip4"].values
            self.data.mocap_pos[3] = [
                x.as_py() * 1.5 + 0.024,
                y.as_py() * 1.5 + 0.019,
                z.as_py() * 1.5 + 0.017,
            ]

    def write_mocap_quat(self, hand):
        # please, a method to access the mocap objects by name...

        if "r_tip1" in hand[0]:
            [w, x, y, z] = hand[0]["r_tip1"].values
            self.data.mocap_quat[0] = [w.as_py(), x.as_py(), y.as_py(), z.as_py()]
        if "r_tip2" in hand[0]:
            [w, x, y, z] = hand[0]["r_tip2"].values
            self.data.mocap_quat[1] = [w.as_py(), x.as_py(), y.as_py(), z.as_py()]
        if "r_tip3" in hand[0]:
            [w, x, y, z] = hand[0]["r_tip3"].values
            self.data.mocap_quat[2] = [w.as_py(), x.as_py(), y.as_py(), z.as_py()]
        if "r_tip4" in hand[0]:
            [w, x, y, z] = hand[0]["r_tip4"].values
            self.data.mocap_quat[3] = [w.as_py(), x.as_py(), y.as_py(), z.as_py()]


def main():
    """Handle dynamic nodes, ask for the name of the node in the dataflow."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["pos", "quat"],
        default="pos",
        help="control mode: pos=position (we control the position of the tip) quat=quaternion (we control the orientation of the tip)",
    )
    args = parser.parse_args()
    client = Client(args.mode)
    client.run()


if __name__ == "__main__":
    main()
