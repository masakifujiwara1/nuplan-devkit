"""Observation module that simulates other agents with simple constant controls.

This module defines :class:`SimpleControlAgentsObservation`, an observation engine that keeps track of
non-ego agents by propagating them forward using a constant longitudinal speed and steering angle.
The geometry of each controlled agent is updated via a unicycle-style kinematic model, and the
resulting agents are returned as a :class:`~nuplan.planning.simulation.observation.observation_type.DetectionsTracks`
object that the simulation loop can consume directly.

Example Hydra configuration (store as ``simulation/observation/simple_control_agents_observation.yaml``)::

    _target_: nuplan.planning.simulation.observation.simple_control_agents.SimpleControlAgentsObservation
    _convert_: 'all'
    default_speed_mps: 5.0
    default_steering_rad: 0.0
    wheelbase_m: 2.8
    controlled_object_types: ["VEHICLE"]
    agent_control_overrides: {
      "track_token_to_override": {"speed_mps": 2.0, "steering_deg": 5.0}
    }

The overrides dictionary accepts object tokens or track tokens as keys and lets you customise the
motion of individual agents while the remaining controlled objects continue using the default
parameters.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.agent_state import AgentState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentControlCommand:
    """Simple control command describing the desired agent motion."""

    speed_mps: float
    steering_rad: float
    wheelbase_m: float


class SimpleControlAgentsObservation(AbstractObservation):
    """Observation engine that propagates other agents with constant controls.

    Each controlled agent follows a unicycle/Ackermann kinematic update with a configurable constant
    longitudinal speed and steering angle. The steering angle is interpreted in radians and combined
    with the wheelbase length to compute a yaw rate. Static objects (barriers, traffic cones, etc.)
    are forwarded from the scenario without modification.

    Args:
        scenario: Hydrated scenario provided by the scenario builder.
        default_speed_mps: Default speed (m/s) used for controlled agents that have no override.
        default_steering_rad: Default steering angle (rad) used for controlled agents that have no override.
        wheelbase_m: Wheelbase (m) used to map steering angles to yaw rates.
        agent_control_overrides: Optional mapping from token/track token to control parameters.
        controlled_object_types: List of tracked object types (string names) to control. Defaults to ["VEHICLE"].
        radius: Optional radius constraint around the initial ego pose. Agents outside the radius are ignored.
    """

    _DEFAULT_CONTROLLED_TYPES = (TrackedObjectType.VEHICLE,)

    def __init__(
        self,
        scenario: AbstractScenario,
        default_speed_mps: float = 5.0,
        default_steering_rad: float = 0.0,
        wheelbase_m: float = 2.8,
        agent_control_overrides: Optional[Dict[str, Dict[str, float]]] = None,
        controlled_object_types: Optional[Sequence[str]] = None,
        radius: Optional[float] = None,
    ) -> None:
        assert scenario is not None, "scenario must be provided"

        self._scenario = scenario
        self._default_control = AgentControlCommand(
            speed_mps=float(default_speed_mps),
            steering_rad=float(default_steering_rad),
            wheelbase_m=float(wheelbase_m),
        )
        self._control_overrides = self._parse_control_overrides(agent_control_overrides)
        self._controlled_types = self._resolve_controlled_types(controlled_object_types)
        self._radius = float(radius) if radius is not None else None

        self._agents: Dict[str, Agent] = {}
        self._agent_commands: Dict[str, AgentControlCommand] = {}
        self._static_tracked_objects: List[TrackedObject] = []
        self._current_observation: Optional[DetectionsTracks] = None
        self._initialized = False
        self._last_iteration_index: int = 0
        self._last_timestamp_us: int = scenario.start_time.time_us
        self._default_dt: float = float(scenario.database_interval)

    # ------------------------------------------------------------------
    # AbstractObservation API
    # ------------------------------------------------------------------

    def observation_type(self) -> type[Observation]:
        return DetectionsTracks

    def reset(self) -> None:
        self._agents.clear()
        self._agent_commands.clear()
        self._static_tracked_objects = []
        self._current_observation = None
        self._initialized = False
        self._last_iteration_index = 0
        self._last_timestamp_us = self._scenario.start_time.time_us

    def initialize(self) -> None:
        # Ensure a clean slate before bootstrapping from scenario data.
        self.reset()
        self._bootstrap_from_scenario()
        self._current_observation = self._compose_observation()
        self._initialized = True

    def get_observation(self) -> DetectionsTracks:
        self._ensure_initialized()
        if self._current_observation is None:
            self._current_observation = self._compose_observation()
        return self._current_observation

    def update_observation(
        self,
        iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        history: SimulationHistoryBuffer,
    ) -> None:
        del history  # Unused in this simple implementation.

        self._ensure_initialized()

        if next_iteration.index <= iteration.index:
            logger.debug(
                "Received non-forward iteration update (current=%s, next=%s); skipping propagation.",
                iteration.index,
                next_iteration.index,
            )
            self._current_observation = self._compose_observation()
            return

        dt = max(next_iteration.time_s - iteration.time_s, 0.0)
        if dt <= 0.0:
            dt = max(self._default_dt, 0.0)

        next_timestamp_us = next_iteration.time_us
        for token, agent in list(self._agents.items()):
            command = self._agent_commands.get(token, self._default_control)
            propagated = self._propagate_agent(agent, command, dt, next_timestamp_us)
            self._agents[token] = propagated

        self._last_iteration_index = next_iteration.index
        self._last_timestamp_us = next_timestamp_us
        self._current_observation = self._compose_observation()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    def _bootstrap_from_scenario(self) -> None:
        detections = self._scenario.initial_tracked_objects
        ego_pose = self._scenario.initial_ego_state.center

        for tracked in detections.tracked_objects:
            try:
                tracked_type = tracked.tracked_object_type
            except AttributeError:
                continue

            if tracked_type == TrackedObjectType.EGO:
                continue

            within_radius = True
            if self._radius is not None:
                try:
                    distance = ego_pose.distance_to(tracked.center)
                    within_radius = distance <= self._radius
                except AttributeError:
                    within_radius = False

            if not within_radius:
                continue

            if tracked_type in self._controlled_types:
                agent = self._as_agent(tracked)
                if agent is None:
                    logger.debug("Skipping tracked object %s: cannot convert to agent", getattr(tracked, "token", "?"))
                    continue
                self._agents[agent.metadata.token] = agent
                self._agent_commands[agent.metadata.token] = self._lookup_command(agent.metadata)
            else:
                self._static_tracked_objects.append(tracked)

    def _compose_observation(self) -> DetectionsTracks:
        controlled = list(self._agents.values())
        tracked_objects = list(self._static_tracked_objects) + controlled
        return DetectionsTracks(tracked_objects=TrackedObjects(tracked_objects))

    def _lookup_command(self, metadata: SceneObjectMetadata) -> AgentControlCommand:
        override_keys: Iterable[Optional[str]] = (metadata.track_token, metadata.token)
        for key in override_keys:
            if key and key in self._control_overrides:
                return self._control_overrides[key]
        return self._default_control

    def _propagate_agent(
        self,
        agent: Agent,
        command: AgentControlCommand,
        dt: float,
        next_timestamp_us: int,
    ) -> Agent:
        if dt <= 0.0:
            return self._clone_agent_with_timestamp(agent, next_timestamp_us)

        pose = agent.box.center
        heading = pose.heading
        speed = command.speed_mps
        steering = command.steering_rad
        wheelbase = max(abs(command.wheelbase_m), 1e-3)

        # Compute yaw rate from steering angle. Guard against extremely small angles.
        if abs(steering) < 1e-6 or abs(speed) < 1e-6:
            yaw_rate = 0.0
            delta_x = speed * dt * math.cos(heading)
            delta_y = speed * dt * math.sin(heading)
            new_heading = heading
        else:
            turning_radius = wheelbase / math.tan(steering)
            # Prevent division by zero for pathological steering values.
            if abs(turning_radius) < 1e-6:
                turning_radius = math.copysign(1e-6, turning_radius if turning_radius != 0.0 else steering)
            yaw_rate = speed / turning_radius
            new_heading = heading + yaw_rate * dt
            delta_x = turning_radius * (math.sin(new_heading) - math.sin(heading))
            delta_y = -turning_radius * (math.cos(new_heading) - math.cos(heading))

        new_pose = StateSE2(x=pose.x + delta_x, y=pose.y + delta_y, heading=new_heading)
        new_box = OrientedBox.from_new_pose(agent.box, new_pose)
        new_velocity = StateVector2D(speed * math.cos(new_heading), speed * math.sin(new_heading))

        new_metadata = replace(agent.metadata, timestamp_us=next_timestamp_us)

        propagated = Agent(
            tracked_object_type=agent.tracked_object_type,
            oriented_box=new_box,
            velocity=new_velocity,
            metadata=new_metadata,
            angular_velocity=yaw_rate,
            predictions=agent.predictions,
            past_trajectory=agent.past_trajectory,
        )
        return propagated

    def _clone_agent_with_timestamp(self, agent: Agent, timestamp_us: int) -> Agent:
        new_metadata = replace(agent.metadata, timestamp_us=timestamp_us)
        return Agent(
            tracked_object_type=agent.tracked_object_type,
            oriented_box=agent.box,
            velocity=agent.velocity,
            metadata=new_metadata,
            angular_velocity=agent.angular_velocity,
            predictions=agent.predictions,
            past_trajectory=agent.past_trajectory,
        )

    @staticmethod
    def _as_agent(tracked: TrackedObject) -> Optional[Agent]:
        if isinstance(tracked, AgentState):
            return Agent.from_agent_state(tracked)
        return None

    def _parse_control_overrides(
        self, agent_control_overrides: Optional[Dict[str, Dict[str, float]]]
    ) -> Dict[str, AgentControlCommand]:
        if not agent_control_overrides:
            return {}

        overrides: Dict[str, AgentControlCommand] = {}
        for key, cfg in agent_control_overrides.items():
            if not isinstance(cfg, dict):
                logger.warning("Agent override for %s is not a dictionary; skipping", key)
                continue

            speed = self._extract_float(cfg, ("speed_mps", "speed"))
            steering = self._extract_float(cfg, ("steering_rad", "steering"))
            steering_deg = self._extract_float(cfg, ("steering_deg",))
            wheelbase = self._extract_float(cfg, ("wheelbase_m", "wheelbase"))

            speed_val = float(speed) if speed is not None else self._default_control.speed_mps
            steering_val: float
            if steering is not None:
                steering_val = float(steering)
            elif steering_deg is not None:
                steering_val = math.radians(float(steering_deg))
            else:
                steering_val = self._default_control.steering_rad

            wheelbase_val = float(wheelbase) if wheelbase is not None else self._default_control.wheelbase_m

            overrides[key] = AgentControlCommand(
                speed_mps=speed_val,
                steering_rad=steering_val,
                wheelbase_m=wheelbase_val,
            )
        return overrides

    @staticmethod
    def _extract_float(config: Dict[str, float], keys: Sequence[str]) -> Optional[float]:
        for key in keys:
            if key in config:
                try:
                    return float(config[key])
                except (TypeError, ValueError):
                    logger.warning("Could not parse float from key '%s' (value=%s)", key, config[key])
                    return None
        return None

    @staticmethod
    def _resolve_controlled_types(controlled_object_types: Optional[Sequence[str]]) -> Tuple[TrackedObjectType, ...]:
        if not controlled_object_types:
            return SimpleControlAgentsObservation._DEFAULT_CONTROLLED_TYPES

        resolved_types = set()
        requested = [name.upper() for name in controlled_object_types]
        if any(name in {"*", "ALL"} for name in requested):
            resolved_types.update({t for t in AGENT_TYPES if t != TrackedObjectType.EGO})
        else:
            for name in requested:
                try:
                    candidate = TrackedObjectType[name]
                except KeyError as exc:
                    raise ValueError(f"Unknown tracked object type '{name}'") from exc
                if candidate == TrackedObjectType.EGO:
                    continue
                resolved_types.add(candidate)

        if not resolved_types:
            resolved_types.update(SimpleControlAgentsObservation._DEFAULT_CONTROLLED_TYPES)

        return tuple(sorted(resolved_types, key=lambda t: t.value))
