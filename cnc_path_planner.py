"""Preview-only adaptive circular toolpath planning from six-face scan data.

Depth grids contain non-negative millimeter depths from a shared datum. Optional
density_signal grids must be calibrated, normalized sensor readings in [0, 1];
surface depth alone cannot determine material density. The module plans motion
and can monitor an externally supplied read-only telemetry port. It never writes
to a serial port or connects to a CNC motion controller.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import queue
import threading
import time
from typing import Callable, Iterator, Mapping, Protocol, Sequence


FACE_NAMES = ("Top", "Bottom", "Front", "Back", "Left", "Right")
SIDE_FACE_ORDER = ("Right", "Back", "Left", "Front")


class SerialReadable(Protocol):
    """Minimal receive-only serial interface expected by the telemetry monitor."""

    timeout: float | None

    def readline(self) -> bytes:
        """Read one newline-terminated telemetry record."""


@dataclass(frozen=True)
class TelemetrySample:
    """One validated sensor record received from the telemetry serial port."""

    spindle_load_pct: float
    depth_mm: float


@dataclass(frozen=True)
class TelemetryFault:
    """Latched reason to request a controller-approved safe hold."""

    reason: str
    sample: TelemetrySample | None


@dataclass(frozen=True)
class TelemetryConnection:
    """Receive-only telemetry source and its selected operating mode."""

    source: SerialReadable
    simulated: bool
    description: str

    def close(self) -> None:
        """Close the source when it provides a close method."""
        close_source = getattr(self.source, "close", None)
        if callable(close_source):
            close_source()


class SimulatedTelemetryPort:
    """Generate bounded JSON sensor samples without opening a hardware port."""

    def __init__(self, timeout: float, interval_seconds: float = 0.1) -> None:
        """Validate timing and initialize the simulated serial-like source."""
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and greater than zero")
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be finite and greater than zero")
        self.timeout = timeout
        self._interval_seconds = min(interval_seconds, timeout)
        self._closed_event = threading.Event()
        self._sample_index = 0

    def readline(self) -> bytes:
        """Wait for the next simulated sensor tick and return one JSON line."""
        if self._closed_event.wait(self._interval_seconds):
            return b""
        sample_index = self._sample_index
        self._sample_index += 1
        record = {
            "spindle_load_pct": 25.0 + 5.0 * math.sin(sample_index / 10.0),
            "depth_mm": 2.0 + 0.05 * math.sin(sample_index / 8.0),
        }
        return (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")

    def close(self) -> None:
        """Stop future simulated reads and wake a pending read immediately."""
        self._closed_event.set()


def open_telemetry_connection(
    port: str | None = None,
    *,
    baudrate: int = 115200,
    timeout: float = 0.25,
    simulation_interval_seconds: float = 0.1,
) -> TelemetryConnection:
    """Open one detected PySerial telemetry port or return simulated telemetry.

    No-port and missing-PySerial cases select simulation. If multiple devices are
    present, pass ``port`` explicitly. A detected physical port that fails to open
    raises an error rather than silently switching modes.
    """
    if baudrate <= 0:
        raise ValueError("baudrate must be greater than zero")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and greater than zero")
    if not math.isfinite(simulation_interval_seconds) or simulation_interval_seconds <= 0:
        raise ValueError("simulation_interval_seconds must be finite and greater than zero")

    # Missing PySerial means no reliable hardware discovery is possible, so remain safe in simulation.
    try:
        import serial
        from serial.tools import list_ports
    except ImportError:
        return TelemetryConnection(
            SimulatedTelemetryPort(timeout, simulation_interval_seconds),
            True,
            "simulated telemetry (PySerial is unavailable)",
        )

    # Enumerate ports before opening anything; never guess when more than one device exists.
    available_ports = tuple(device.device for device in list_ports.comports())
    if not available_ports:
        return TelemetryConnection(
            SimulatedTelemetryPort(timeout, simulation_interval_seconds),
            True,
            "simulated telemetry (no physical serial port detected)",
        )
    if port is None:
        if len(available_ports) != 1:
            raise ValueError(
                "multiple serial ports detected; pass the telemetry device port explicitly"
            )
        selected_port = available_ports[0]
    elif port not in available_ports:
        raise ValueError(f"requested telemetry port {port!r} was not detected")
    else:
        selected_port = port

    # Configure bounded reads only; this module never writes motion or control commands.
    source = serial.Serial(
        port=selected_port,
        baudrate=baudrate,
        timeout=timeout,
        write_timeout=timeout,
    )
    return TelemetryConnection(source, False, f"serial telemetry on {selected_port}")


def _validate_grid_map(
    values: Mapping[str, Sequence[Sequence[float]]],
    name: str,
    normalized: bool = False,
) -> dict[str, tuple[tuple[float, ...], ...]]:
    """Validate and freeze all six rectangular scan matrices."""
    if set(values) != set(FACE_NAMES):
        raise ValueError(f"{name} must contain exactly these faces: {', '.join(FACE_NAMES)}")

    validated: dict[str, tuple[tuple[float, ...], ...]] = {}
    for face in FACE_NAMES:
        rows = values[face]
        if not rows or not rows[0]:
            raise ValueError(f"{name}[{face!r}] must be a non-empty matrix")
        width = len(rows[0])
        converted_rows: list[tuple[float, ...]] = []
        for row in rows:
            if len(row) != width:
                raise ValueError(f"{name}[{face!r}] must be rectangular")
            converted_row = tuple(float(value) for value in row)
            for value in converted_row:
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name} values must be finite and non-negative")
                if normalized and value > 1:
                    raise ValueError(f"{name} values must be normalized to [0, 1]")
            converted_rows.append(converted_row)
        validated[face] = tuple(converted_rows)
    return validated


def _mean(values: Sequence[Sequence[float]]) -> float:
    """Return the arithmetic mean of a non-empty matrix."""
    flattened = [value for row in values for value in row]
    return sum(flattened) / len(flattened)


@dataclass(frozen=True)
class SixDirectionScan:
    """Six directional depth matrices and optional calibrated density readings."""

    depth_mm: Mapping[str, Sequence[Sequence[float]]]
    density_signal: Mapping[str, Sequence[Sequence[float]]] | None = None

    def __post_init__(self) -> None:
        """Validate scan shape, units, and optional density signal bounds."""
        object.__setattr__(self, "depth_mm", _validate_grid_map(self.depth_mm, "depth_mm"))
        if self.density_signal is not None:
            object.__setattr__(
                self,
                "density_signal",
                _validate_grid_map(self.density_signal, "density_signal", normalized=True),
            )


@dataclass(frozen=True)
class PlannerConfig:
    """Machine-independent geometry and conservative planning limits."""

    radius_mm: float
    max_cut_depth_mm: float
    safe_z_mm: float
    min_feed_mm_min: float
    max_feed_mm_min: float
    segments_per_turn: int = 24
    clockwise: bool = False

    def __post_init__(self) -> None:
        """Reject unsafe or geometrically invalid planner parameters."""
        positive_values = {
            "radius_mm": self.radius_mm,
            "max_cut_depth_mm": self.max_cut_depth_mm,
            "safe_z_mm": self.safe_z_mm,
            "min_feed_mm_min": self.min_feed_mm_min,
            "max_feed_mm_min": self.max_feed_mm_min,
        }
        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and greater than zero")
        if self.min_feed_mm_min > self.max_feed_mm_min:
            raise ValueError("min_feed_mm_min cannot exceed max_feed_mm_min")
        if self.segments_per_turn < 4:
            raise ValueError("segments_per_turn must be at least 4")


@dataclass(frozen=True)
class FeedbackSample:
    """A calibrated per-segment sensor result supplied by an external caller."""

    depth_mm: float
    density_signal: float


@dataclass(frozen=True)
class ArcMove:
    """One circular interpolation segment in the XY plane with a Z endpoint."""

    x_mm: float
    y_mm: float
    depth_mm: float
    i_mm: float
    j_mm: float
    feed_mm_min: float
    clockwise: bool


@dataclass(frozen=True)
class TorqueFeedbackDecision:
    """Controller-independent response to one spindle-torque sample."""

    action: str
    feed_mm_min: float
    move: ArcMove | None
    next_vector_xy: tuple[float, float]


@dataclass(frozen=True)
class Toolpath:
    """A planned circular path that can be inspected before G-code rendering."""

    moves: tuple[ArcMove, ...]
    initial_depth_mm: float
    initial_feed_mm_min: float
    config: PlannerConfig

    def to_gcode(self, *, acknowledge_machine_validation: bool = False) -> str:
        """Render preview G-code only after the caller acknowledges validation."""
        if not acknowledge_machine_validation:
            raise ValueError(
                "G-code rendering requires acknowledge_machine_validation=True; "
                "verify the work offset, fixture, tool, controller arc conventions, "
                "and travel limits first"
            )

        # G17 selects XY arcs, so I/J are center offsets and K is intentionally omitted.
        lines = [
            "(PREVIEW ONLY: validate controller, workholding, tool, and travel limits)",
            "(No spindle or machine-controller commands are emitted)",
            "G21 G90 G17 G94",
            f"G0 X{self.config.radius_mm:.3f} Y0.000 Z{self.config.safe_z_mm:.3f}",
            f"G1 Z{-self.initial_depth_mm:.3f} F{self.initial_feed_mm_min:.1f}",
        ]

        # Each short arc has its center at the origin and uses offsets from its start point.
        for move in self.moves:
            command = "G2" if move.clockwise else "G3"
            lines.append(
                f"{command} X{move.x_mm:.3f} Y{move.y_mm:.3f} "
                f"Z{-move.depth_mm:.3f} I{move.i_mm:.3f} J{move.j_mm:.3f} "
                f"F{move.feed_mm_min:.1f}"
            )

        # Retract after the planned contour; no spindle, homing, or offset changes are included.
        lines.extend((f"G0 Z{self.config.safe_z_mm:.3f}", "M30"))
        return "\n".join(lines)


def _circular_side_depth(face_means: Mapping[str, float], angle: float) -> float:
    """Smoothly interpolate side-face depths around the XY circle."""
    # Face order follows +X, +Y, -X, -Y; Catmull-Rom keeps the profile continuous.
    position = (angle % (2 * math.pi)) * len(SIDE_FACE_ORDER) / (2 * math.pi)
    index = math.floor(position)
    fraction = position - index
    values = [face_means[face] for face in SIDE_FACE_ORDER]
    p0 = values[(index - 1) % len(values)]
    p1 = values[index % len(values)]
    p2 = values[(index + 1) % len(values)]
    p3 = values[(index + 2) % len(values)]
    fraction2 = fraction * fraction
    fraction3 = fraction2 * fraction
    return 0.5 * (
        2 * p1
        + (-p0 + p2) * fraction
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * fraction2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * fraction3
    )


def plan_circular_toolpath(
    scan: SixDirectionScan,
    config: PlannerConfig,
    feedback_provider: Callable[[int, float], FeedbackSample | None] | None = None,
) -> Toolpath:
    """Build a static path by collecting one optional feedback sample per arc."""
    axial_baseline, side_means, density_baseline = _scan_summary(scan, config)
    initial_depth, initial_feed = _initial_state(
        axial_baseline, side_means, density_baseline, config
    )
    moves = tuple(_iter_adaptive_moves(
        config, feedback_provider, axial_baseline, side_means, density_baseline
    ))
    return Toolpath(moves, initial_depth, initial_feed, config)


def _scan_summary(
    scan: SixDirectionScan,
    config: PlannerConfig,
) -> tuple[float, dict[str, float], float]:
    """Summarize depth and calibrated density grids, enforcing the depth limit."""
    depth_means = {face: _mean(scan.depth_mm[face]) for face in FACE_NAMES}
    if any(
        value > config.max_cut_depth_mm
        for grid in scan.depth_mm.values()
        for row in grid
        for value in row
    ):
        raise ValueError("scan depth exceeds max_cut_depth_mm")
    axial_baseline = (depth_means["Top"] + depth_means["Bottom"]) / 2
    side_means = {face: depth_means[face] for face in SIDE_FACE_ORDER}

    # Missing calibration is represented as neutral density and never inferred from depth.
    density_baseline = 0.5
    if scan.density_signal is not None:
        density_baseline = sum(_mean(scan.density_signal[face]) for face in FACE_NAMES) / len(FACE_NAMES)
    return axial_baseline, side_means, density_baseline


def _initial_state(
    axial_baseline: float,
    side_means: Mapping[str, float],
    density_signal: float,
    config: PlannerConfig,
) -> tuple[float, float]:
    """Calculate the starting cut depth and feed from the scan profile."""
    initial_depth = (axial_baseline + _circular_side_depth(side_means, 0.0)) / 2
    initial_depth = min(config.max_cut_depth_mm, max(0.0, initial_depth))
    initial_feed = config.max_feed_mm_min - density_signal * (
        config.max_feed_mm_min - config.min_feed_mm_min
    )
    return initial_depth, initial_feed


def _iter_adaptive_moves(
    config: PlannerConfig,
    feedback_provider: Callable[[int, float], FeedbackSample | None] | None,
    axial_baseline: float,
    side_means: Mapping[str, float],
    density_baseline: float,
) -> Iterator[ArcMove]:
    """Yield one bounded arc at a time, reading feedback immediately before each."""
    segment_count = config.segments_per_turn
    direction = -1 if config.clockwise else 1
    for segment_index in range(1, segment_count + 1):
        angle = direction * (2 * math.pi * segment_index / segment_count)
        profile_depth = (axial_baseline + _circular_side_depth(side_means, angle)) / 2
        profile_depth = min(config.max_cut_depth_mm, max(0.0, profile_depth))

        # Read current caller-supplied sensor values when this segment is requested.
        feedback = feedback_provider(segment_index - 1, angle) if feedback_provider else None
        if feedback is None:
            depth_mm = profile_depth
            density_signal = density_baseline
        else:
            depth_mm = float(feedback.depth_mm)
            density_signal = float(feedback.density_signal)
            if not math.isfinite(depth_mm) or not 0 <= depth_mm <= config.max_cut_depth_mm:
                raise ValueError("feedback depth_mm must be finite and within configured depth limits")
            if not math.isfinite(density_signal) or not 0 <= density_signal <= 1:
                raise ValueError("feedback density_signal must be finite and normalized to [0, 1]")

        # Higher density signal reduces feed; the configured feed range is always respected.
        feed = config.max_feed_mm_min - density_signal * (
            config.max_feed_mm_min - config.min_feed_mm_min
        )
        x_mm = config.radius_mm * math.cos(angle)
        y_mm = config.radius_mm * math.sin(angle)
        start_angle = direction * (2 * math.pi * (segment_index - 1) / segment_count)
        start_x = config.radius_mm * math.cos(start_angle)
        start_y = config.radius_mm * math.sin(start_angle)

        # I/J point from the segment start to the common circle center at (0, 0).
        yield ArcMove(
            x_mm=x_mm,
            y_mm=y_mm,
            depth_mm=depth_mm,
            i_mm=-start_x,
            j_mm=-start_y,
            feed_mm_min=feed,
            clockwise=config.clockwise,
        )


def iter_adaptive_gcode(
    scan: SixDirectionScan,
    config: PlannerConfig,
    feedback_provider: Callable[[int, float], FeedbackSample | None] | None = None,
    *,
    acknowledge_machine_validation: bool = False,
) -> Iterator[str]:
    """Yield preview G-code incrementally, sampling feedback before each arc.

    The caller controls when each next line is requested and is responsible for
    supplying calibrated sensor data and any safety-rated controller integration.
    This iterator only produces text; it never sends commands to machine hardware.
    """
    if not acknowledge_machine_validation:
        raise ValueError(
            "G-code streaming requires acknowledge_machine_validation=True; "
            "verify the work offset, fixture, tool, controller arc conventions, "
            "and travel limits first"
        )

    # Prepare the validated scan summary before producing any motion text.
    axial_baseline, side_means, density_baseline = _scan_summary(scan, config)
    initial_depth, initial_feed = _initial_state(
        axial_baseline, side_means, density_baseline, config
    )

    # Emit safe-start text and plunge before asking for the first segment's feedback.
    yield "(PREVIEW ONLY: validate controller, workholding, tool, and travel limits)"
    yield "(No spindle or machine-controller commands are emitted)"
    yield "G21 G90 G17 G94"
    yield f"G0 X{config.radius_mm:.3f} Y0.000 Z{config.safe_z_mm:.3f}"
    yield f"G1 Z{-initial_depth:.3f} F{initial_feed:.1f}"

    # Request and emit each sensor-adjusted arc individually for incremental consumers.
    for move in _iter_adaptive_moves(
        config, feedback_provider, axial_baseline, side_means, density_baseline
    ):
        command = "G2" if move.clockwise else "G3"
        yield (
            f"{command} X{move.x_mm:.3f} Y{move.y_mm:.3f} "
            f"Z{-move.depth_mm:.3f} I{move.i_mm:.3f} J{move.j_mm:.3f} "
            f"F{move.feed_mm_min:.1f}"
        )

    # Retract after the final arc; controller-specific stop behavior is not configured here.
    yield f"G0 Z{config.safe_z_mm:.3f}"
    yield "M30"


def torque_feedback_loop(
    torque_nm: float,
    max_material_torque_nm: float,
    position_xy: tuple[float, float],
    vector_xy: tuple[float, float],
    depth_mm: float,
    feed_mm_min: float,
    bypass_radius_mm: float,
    *,
    clockwise: bool = False,
    sweep_rad: float = math.pi / 2,
    bypass_clearance_confirmed: bool = False,
) -> TorqueFeedbackDecision:
    """Convert one torque sample into continue, stop, or a tangential arc decision.

    Call once per externally sampled sensor update. Above the material torque
    threshold, the arc endpoint and outgoing vector are calculated in XY and feed
    is reduced to 60 percent. Clearance must be confirmed before motion is offered;
    this function returns a decision only and never communicates with a controller.
    """
    # Validate sensor values and motion inputs before using them to plan an adjustment.
    finite_values = (torque_nm, max_material_torque_nm, depth_mm, feed_mm_min,
                     bypass_radius_mm, sweep_rad, *position_xy, *vector_xy)
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("torque and trajectory inputs must be finite")
    if torque_nm < 0 or max_material_torque_nm <= 0:
        raise ValueError("torque_nm must be non-negative and threshold must be positive")
    if depth_mm < 0 or feed_mm_min <= 0 or bypass_radius_mm <= 0:
        raise ValueError("depth must be non-negative; feed and bypass radius must be positive")
    if not 0 < sweep_rad <= math.pi / 2:
        raise ValueError("sweep_rad must be greater than zero and at most 90 degrees")

    # Normalize the requested path vector so the bypass geometry is independent of scale.
    vector_length = math.hypot(*vector_xy)
    if vector_length == 0:
        raise ValueError("vector_xy must have non-zero length")
    heading_x = vector_xy[0] / vector_length
    heading_y = vector_xy[1] / vector_length

    # Under threshold, retain the requested feed and direction without generating an arc.
    if torque_nm <= max_material_torque_nm:
        return TorqueFeedbackDecision(
            action="continue",
            feed_mm_min=feed_mm_min,
            move=None,
            next_vector_xy=(heading_x, heading_y),
        )

    # If the bypass region is not known clear, request an external controlled stop.
    if not bypass_clearance_confirmed:
        return TorqueFeedbackDecision(
            action="stop",
            feed_mm_min=0.0,
            move=None,
            next_vector_xy=(heading_x, heading_y),
        )

    # Place the arc center on the selected side so the circular path starts tangent to travel.
    turn_sign = -1.0 if clockwise else 1.0
    normal_x = -heading_y * turn_sign
    normal_y = heading_x * turn_sign
    center_x = position_xy[0] + normal_x * bypass_radius_mm
    center_y = position_xy[1] + normal_y * bypass_radius_mm

    # Rotate the start radius around the center to find the bypass endpoint.
    signed_sweep = turn_sign * sweep_rad
    radius_start_x = position_xy[0] - center_x
    radius_start_y = position_xy[1] - center_y
    cosine = math.cos(signed_sweep)
    sine = math.sin(signed_sweep)
    radius_end_x = radius_start_x * cosine - radius_start_y * sine
    radius_end_y = radius_start_x * sine + radius_start_y * cosine
    end_x = center_x + radius_end_x
    end_y = center_y + radius_end_y

    # Rotate the incoming vector by the same sweep to provide a tangent outgoing direction.
    next_vector_x = heading_x * cosine - heading_y * sine
    next_vector_y = heading_x * sine + heading_y * cosine

    # Reduce feed by 40 percent and return center offsets relative to the arc start.
    reduced_feed = feed_mm_min * 0.60
    bypass_move = ArcMove(
        x_mm=end_x,
        y_mm=end_y,
        depth_mm=depth_mm,
        i_mm=center_x - position_xy[0],
        j_mm=center_y - position_xy[1],
        feed_mm_min=reduced_feed,
        clockwise=clockwise,
    )
    return TorqueFeedbackDecision(
        action="bypass",
        feed_mm_min=reduced_feed,
        move=bypass_move,
        next_vector_xy=(next_vector_x, next_vector_y),
    )


class SerialTelemetryMonitor:
    """Read sensor telemetry in the background and latch anomalies for the caller.

    The supplied port must be a dedicated receive-only telemetry port configured
    with a finite timeout. The worker never writes commands or changes machine
    motion. A controller integration should observe ``fault_event`` and request
    its documented safe hold from the controller's owning thread.
    """

    def __init__(
        self,
        port: SerialReadable,
        *,
        max_spindle_load_pct: float,
        max_depth_variance_mm: float,
        depth_window_samples: int = 8,
        telemetry_timeout_seconds: float = 1.0,
        poll_interval_seconds: float = 0.01,
    ) -> None:
        """Validate monitor limits and prepare its thread-safe sample channels."""
        limits = (
            max_spindle_load_pct,
            max_depth_variance_mm,
            telemetry_timeout_seconds,
            poll_interval_seconds,
        )
        if not all(math.isfinite(value) for value in limits):
            raise ValueError("monitor limits must be finite")
        if not 0 < max_spindle_load_pct <= 100:
            raise ValueError("max_spindle_load_pct must be in (0, 100]")
        if max_depth_variance_mm < 0:
            raise ValueError("max_depth_variance_mm must be non-negative")
        if depth_window_samples < 2:
            raise ValueError("depth_window_samples must be at least 2")
        if telemetry_timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("telemetry and poll timeouts must be greater than zero")

        # A finite serial timeout ensures the worker can periodically observe shutdown.
        port_timeout = getattr(port, "timeout", None)
        if port_timeout is None or not math.isfinite(float(port_timeout)):
            raise ValueError("serial port must use a finite read timeout")
        if float(port_timeout) > telemetry_timeout_seconds:
            raise ValueError("serial read timeout cannot exceed telemetry_timeout_seconds")

        self._port = port
        self._max_spindle_load_pct = max_spindle_load_pct
        self._max_depth_variance_mm = max_depth_variance_mm
        self._depth_window_samples = depth_window_samples
        self._telemetry_timeout_seconds = telemetry_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._stop_event = threading.Event()
        self.fault_event = threading.Event()
        self._fault_queue: queue.Queue[TelemetryFault] = queue.Queue(maxsize=1)
        self._sample_queue: queue.Queue[TelemetrySample] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the single background reader unless it is already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        if self.fault_event.is_set():
            raise RuntimeError("monitor has latched a fault; create a fresh monitor after inspection")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._read_loop,
            name="cnc-serial-telemetry",
            daemon=True,
        )
        self._thread.start()

    def stop(self, join_timeout_seconds: float = 1.0) -> None:
        """Request shutdown and wait for the bounded serial read to return."""
        if not math.isfinite(join_timeout_seconds) or join_timeout_seconds <= 0:
            raise ValueError("join_timeout_seconds must be finite and greater than zero")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(join_timeout_seconds)
            if self._thread.is_alive():
                raise TimeoutError("telemetry reader did not stop; check the serial timeout")

    def latest_sample(self) -> TelemetrySample | None:
        """Return the most recently received sample without blocking."""
        latest: TelemetrySample | None = None
        while True:
            try:
                latest = self._sample_queue.get_nowait()
            except queue.Empty:
                return latest

    def get_fault(self, timeout_seconds: float = 0.0) -> TelemetryFault | None:
        """Return the latched anomaly, waiting at most the requested duration."""
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("timeout_seconds must be finite and non-negative")
        try:
            return self._fault_queue.get(timeout=timeout_seconds)
        except queue.Empty:
            return None

    def _latch_fault(self, reason: str, sample: TelemetrySample | None) -> None:
        """Publish the first fault and prevent further reads until inspected."""
        if self.fault_event.is_set():
            return
        self.fault_event.set()
        self._fault_queue.put_nowait(TelemetryFault(reason, sample))

    def _publish_sample(self, sample: TelemetrySample) -> None:
        """Keep only the newest sample for non-blocking caller access."""
        try:
            self._sample_queue.get_nowait()
        except queue.Empty:
            pass
        self._sample_queue.put_nowait(sample)

    def _parse_sample(self, raw_line: bytes) -> TelemetrySample:
        """Decode and validate one JSON telemetry line."""
        record = json.loads(raw_line.decode("utf-8"))
        if not isinstance(record, dict):
            raise ValueError("telemetry record must be a JSON object")
        sample = TelemetrySample(
            spindle_load_pct=float(record["spindle_load_pct"]),
            depth_mm=float(record["depth_mm"]),
        )
        if not math.isfinite(sample.spindle_load_pct) or not 0 <= sample.spindle_load_pct <= 100:
            raise ValueError("spindle_load_pct must be finite and within [0, 100]")
        if not math.isfinite(sample.depth_mm) or sample.depth_mm < 0:
            raise ValueError("depth_mm must be finite and non-negative")
        return sample

    def _read_loop(self) -> None:
        """Read telemetry until shutdown or the first detected anomaly."""
        depth_samples: deque[float] = deque(maxlen=self._depth_window_samples)
        last_record_time = time.monotonic()
        while not self._stop_event.is_set():
            try:
                raw_line = self._port.readline()
            except Exception:
                self._latch_fault("serial_read_error", None)
                return

            # Empty reads are normal serial timeouts until telemetry becomes stale.
            if not raw_line:
                if time.monotonic() - last_record_time > self._telemetry_timeout_seconds:
                    self._latch_fault("telemetry_timeout", None)
                    return
                self._stop_event.wait(self._poll_interval_seconds)
                continue

            try:
                sample = self._parse_sample(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                self._latch_fault("invalid_telemetry", None)
                return

            # Publish valid readings before threshold checks for diagnostics.
            last_record_time = time.monotonic()
            self._publish_sample(sample)
            if sample.spindle_load_pct > self._max_spindle_load_pct:
                self._latch_fault("spindle_load_exceeded", sample)
                return

            # Peak-to-peak depth spread detects abrupt variance in the recent sample window.
            depth_samples.append(sample.depth_mm)
            if (
                len(depth_samples) == self._depth_window_samples
                and max(depth_samples) - min(depth_samples) > self._max_depth_variance_mm
            ):
                self._latch_fault("depth_variance_exceeded", sample)
                return


if __name__ == "__main__":
    # Build matching six-face depth and density grids with a dense center knot.
    faces = FACE_NAMES
    depth_grid = [[1.0 for _ in range(7)] for _ in range(7)]
    density_grid = [[0.2 for _ in range(7)] for _ in range(7)]
    density_grid[3][3] = 0.98
    for row, column in ((2, 3), (3, 2), (3, 4), (4, 3)):
        density_grid[row][column] = 0.8
    scan = SixDirectionScan(
        depth_mm={face: depth_grid for face in faces},
        density_signal={face: density_grid for face in faces},
    )

    # Use the explicit simulator so this entrypoint never discovers or opens hardware.
    simulation_port = SimulatedTelemetryPort(timeout=0.25, interval_seconds=0.01)
    planner_config = PlannerConfig(
        radius_mm=10.0,
        max_cut_depth_mm=2.0,
        safe_z_mm=5.0,
        min_feed_mm_min=100.0,
        max_feed_mm_min=1000.0,
        segments_per_turn=24,
    )
    planned_path = plan_circular_toolpath(scan, planner_config)
    baseline_record = json.loads(simulation_port.readline().decode("utf-8"))
    print("[SIM] Six-face scan loaded; high-density knot at grid index (3, 3).")
    print(f"[SIM] Planned baseline: {len(planned_path.moves)} circular segments.")
    print(
        "[SIM] Telemetry baseline: "
        f"spindle_load={baseline_record['spindle_load_pct']:.1f}%, "
        f"depth={baseline_record['depth_mm']:.3f} mm."
    )

    # Inject a deterministic 90% simulated load against an 85% threshold.
    simulated_load_pct = 90.0
    load_threshold_pct = 85.0
    current_feed_mm_min = 1000.0
    start_xy = (10.0, 0.0)
    incoming_vector = (0.0, 1.0)
    decision = torque_feedback_loop(
        torque_nm=simulated_load_pct,
        max_material_torque_nm=load_threshold_pct,
        position_xy=start_xy,
        vector_xy=incoming_vector,
        depth_mm=1.0,
        feed_mm_min=current_feed_mm_min,
        bypass_radius_mm=2.0,
        clockwise=False,
        sweep_rad=math.pi / 2,
        bypass_clearance_confirmed=True,
    )
    if decision.action != "bypass" or decision.move is None:
        raise AssertionError("simulated over-threshold load did not trigger a bypass")

    # Verify the generated arc begins tangent to travel and reduces feed by exactly 40%.
    move = decision.move
    tangent_at_start = (move.j_mm, -move.i_mm)
    tangent_length = math.hypot(*tangent_at_start)
    tangent_alignment = (
        tangent_at_start[0] * incoming_vector[0]
        + tangent_at_start[1] * incoming_vector[1]
    ) / tangent_length
    expected_feed_mm_min = current_feed_mm_min * 0.60
    if abs(tangent_alignment - 1.0) > 1e-9:
        raise AssertionError("generated G03 arc is not tangent to the incoming vector")
    if abs(decision.feed_mm_min - expected_feed_mm_min) > 1e-9:
        raise AssertionError("simulated bypass did not reduce feed by exactly 40 percent")

    print(
        f"[SIM] Load spike {simulated_load_pct:.1f}% exceeds "
        f"{load_threshold_pct:.1f}% threshold."
    )
    print("[SIM] Verified tangent-continuous G03 bypass; proposed block:")
    print(
        f"G03 X{move.x_mm:.3f} Y{move.y_mm:.3f} Z{-move.depth_mm:.3f} "
        f"I{move.i_mm:.3f} J{move.j_mm:.3f} F{decision.feed_mm_min:.1f}"
    )
    print(
        f"[SIM] Feed reduced from {current_feed_mm_min:.1f} to "
        f"{decision.feed_mm_min:.1f} mm/min (40%)."
    )
    print("[SIM] Proposal only; no CNC controller commands were transmitted.")
    simulation_port.close()