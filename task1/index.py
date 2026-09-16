import queue
import random
import threading
import time
from dataclasses import dataclass
from math import isfinite

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import Button, RadioButtons
import mplcursors

try:
    from digi.xbee.devices import XBeeDevice
    XBEE_INSTALLED = True
except ImportError:
    XBEE_INSTALLED = False


# --- Configuration ---
USE_MOCK = False
PORT = "COM3"
BAUD_RATE = 9600
QUEUE_MAXSIZE = 500
HISTORY_LIMIT = 2000
PLOT_WINDOW_SIZE = 120
MAX_POINTS_PER_FRAME = 200
LINK_TIMEOUT_SEC = 5.0
STATUS_PRINT_INTERVAL_SEC = 2.0

# Timestamp, State, Temperature, Pressure, Altitude, Battery Voltage,
# Battery Current, Latitude, Longitude, Prev_CMD_echo
PACKET_FIELD_COUNT = 10
STATE_NAMES = {0: "IDLE", 1: "LAUNCH", 2: "ASCENT", 3: "DESCENT", 4: "LANDED"}
STATE_COLORS = {0: "#5ac8fa", 1: "#ffd166", 2: "#06d6a0", 3: "#ff6b6b", 4: "#c77dff"}


@dataclass(frozen=True)
class Telemetry:
    timestamp: int
    state: int
    temperature: float
    pressure: float
    altitude: float
    battery_voltage: float
    battery_current: float
    latitude: float
    longitude: float
    command_echo: str


data_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
telemetry_history = []
stats_lock = threading.Lock()
stats = {"received": 0, "plotted": 0, "parse_errors": 0, "decode_errors": 0, "queue_drops": 0}


def parse_telemetry_packet(raw_str):
    """Parse the exact ten-field CSV telemetry contract."""
    parts = [part.strip() for part in raw_str.strip().split(",")]
    if len(parts) != PACKET_FIELD_COUNT:
        raise ValueError(f"Expected {PACKET_FIELD_COUNT} fields, got {len(parts)}")

    try:
        telemetry = Telemetry(
            timestamp=int(parts[0]),
            state=int(parts[1]),
            temperature=float(parts[2]),
            pressure=float(parts[3]),
            altitude=float(parts[4]),
            battery_voltage=float(parts[5]),
            battery_current=float(parts[6]),
            latitude=float(parts[7]),
            longitude=float(parts[8]),
            command_echo=parts[9],
        )
    except ValueError as exc:
        raise ValueError("Telemetry contains an invalid numeric field") from exc

    numeric_values = (
        telemetry.temperature,
        telemetry.pressure,
        telemetry.altitude,
        telemetry.battery_voltage,
        telemetry.battery_current,
        telemetry.latitude,
        telemetry.longitude,
    )
    if not all(isfinite(value) for value in numeric_values):
        raise ValueError("Telemetry contains a non-finite numeric field")

    if not -90 <= telemetry.latitude <= 90:
        raise ValueError(f"Latitude out of range: {telemetry.latitude}")
    if not -180 <= telemetry.longitude <= 180:
        raise ValueError(f"Longitude out of range: {telemetry.longitude}")
    if not -1000 <= telemetry.altitude <= 100000:
        raise ValueError(f"Altitude out of range: {telemetry.altitude}")
    return telemetry


def increment_stat(name, amount=1):
    with stats_lock:
        stats[name] += amount


def enqueue_telemetry(record):
    try:
        data_queue.put_nowait(record)
    except queue.Full:
        increment_stat("queue_drops")
        try:
            data_queue.get_nowait()
            data_queue.put_nowait(record)
        except (queue.Empty, queue.Full):
            pass


def xbee_data_callback(xbee_message):
    # Keep the Digi XBee receive thread short: decode, validate, and enqueue only.
    try:
        payload = xbee_message.data.decode("utf-8")
    except UnicodeDecodeError as exc:
        increment_stat("decode_errors")
        print(f"[XBee Decode Error] {exc}")
        return

    try:
        enqueue_telemetry(parse_telemetry_packet(payload))
        increment_stat("received")
    except ValueError as exc:
        increment_stat("parse_errors")
        print(f"[XBee Parse Error] {exc}")


def mock_telemetry_stream():
    lat, lon, altitude = 17.44, 78.37, 100.0
    battery_voltage = 12.0
    timestamp = 0
    while True:
        timestamp += 1
        altitude += random.uniform(-0.5, 1.5)
        lat += random.uniform(-0.0002, 0.0002)
        lon += random.uniform(-0.0002, 0.0002)
        battery_voltage = max(
            9.0,
            battery_voltage - random.uniform(0.002, 0.012) + random.uniform(-0.004, 0.004),
        )
        state = 2 if altitude > 105 else 1
        packet = (
            f"{timestamp},{state},15.2,951.0,{altitude:.2f},{battery_voltage:.3f},0.62,"
            f"{lat:.6f},{lon:.6f},SET_FREQ"
        )
        enqueue_telemetry(parse_telemetry_packet(packet))
        increment_stat("received")
        time.sleep(0.2)


def format_latitude(value, _position):
    return f"{abs(value):.2f}°{'N' if value >= 0 else 'S'}"


def format_longitude(value, _position):
    return f"{abs(value):.2f}°{'E' if value >= 0 else 'W'}"


def compute_limits(values, minimum_margin):
    low, high = min(values), max(values)
    if low == high:
        margin = max(abs(low) * 0.01, minimum_margin)
        return low - margin, high + margin
    margin = max((high - low) * 0.03, minimum_margin)
    return low - margin, high + margin


def initialize_receiver():
    device = None
    use_mock = USE_MOCK
    if not use_mock and XBEE_INSTALLED:
        try:
            device = XBeeDevice(PORT, BAUD_RATE)
            device.open()
            device.add_data_received_callback(xbee_data_callback)
            print(f"[XBee] Connected on {PORT} @ {BAUD_RATE} baud")
        except Exception as exc:
            print(f"[XBee Error] {exc}; falling back to mock telemetry")
            use_mock = True
    elif not use_mock:
        print("[Warning] digi-xbee is unavailable; falling back to mock telemetry")
        use_mock = True

    if use_mock:
        threading.Thread(target=mock_telemetry_stream, daemon=True).start()
        print("[Mock] Streaming synthetic telemetry")
    return device


def main():
    device = initialize_receiver()
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10), facecolor="#121212")
    fig.suptitle("CANSAT GROUND CONTROL CENTER", fontsize=16, fontweight="bold")

    banner = fig.add_axes([0.02, 0.92, 0.96, 0.055], facecolor="#20252b")
    banner.set_xticks([])
    banner.set_yticks([])
    banner_text = banner.text(
        0.02,
        0.5,
        "State: -- | MET: -- | Last CMD: -- | Link: WAITING",
        va="center",
        fontsize=11,
        family="monospace",
    )

    view_ax = fig.add_axes([0.03, 0.27, 0.57, 0.62], projection="3d", facecolor="#121212")
    readout_ax = fig.add_axes([0.63, 0.57, 0.34, 0.32], facecolor="#171b20")
    altitude_ax = fig.add_axes([0.63, 0.40, 0.16, 0.13], facecolor="#171b20")
    power_ax = fig.add_axes([0.80, 0.40, 0.16, 0.13], facecolor="#171b20")
    atmosphere_ax = fig.add_axes([0.63, 0.23, 0.16, 0.13], facecolor="#171b20")
    vertical_ax = fig.add_axes([0.80, 0.23, 0.16, 0.13], facecolor="#171b20")
    map_ax = fig.add_axes([0.63, 0.08, 0.34, 0.12], facecolor="#171b20")
    log_ax = fig.add_axes([0.03, 0.04, 0.57, 0.16], facecolor="#0b0d0f")
    profile_button_ax = fig.add_axes([0.88, 0.05, 0.09, 0.06])

    view_ax.set_title("PRIMARY 3D SPATIAL VIEWPORT — X: Longitude | Y: Latitude | Z: Altitude")
    view_ax.set_xlabel("Longitude (°E / °W)")
    view_ax.set_ylabel("Latitude (°N / °S)")
    view_ax.set_zlabel("Altitude (m)")
    view_ax.xaxis.set_major_formatter(FuncFormatter(format_longitude))
    view_ax.yaxis.set_major_formatter(FuncFormatter(format_latitude))
    view_ax.zaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.1f} m"))
    # Longitude/latitude form the ground plane; altitude is the vertical axis.
    view_ax.set_box_aspect((1.8, 1.0, 2.8))
    view_ax.view_init(elev=18, azim=-58)

    trajectory_segments = {}
    ground_line, = view_ax.plot([], [], [], color="#42d4f5", linestyle="--", linewidth=1.0)
    current_point = view_ax.scatter([], [], [], color="#ffffff", s=50)

    readout_ax.set_xticks([])
    readout_ax.set_yticks([])
    readout_ax.set_title("PERSISTENT TELEMETRY READOUT", loc="left", fontsize=10)
    readout_text = readout_ax.text(0.02, 0.98, "", va="top", family="monospace", fontsize=10,
                                  linespacing=1.5)

    altitude_ax.set_title("Altitude Profile", fontsize=9)
    altitude_ax.grid(alpha=0.2)
    (altitude_line,) = altitude_ax.plot([], [], color="#ff7f0e", linewidth=1.5)

    power_ax.set_title("Electrical Power", fontsize=9)
    power_ax.grid(alpha=0.2)
    power_ax2 = power_ax.twinx()
    power_ax2.set_ylabel("Current (A)", color="#ff5c5c")
    power_voltage_line, = power_ax.plot([], [], color="#42d4f5", linewidth=1.5, label="Voltage")
    power_current_line, = power_ax2.plot([], [], color="#ff5c5c", linewidth=1.5, label="Current")
    power_ax.legend(loc="upper left", fontsize=7)

    atmosphere_ax.set_title("Atmospheric Profile", fontsize=9)
    atmosphere_ax.grid(alpha=0.2)
    atmosphere_ax2 = atmosphere_ax.twinx()
    atmosphere_ax2.set_ylabel("Pressure (hPa)", color="#00d1ff")
    temperature_line, = atmosphere_ax.plot([], [], color="#ffb000", linewidth=1.5, label="Temp")
    pressure_line, = atmosphere_ax2.plot([], [], color="#00d1ff", linewidth=1.5, label="Pressure")
    atmosphere_ax.legend(loc="upper left", fontsize=7)

    vertical_ax.set_title("Vertical Rate", fontsize=9)
    vertical_ax.grid(alpha=0.2)
    (vertical_speed_line,) = vertical_ax.plot([], [], color="#8be9fd", linewidth=1.5)

    plot_windows = {}

    def style_popout_button(button):
        button.color = "#27313b"
        button.hovercolor = "#3b5366"
        button.label.set_color("#ffffff")
        button.label.set_fontsize(6)

    def popout_plot(name, source_axes, title, lines, y_label=None):
        existing = plot_windows.get(name)
        if existing is not None and plt.fignum_exists(existing.number):
            existing.show()
            return

        popout = plt.figure(figsize=(8, 4.5), facecolor="#121212")
        popout_ax = popout.add_subplot(111, facecolor="#171b20")
        popout_ax.set_title(title)
        popout_ax.grid(alpha=0.2)
        popout_lines = []
        for source_line, color, label in lines:
            line, = popout_ax.plot([], [], color=color, label=label)
            popout_lines.append((source_line, line))
        if y_label:
            popout_ax.set_ylabel(y_label)
        if len(popout_lines) > 1:
            popout_ax.legend()
        plot_windows[name] = popout
        popout.show()

        # Store the source axis and mirrored artists for live updates.
        popout._telemetry_axes = source_axes
        popout._telemetry_lines = popout_lines

    def add_popout_button(axis, name, title, lines, y_label=None):
        button_axis = axis.inset_axes([0.78, 0.76, 0.20, 0.22])
        button = Button(button_axis, "↗", color="#27313b", hovercolor="#3b5366")
        style_popout_button(button)
        button.on_clicked(
            lambda _event: popout_plot(name, axis, title, lines, y_label)
        )
        return button

    popout_buttons = [
        add_popout_button(
            altitude_ax,
            "altitude",
            "Altitude Profile",
            [(altitude_line, "#ff7f0e", "Altitude")],
            "Altitude (m)",
        ),
        add_popout_button(
            power_ax,
            "power",
            "Electrical Power",
            [
                (power_voltage_line, "#42d4f5", "Voltage"),
                (power_current_line, "#ff5c5c", "Current"),
            ],
        ),
        add_popout_button(
            atmosphere_ax,
            "atmosphere",
            "Atmospheric Profile",
            [
                (temperature_line, "#ffb000", "Temperature"),
                (pressure_line, "#00d1ff", "Pressure"),
            ],
        ),
        add_popout_button(
            vertical_ax,
            "vertical",
            "Vertical Rate",
            [(vertical_speed_line, "#8be9fd", "Vertical speed")],
            "m/s",
        ),
    ]

    map_ax.set_title("Latitude / Longitude Ground Track", fontsize=9)
    map_ax.set_xlabel("Longitude", fontsize=8)
    map_ax.set_ylabel("Latitude", fontsize=8)
    map_ax.set_xlim(-180, 180)
    map_ax.set_ylim(-90, 90)
    map_ax.set_aspect("equal", adjustable="box")
    map_ax.xaxis.set_major_formatter(FuncFormatter(format_longitude))
    map_ax.yaxis.set_major_formatter(FuncFormatter(format_latitude))
    map_ax.axvline(0, color="#777777", linewidth=0.8, linestyle=":")
    map_ax.axhline(0, color="#777777", linewidth=0.8, linestyle=":")
    map_ax.grid(alpha=0.18)
    (map_line,) = map_ax.plot([], [], color="#42d4f5", linewidth=1.2)
    map_point, = map_ax.plot([], [], marker="o", color="#ffffff", markersize=5)

    log_ax.set_title("COMMAND & TERMINAL LOGS", loc="left", fontsize=10)
    log_ax.set_xticks([])
    log_ax.set_yticks([])
    log_text = log_ax.text(0.01, 0.95, "", va="top", family="monospace", fontsize=8,
                           color="#b7f7c4")
    log_lines = []
    last_command = None

    profile_window = {"fig": None}
    profile_selection = {"x": "Longitude", "y": "Altitude"}

    def open_profile_window(_event=None):
        if profile_window["fig"] is not None and plt.fignum_exists(profile_window["fig"].number):
            profile_window["fig"].show()
            return
        profile_fig = plt.figure(figsize=(8, 5), facecolor="#121212")
        profile_ax = profile_fig.add_axes([0.12, 0.18, 0.62, 0.7])
        x_select_ax = profile_fig.add_axes([0.77, 0.55, 0.18, 0.28], facecolor="#1e1e1e")
        y_select_ax = profile_fig.add_axes([0.77, 0.18, 0.18, 0.28], facecolor="#1e1e1e")
        profile_line, = profile_ax.plot([], [], color="#42d4f5")
        x_radio = RadioButtons(x_select_ax, ("Latitude", "Longitude", "Altitude"), active=1)
        y_radio = RadioButtons(y_select_ax, ("Latitude", "Longitude", "Altitude"), active=2)
        x_select_ax.set_title("X axis", fontsize=9)
        y_select_ax.set_title("Y axis", fontsize=9)
        profile_window.update({
            "fig": profile_fig,
            "ax": profile_ax,
            "line": profile_line,
            "x_radio": x_radio,
            "y_radio": y_radio,
        })

        def choose_x(label):
            profile_selection["x"] = label

        def choose_y(label):
            profile_selection["y"] = label

        x_radio.on_clicked(choose_x)
        y_radio.on_clicked(choose_y)
        profile_fig.suptitle("Selectable Telemetry Profile")
        profile_fig.show()

    profile_button = Button(profile_button_ax, "Profiles")
    profile_button.color = "#27313b"
    profile_button.hovercolor = "#3b5366"
    profile_button.label.set_color("#ffffff")
    profile_button.label.set_fontsize(9)
    profile_button.on_clicked(open_profile_window)
    profile_window["button"] = profile_button

    camera_mode = {"mode": "payload_follow"}

    def set_camera_mode(mode):
        camera_mode["mode"] = mode
        if mode == "payload_follow":
            view_ax.view_init(elev=18, azim=-58)
        elif mode == "free_orbit":
            view_ax.view_init(elev=18, azim=-58)
        elif mode == "top_down":
            # Looking down the altitude/Z axis shows longitude vs latitude.
            view_ax.view_init(elev=90, azim=-90)

    camera_button_ax = fig.add_axes([0.70, 0.05, 0.16, 0.06], facecolor="#27313b")
    mode_button = Button(camera_button_ax, "Payload Follow")
    mode_button.color = "#27313b"
    mode_button.hovercolor = "#3b5366"
    mode_button.label.set_color("#ffffff")
    mode_button.label.set_fontsize(9)
    mode_button.on_clicked(lambda event: set_camera_mode("payload_follow"))

    def toggle_orbit(event):
        set_camera_mode("free_orbit")

    orbit_button_ax = fig.add_axes([0.88, 0.05, 0.09, 0.06], facecolor="#27313b")
    orbit_button = Button(orbit_button_ax, "Free Orbit")
    orbit_button.color = "#27313b"
    orbit_button.hovercolor = "#3b5366"
    orbit_button.label.set_color("#ffffff")
    orbit_button.label.set_fontsize(9)
    orbit_button.on_clicked(toggle_orbit)

    top_down_button_ax = fig.add_axes([0.70, 0.12, 0.16, 0.06], facecolor="#27313b")
    top_button = Button(top_down_button_ax, "Top-Down")
    top_button.color = "#27313b"
    top_button.hovercolor = "#3b5366"
    top_button.label.set_color("#ffffff")
    top_button.label.set_fontsize(9)
    top_button.on_clicked(lambda event: set_camera_mode("top_down"))

    cursor = mplcursors.cursor(current_point, hover=True)

    @cursor.connect("add")
    def on_hover(selection):
        index = int(selection.index)
        if index < len(telemetry_history):
            item = telemetry_history[index]
            selection.annotation.set_text(
                f"Alt: {item.altitude:.2f} m\nLat: {item.latitude:.6f}\nLon: {item.longitude:.6f}"
            )

    def update_profile_window():
        profile_fig = profile_window["fig"]
        if profile_fig is None or not plt.fignum_exists(profile_fig.number):
            return
        series = {
            "Latitude": ([item.latitude for item in telemetry_history], format_latitude),
            "Longitude": ([item.longitude for item in telemetry_history], format_longitude),
            "Altitude": ([item.altitude for item in telemetry_history], None),
        }
        x_name, y_name = profile_selection["x"], profile_selection["y"]
        if x_name == y_name:
            profile_window["ax"].set_title("Select two different axes")
            profile_window["line"].set_data([], [])
            return
        x_values, x_formatter = series[x_name]
        y_values, y_formatter = series[y_name]
        profile_window["line"].set_data(x_values, y_values)
        profile_window["ax"].set_xlabel(x_name)
        profile_window["ax"].set_ylabel(y_name)
        profile_window["ax"].xaxis.set_major_formatter(
            FuncFormatter(x_formatter) if x_formatter else FuncFormatter(lambda value, _: f"{value:.1f}")
        )
        profile_window["ax"].yaxis.set_major_formatter(
            FuncFormatter(y_formatter) if y_formatter else FuncFormatter(lambda value, _: f"{value:.1f}")
        )
        profile_window["ax"].relim()
        profile_window["ax"].autoscale_view()
        profile_fig.canvas.draw_idle()

    last_received_at = 0.0
    previous = None
    previous_time = None

    def update(_frame):
        nonlocal last_received_at, previous, previous_time, last_command
        drained = 0
        while drained < MAX_POINTS_PER_FRAME:
            try:
                item = data_queue.get_nowait()
            except queue.Empty:
                break
            telemetry_history.append(item)
            last_received_at = time.monotonic()
            drained += 1
            if item.command_echo and item.command_echo != last_command:
                log_lines.append(f"[{item.timestamp:06d}] CMD: {item.command_echo}")
                last_command = item.command_echo
        if len(telemetry_history) > HISTORY_LIMIT:
            del telemetry_history[:len(telemetry_history) - HISTORY_LIMIT]
        if len(log_lines) > 18:
            del log_lines[:len(log_lines) - 18]
        if not telemetry_history:
            return

        latest = telemetry_history[-1]
        state_name = STATE_NAMES.get(latest.state, f"STATE_{latest.state}")
        met = time.strftime("%H:%M:%S", time.gmtime(max(0, latest.timestamp)))
        link_ok = time.monotonic() - last_received_at <= LINK_TIMEOUT_SEC

        if previous is None:
            vertical_speed = 0.0
            pressure_trend = 0.0
            power_draw = latest.battery_voltage * latest.battery_current
        else:
            dt = latest.timestamp - previous.timestamp
            vertical_speed = (latest.altitude - previous.altitude) / dt if dt > 0 else 0.0
            pressure_trend = (latest.pressure - previous.pressure) / dt if dt > 0 else 0.0
            power_draw = latest.battery_voltage * latest.battery_current

        banner_text.set_text(
            f"State / Mode: {state_name:<10} | MET: {met} | "
            f'Last CMD: "{latest.command_echo or "--"}" | Link: {"OK" if link_ok else "STALE"}'
        )

        readout_text.set_text(
            f"MET:              {met}\n"
            f"State Badge:      {latest.state} ({state_name})\n"
            f"Altitude:         {latest.altitude:8.2f} m\n"
            f"Pressure:         {latest.pressure:8.2f} hPa\n"
            f"Temperature:      {latest.temperature:8.2f} °C\n"
            f"Battery Voltage:  {latest.battery_voltage:8.2f} V\n"
            f"Battery Current:  {latest.battery_current:8.2f} A\n"
            f"Lat / Lon:        {latest.latitude:8.4f}, {latest.longitude:8.4f}\n"
            f"Command Echo:     {latest.command_echo or '--'}\n"
            f"Vertical Speed:   {vertical_speed:8.2f} m/s\n"
            f"Power Draw:       {power_draw:8.2f} W\n"
            f"Pressure Trend:   {pressure_trend:8.2f} hPa/s"
        )
        log_text.set_text("\n".join(log_lines))

        spatial_history = telemetry_history
        plot_history = telemetry_history[-PLOT_WINDOW_SIZE:]
        times = [item.timestamp for item in plot_history]
        altitudes = [item.altitude for item in plot_history]
        latitudes = [item.latitude for item in plot_history]
        longitudes = [item.longitude for item in plot_history]
        temperatures = [item.temperature for item in plot_history]
        pressures = [item.pressure for item in plot_history]
        voltages = [item.battery_voltage for item in plot_history]
        currents = [item.battery_current for item in plot_history]
        vertical_speeds = []
        for idx, item in enumerate(plot_history):
            if idx == 0:
                if len(telemetry_history) > len(plot_history):
                    previous_item = telemetry_history[-len(plot_history) - 1]
                    delta_t = item.timestamp - previous_item.timestamp
                    vertical_speeds.append(
                        (item.altitude - previous_item.altitude) / delta_t if delta_t > 0 else 0.0
                    )
                else:
                    vertical_speeds.append(0.0)
            else:
                prev = plot_history[idx - 1]
                delta_t = item.timestamp - prev.timestamp
                vertical_speeds.append((item.altitude - prev.altitude) / delta_t if delta_t > 0 else 0.0)

        for state, artist in trajectory_segments.items():
            artist.remove()
        trajectory_segments.clear()

        for state in sorted(set(item.state for item in telemetry_history)):
            xs = [item.longitude for item in telemetry_history if item.state == state]
            ys = [item.latitude for item in telemetry_history if item.state == state]
            zs = [item.altitude for item in telemetry_history if item.state == state]
            if not xs:
                continue
            line, = view_ax.plot(xs, ys, zs, color=STATE_COLORS.get(state, '#ffffff'), linewidth=1.5)
            trajectory_segments[state] = line

        latest_lon = latest.longitude
        latest_alt = latest.altitude
        latest_lat = latest.latitude
        ground_line.set_data([latest_lon, latest_lon], [latest_lat, latest_lat])
        ground_line.set_3d_properties([latest_alt, 0.0])
        current_point._offsets3d = ([latest_lon], [latest_lat], [latest_alt])

        if camera_mode["mode"] == "payload_follow":
            view_ax.set_xlim(*compute_limits(longitudes, 1e-5))
            view_ax.set_ylim(*compute_limits(latitudes, 1e-5))
            view_ax.set_zlim(*compute_limits(altitudes + [0.0], 1.0))
        else:
            view_ax.set_xlim(*compute_limits(longitudes + [0.0], 1e-5))
            view_ax.set_ylim(*compute_limits(latitudes, 1e-5))
            view_ax.set_zlim(*compute_limits(altitudes + [0.0], 1.0))

        altitude_line.set_data(times, altitudes)
        power_voltage_line.set_data(times, voltages)
        power_current_line.set_data(times, currents)
        temperature_line.set_data(times, temperatures)
        pressure_line.set_data(times, pressures)
        vertical_speed_line.set_data(times, vertical_speeds)

        for axis in (altitude_ax, power_ax, atmosphere_ax, vertical_ax):
            axis.relim()
            axis.autoscale_view()

        for popout in list(plot_windows.values()):
            if not plt.fignum_exists(popout.number):
                continue
            for source_line, popout_line in popout._telemetry_lines:
                popout_line.set_data(source_line.get_xdata(), source_line.get_ydata())
            popout_ax = popout.axes[0]
            popout_ax.relim()
            popout_ax.autoscale_view()
            popout.canvas.draw_idle()

        map_line.set_data(
            [item.longitude for item in spatial_history],
            [item.latitude for item in spatial_history],
        )
        map_point.set_data([latest.longitude], [latest.latitude])

        update_profile_window()
        previous = latest
        previous_time = latest.timestamp
        increment_stat("plotted", drained)

    try:
        _animation = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
        plt.show()
    finally:
        if device is not None and device.is_open():
            device.close()
            print("[XBee] Connection closed")


if __name__ == "__main__":
    main()
