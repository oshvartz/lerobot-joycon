from inputs import get_gamepad
#from __future__ import print_function


def normalize(value, deadzone=8000, max_value=32767):
    """Normalize axis values to the range -1 to 1."""
    if abs(value) < deadzone:
        return 0
    return round(value / max_value, 2)

try:
    while True:
        events = get_gamepad()
        print(".")
        for event in events:
            if "ABS_" in event.code:
                # Axis event
                normalized = normalize(event.state)
                print(f"{event.code}: {normalized}")
            elif "BTN_" in event.ev_type:
                # Button event
                print(f"{event.code}: {event.state}")
except KeyboardInterrupt:
    print("\nExiting...")
