# pip install joycon-python
import logging
import math
import struct
import threading
import time
import json
import signal
import numpy as np
from pyjoycon import JoyCon, get_R_id, get_L_id

class JoyConController:
    def __init__(
        self,
        motor_names,
        initial_position=None,
        l1=117.0,  # Length of first lever in mm
        l2=136.0,  # Length of second lever in mm
        *args,
        **kwargs,
    ):
    
        self.motor_names = motor_names
        self.initial_position = initial_position if initial_position else [90, 170, 170, 0, 0, 10]
        self.current_positions = dict(zip(self.motor_names, self.initial_position, strict=False))
        self.new_positions = self.current_positions.copy()

        # Inverse Kinematics parameters are used to compute x and y positions
        self.l1 = l1
        self.l2 = l2

        # x and y are coordinates of the axis of wrist_flex motor relative to the axis of the shoulder_pan motor in mm
        self.x, self.y = self._compute_position(
            self.current_positions["shoulder_lift"], self.current_positions["elbow_flex"]
        )
        
        self.joycon_L = JoyCon(*get_L_id())
        self.joycon_R = JoyCon(*get_R_id())
        logging.info(f"Connected to JoyCon")
        self.send_rumble(rumble=True)
        
        self.calibrate()
        
        self.running = True

        # Gamepad states
        self.axes = {
            "RX": 0.0,
            "RY": 0.0,
            "LX": 0.0,
            "LY": 0.0,
            "L2": 0.0,
            "R2": 0.0,
        }
        self.buttons = {
            "L2": 0,
            "R2": 0,
            "DPAD_LEFT": 0,
            "DPAD_RIGHT": 0,
            "DPAD_UP": 0,
            "DPAD_DOWN": 0,
            "X": 0,
            "O": 0,
            "T": 0,
            "S": 0,
            "L1": 0,
            "R1": 0,
            "SHARE": 0,
            "OPTIONS": 0,
            "PS": 0,
            "L3": 0,
            "R3": 0,
        }
        self.previous_buttons = self.buttons.copy()

        # PS4 Controller constants
        self.VENDOR_ID = 0x054C  # Sony
        self.PRODUCT_ID = 0x09CC  # DualShock 4 Wireless Controller

        

        
        # Gyro control mode variables
        self.gyro_mode = False
        self.gyro_reference = {"pitch": 0.0, "roll": 0.0}
        self.pitch_deg = 0.0
        self.roll_deg = 0.0

        # Start the thread to read inputs
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.read_loop, daemon=True)
        self.thread.start()

    def connect(self):
        try:
          
            self.running = True
        except OSError as e:
            logging.error(f"Unable to open device: {e}")
            self.device = None

    def disconnect(self):
        self.running = False
    
    def calibrate(self):
        num_samples = 500
        right_samples = []
        left_samples = []
        for _ in range(num_samples):
            status = self.joycon_L.get_status()
            status_R = self.joycon_R.get_status()
            accel = status['accel']
            accel_R = status_R['accel']
            rot = status['gyro']
            rot_R = status_R['gyro']
            joystick = status['analog-sticks']['left']
            joystick_R = status_R['analog-sticks']['right']

            left_samples.append([accel['x'], accel['y'], accel['z'], rot['x'], rot['y'], rot['z'], joystick['horizontal'], joystick['vertical']])
            right_samples.append([accel_R['x'], accel_R['y'], accel_R['z'], rot_R['x'], rot_R['y'], rot_R['z'], joystick_R['horizontal'], joystick_R['vertical']])
            #time.sleep(0.01)
        
        self.right_calibration_offset = np.mean(right_samples, axis=0)
        self.left_calibration_offset = np.mean(left_samples, axis=0)

       
    def read_loop(self):
        while self.running:
            try:
                status = self.joycon_L.get_status()
                status_r = self.joycon_R.get_status()
                self._process_gamepad_input_new(status,status_r)
                #accel = status['accel']
                #accel_R = status_R['accel']
                #rot = status['gyro']
                #rot_R = status_R['gyro']
                #joystick = status['analog-sticks']['left']
                #joystick_r = status_r['analog-sticks']['right']
                #print(json.dumps(joystick_r, indent=4))
                #time.sleep(1)
                
            except Exception as e:
                logging.error(f"Error reading from device: {e}")
                
                
    def _process_gamepad_input_new(self, status,status_r):
        with self.lock:
            
            joystick = status['analog-sticks']['left']
            joystick_r = status_r['analog-sticks']['right']
        
            # Normalize to -1.0 to 1.0
            self.axes["LX"] = self._filter_deadzone(joystick['horizontal'] - self.left_calibration_offset[6]) * 0.1
            self.axes["LY"] = self._filter_deadzone(joystick['vertical'] - self.left_calibration_offset[7])
            self.axes["RX"] = self._filter_deadzone(joystick_r['horizontal'] - self.right_calibration_offset[6]) #* 0.1
            self.axes["RY"] = self._filter_deadzone(joystick_r['vertical'] - self.right_calibration_offset[7])
            #print(self.axes["LX"])
             # Reset D-Pad buttons
            up = status['buttons']['left']['up']
            down = status['buttons']['left']['down']
            self.buttons["DPAD_UP"] = 0.005 if up == 1 else 0
            self.buttons["DPAD_DOWN"] = 0.005 if down == 1 else 0
            left = status['buttons']['left']['left']
            right = status['buttons']['left']['right']
            self.buttons["DPAD_LEFT"] = 0.005 if left == 1 else 0
            self.buttons["DPAD_RIGHT"] =  0.005 if right == 1 else 0
            zlpressed = status['buttons']['left']['zl']
            zrpressed = status_r['buttons']['right']['zr']
            self.axes["L2"] = 0.01 if zlpressed == 1 else 0
            self.axes["R2"] = 0.01  if zrpressed == 1 else 0
            
            self.buttons["H"] = status_r['buttons']['shared']['home']
            self.buttons["P"] = status_r['buttons']['shared']['plus']
            
            axes = self.axes.copy()
            buttons = self.buttons.copy()

            
        self._update_positions(axes, buttons)

    
    def toggle_gyro_mode(self):
        self.gyro_mode = not self.gyro_mode
        if self.gyro_mode:
            # Turn on gyro mode
            #self.light_bar_color = (0, 255, 0)  # Green
            #self.send_output_report(red=0, green=255, blue=0)
            # Record the current pitch and roll as reference points
            self.gyro_reference = {"pitch": self.pitch_deg, "roll": self.roll_deg}
            logging.info("Gyro control mode activated")
        else:
            # Turn off gyro mode
            #self.light_bar_color = (0, 0, 255)  # Blue
            #self.send_output_report(red=0, green=0, blue=255)
            logging.info("Gyro control mode deactivated")

    def _filter_deadzone(self, value):
        """
        Apply a deadzone to the joystick input to avoid drift.
        """
        value = value * 0.00002
        
        if abs(value) < 0.01:
            return 0
        
        if abs(value) > 1: 
            return 0
        return value

    def get_command(self):
        """
        Return the current motor positions after reading and processing inputs.
        """
        return self.current_positions.copy()

    def _update_positions(self, axes, buttons):
        # Compute new positions based on inputs
        speed = 0.4
        # TODO: speed can be different for different directions

        temp_positions = self.current_positions.copy()

        # Handle macro buttons
        # Buttons have assigned states where robot can move directly
        used_macros = False
        macro_buttons = ["H","P"]
        for button in macro_buttons:
            if buttons.get(button):
                temp_positions = self._execute_macro(button, temp_positions)
                temp_x, temp_y = self._compute_position(
                    temp_positions["shoulder_lift"], temp_positions["elbow_flex"]
                )
                correct_inverse_kinematics = True
                used_macros = True

        if not used_macros:
            # Map joystick inputs to motor positions
            if self.gyro_mode:
                # Use gyro data for wrist_flex and wrist_roll
                delta_pitch = self.pitch_deg - self.gyro_reference["pitch"]
                delta_roll = self.roll_deg - self.gyro_reference["roll"]
                scaling_factor = 1.0  # Adjust as needed
                temp_positions["wrist_flex"] += delta_pitch * scaling_factor
                temp_positions["wrist_roll"] += delta_roll * scaling_factor
                self.gyro_reference = {"pitch": self.pitch_deg, "roll": self.roll_deg}

            # Right joystick controls "wrist_roll" (left/right) and "wrist_flex" (up/down)
            temp_positions["wrist_roll"] += axes["RX"] * speed  # degrees per update
            temp_positions["wrist_flex"] -= axes["RY"] * speed  # degrees per update

            # L2 and R2 control gripper
            temp_positions["gripper"] -= axes["R2"]  # Close gripper
            temp_positions["gripper"] += axes["L2"]  # Open gripper

            # Left joystick and dpad left and right control shoulder_pan
            temp_positions["shoulder_pan"] += (
                axes["LX"] - buttons["DPAD_LEFT"] + buttons["DPAD_RIGHT"]
            ) #* speed  # degrees per update

            # Handle the linear movement of the arm
            # Left joystick up/down changes x
            temp_x = self.x + axes["LY"] * speed  # mm per update

            # D-pad up/down change y
            temp_y = self.y + (buttons["DPAD_UP"] - buttons["DPAD_DOWN"]) #* speed

            correct_inverse_kinematics = False

            # Compute shoulder_lift and elbow_flex angles based on x and y
            try:
                temp_positions["shoulder_lift"], temp_positions["elbow_flex"] = (
                    self._compute_inverse_kinematics(temp_x, temp_y)
                )
                shoulder_lift_change = (
                    temp_positions["shoulder_lift"] - self.current_positions["shoulder_lift"]
                )
                elbow_flex_change = temp_positions["elbow_flex"] - self.current_positions["elbow_flex"]
                temp_positions["wrist_flex"] += shoulder_lift_change - elbow_flex_change
                correct_inverse_kinematics = True
            except ValueError as e:
                logging.error(f"Error computing inverse kinematics: {e}")
        # Perform eligibility check
        if self._is_position_valid(temp_positions, temp_x, temp_y) and correct_inverse_kinematics:
            # Atomic update: all positions are valid, apply the changes
            
            self.current_positions = temp_positions
            self.x = temp_x
            self.y = temp_y
            #print(self.current_positions)
        else:
            # Invalid positions detected, do not update
            logging.warning("Invalid motor positions detected. Changes have been discarded.")
            self.indicate_error()

    def indicate_error(self):
        # Set light bar color to red and rumble
        self.send_rumble(rumble=True)

    
    def send_rumble(self, rumble = False):
        try:
            if rumble:
                joyconL = RumbleJoyCon(*get_L_id())
                joyconL.rumble_simple()    
        
        except Exception as e:
            logging.error(f"Error rumble: {e}")

    def _is_position_valid(self, positions, x, y):
        """
        Check if all positions are within their allowed ranges.
        Define the allowed ranges for each motor.
        """
        allowed_ranges = {
            "shoulder_pan": (-40, 190),
            "shoulder_lift": (-5, 185),
            "elbow_flex": (-5, 185),
            "wrist_flex": (-110, 110),
            "wrist_roll": (-110, 110),
            "gripper": (0, 100),
            "x": (15, 250),
            "y": (-110, 250),
        }

        for motor, (min_val, max_val) in allowed_ranges.items():
            if motor in positions and not (min_val <= positions[motor] <= max_val):
                logging.error(
                    f"Motor '{motor}' position {positions[motor]} out of range [{min_val}, {max_val}]."
                )
                return False

        # Check if x and y positions are within the allowed ranges
        if x < allowed_ranges["x"][0] or x > allowed_ranges["x"][1]:
            logging.error(f"X position {x} out of range {allowed_ranges['x']}.")
            return False

        if y < allowed_ranges["y"][0] or y > allowed_ranges["y"][1]:
            logging.error(f"Y position {y} out of range {allowed_ranges['y']}.")
            return False

        return True

    def _execute_macro(self, button, positions):
        """
        Define macros for specific buttons. When a macro button is pressed,
        set the motors to predefined positions.
        """
        macros = {
            "H": [0, 170, 170, 0, 0, 10],  # initial position
            "P": [0, 50, 130, -90, 90, 80],  # low horizontal gripper
            #"T": [90, 130, 150, 70, 90, 80],  # top down gripper
            #"S": [90, 160, 140, 20, 0, 0],  # looking forward
            # can add more macros for all other buttons
        }

        if button in macros:
            motor_positions = macros[button][:6]
            for name, pos in zip(self.motor_names, motor_positions, strict=False):
                positions[name] = pos
            logging.info(f"Macro '{button}' executed. Motors set to {motor_positions}.")
        return positions

    def _compute_inverse_kinematics(self, x, y):
        """
        Compute motor 2 and motor 3 angles based on the desired x and y positions.
        """
        # TODO: add explanation of the math behind this
        # TODO: maybe the math can be optimized, check it

        l1 = self.l1
        l2 = self.l2

        # Compute the distance from motor 2 to the desired point
        distance = math.hypot(x, y)

        # Check if the point is reachable
        if distance > (l1 + l2) or distance < abs(l1 - l2):
            raise ValueError(f"Point ({x}, {y}) is out of reach.")

        # Compute angle for motor3 (theta2)
        cos_theta2 = (l1**2 + l2**2 - distance**2) / (2 * l1 * l2)
        theta2_rad = math.acos(cos_theta2)
        theta2_deg = math.degrees(theta2_rad)
        # Adjust motor3 angle

        offset = math.degrees(math.asin(32 / l1))

        motor3_angle = 180 - (theta2_deg - offset)

        # Compute angle for motor2 (theta1)
        cos_theta1 = (l1**2 + distance**2 - l2**2) / (2 * l1 * distance)
        theta1_rad = math.acos(cos_theta1)
        theta1_deg = math.degrees(theta1_rad)
        alpha_rad = math.atan2(y, x)
        alpha_deg = math.degrees(alpha_rad)

        beta_deg = 180 - alpha_deg - theta1_deg
        motor2_angle = 180 - beta_deg + offset

        return motor2_angle, motor3_angle

    def _compute_position(self, motor2_angle, motor3_angle):
        """
        Compute the x and y positions based on the motor 2 and motor 3 angles.
        """
        l1 = self.l1
        l2 = self.l2
        offset = math.degrees(math.asin(32 / l1))

        beta_deg = 180 - motor2_angle + offset
        beta_rad = math.radians(beta_deg)

        theta2_deg = 180 - motor3_angle + offset
        theta2_rad = math.radians(theta2_deg)

        y = l1 * math.sin(beta_rad) - l2 * math.sin(beta_rad - theta2_rad)
        x = -l1 * math.cos(beta_rad) + l2 * math.cos(beta_rad - theta2_rad)
        return x, y

    def stop(self):
        """
        Clean up resources.
        """
        self.disconnect()
        self.thread.join()
        
class RumbleJoyCon(JoyCon):
    def __init__(self, *args, **kwargs):
        JoyCon.__init__(self,*args, **kwargs)
        
    def _send_rumble(self,data=b'\x00\x00\x00\x00\x00\x00\x00\x00'):
        self._RUMBLE_DATA = data
        self._write_output_report(b'\x10', b'', b'')

    def enable_vibration(self,enable=True):
        """Sends enable or disable command for vibration. Seems to do nothing."""
        self._write_output_report(b'\x01', b'\x48', b'\x01' if enable else b'\x00')
        
    def rumble_simple(self):
        """Rumble for approximately 1.5 seconds (why?). Repeat sending to keep rumbling."""
        self._send_rumble(b'\x98\x1e\xc6\x47\x98\x1e\xc6\x47')

    def rumble_stop(self):
        """Instantly stops the rumble"""
        self._send_rumble()

        
def main():
  ctl = JoyConController(['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'],[90, 170, 170, 0, 0, 10])
  ctl.thread.join()

if __name__ == "__main__":
    main()