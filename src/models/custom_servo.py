from gpiozero import PWMOutputDevice
import time

class CustomAngularServo:
    def __init__(self, pin, 
                 min_angle=-90, 
                 max_angle=90, 
                 min_pulse_width=1/1000, 
                 max_pulse_width=2/1000, 
                 frame_width=20/1000):
        """
        Initialize an angular servo with configurable parameters
        
        :param pin: GPIO pin number
        :param min_angle: Minimum angle of servo movement
        :param max_angle: Maximum angle of servo movement
        :param min_pulse_width: Minimum pulse width in seconds
        :param max_pulse_width: Maximum pulse width in seconds
        :param frame_width: PWM frame width in seconds
        """
        # Validate input parameters
        if min_pulse_width >= max_pulse_width:
            raise ValueError("min_pulse_width must be less than max_pulse_width")
        
        self._min_angle = min_angle
        self._max_angle = max_angle
        self._min_pulse_width = min_pulse_width
        self._max_pulse_width = max_pulse_width
        
        # Create PWM device
        self._pwm = PWMOutputDevice(
            pin=pin, 
            initial_value=0, 
            frequency=1/frame_width
        )
        
        # Initial state
        self._last_angle = None
        self._value = None
    
    def _angle_to_value(self, angle):
        """
        Convert angle to PWM value
        
        :param angle: Desired servo angle
        :return: Corresponding PWM value
        """
        # Constrain angle to valid range
        angle = max(self._min_angle, min(self._max_angle, angle))
        
        # Linear interpolation of angle to pulse width
        pulse_width = (
            self._min_pulse_width + 
            (angle - self._min_angle) / 
            (self._max_angle - self._min_angle) * 
            (self._max_pulse_width - self._min_pulse_width)
        )
        
        # Convert pulse width to duty cycle
        return pulse_width / (1/self._pwm.frequency)
    
    def angle(self, new_angle=None, speed=1):
        """
        Get or set servo angle with optional speed control
        
        :param new_angle: Target angle
        :param speed: Movement speed (1-0, where 1 is fastest)
        :return: Current angle if no new_angle provided
        """
        if new_angle is None:
            return self._last_angle
        
        # Validate speed
        speed = max(0, min(1, speed))
        
        # If no previous angle, move directly
        if self._last_angle is None:
            target_value = self._angle_to_value(new_angle)
            self._pwm.value = target_value
            self._last_angle = new_angle
            return
        
        # Smooth movement
        start_angle = self._last_angle
        steps = max(int(abs(new_angle - start_angle) * 10 * speed), 10)
        
        for i in range(steps + 1):
            interpolated_angle = start_angle + (new_angle - start_angle) * (i / steps)
            target_value = self._angle_to_value(interpolated_angle)
            self._pwm.value = target_value
            time.sleep(0.02)  # Small delay between steps
        
        self._last_angle = new_angle
    
    def close(self):
        """
        Close the PWM device
        """
        self._pwm.close()
    
    def __enter__(self):
        """
        Support context manager protocol
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Close PWM device when exiting context
        """
        self.close()