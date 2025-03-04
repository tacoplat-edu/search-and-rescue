from models.devices import devices 

class ServoController():
    def __init__(self, pin) -> None:
        self.active_flag = False
        self.servo = devices.servo_motor
        self.servo.angle = 0  # Initialize the servo angle

    def set_servo_state(self, is_active: bool):
        self.active_flag = is_active

    def set_servo_angle(self, angle):
        self.servo.angle = angle

    def release_grip(self):
        if self.active_flag == True:
            self.set_servo_angle(0)
            self.set_servo_state(False)

    def grip(self):
        if self.active_flag == False:
            self.set_servo_angle(90)
            self.set_servo_state(True)
