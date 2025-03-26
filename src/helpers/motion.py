def assert_speed(func):
    def wrapper_assert_speed(*args, **kwargs):
        speed = args[1] if args else kwargs["speed"] or -1
        print("speed: ", speed)
        try:
            assert 0 <= abs(speed) <= 1
            func(*args, **kwargs)
        except AssertionError:
            func(0, 0)
            print("Speed is not in range [0,1]")
    return wrapper_assert_speed
