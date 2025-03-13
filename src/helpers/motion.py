def assert_speed(func):
    def wrapper_assert_speed(*args, **kwargs):
        speed = args[1] if args else kwargs["speed"] or -1
        try:
            assert 0 <= speed <= 1
            func(*args, **kwargs)
        except AssertionError:
            print("Speed is not in range [0,1]")
    return wrapper_assert_speed
