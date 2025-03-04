class RescueState:
    is_rescue_complete: bool
    is_figure_held: bool

    def __init__(self, is_rescue_complete = False, is_figure_held = False):
        self.is_rescue_complete = is_rescue_complete
        self.is_figure_held = is_figure_held
