class RescueState:
    is_rescue_complete: bool
    is_holding_figure: bool

    def __init__(self, is_rescue_complete = False, is_holding_figure = False):
        self.is_rescue_complete = is_rescue_complete
        self.is_holding_figure = is_holding_figure
        