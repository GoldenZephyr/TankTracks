class AutofocusMessage:
    def __init__(self, focus_min, focus_max, focus_step):
        """
        :param focus_min: minimum value for focus sweep
        :param focus_max: max value for focus sweep
        :param focus_step:  focus sweep step size
        """
        self.focus_min = focus_min
        self.focus_max = focus_max
        self.focus_step = focus_step


class SetFocusMessage:
    def __init__(self, focus):
        self.focus = focus


class SetFocusROI:
    def __init__(self, ul, lr):
        # upper left corner tuple, lower right corner tuple
        self.ul = ul
        self.lr = lr
