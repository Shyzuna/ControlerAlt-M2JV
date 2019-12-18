from enum import Enum
# Maybe change for string code


class NetworkMessageType(Enum):
    # RECEIVE
    CHANGE_TEMPLATE = 10
    ASK_SHAPE_INOUT = 20
    MENU_MODE = 30
    MENU_CHECK = 40

    # SEND
    TEMPLATE_CHANGED = 100
    TEMPLATE_UNKNOWN = 110
    ERROR_TEMPLATE_CHANGE = 120
    COUNTDOWN_START = 130
    COUNTDOWN_STOP = 140
    MENU_MODE_OP = 150
    MENU_MODE_FAIL = 160
