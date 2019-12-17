from enum import Enum
# Maybe change for string code


class NetworkMessageType(Enum):
    # RECEIVE
    CHANGE_TEMPLATE = 10,
    ASK_SHAPE_INOUT = 20,

    # SEND
    TEMPLATE_CHANGED = 100,
    TEMPLATE_UNKNOWN = 110,
    ERROR_TEMPLATE_CHANGE = 120,
    COUNTDOWN_START = 130,
    COUNTDOWN_STOP = 140
