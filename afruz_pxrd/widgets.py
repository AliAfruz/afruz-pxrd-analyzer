from PySide6.QtWidgets import (
    QComboBox,
    QDial,
    QDoubleSpinBox,
    QSlider,
    QSpinBox,
)


class _NoWheelMixin:
    """
    Prevent mouse-wheel parameter changes.

    Ignoring the event lets a surrounding scroll area handle the wheel,
    while keyboard entry, arrow buttons, and deliberate slider dragging
    continue to work normally.
    """

    def wheelEvent(self, event):
        event.ignore()


class NoWheelDoubleSpinBox(_NoWheelMixin, QDoubleSpinBox):
    pass


class NoWheelSpinBox(_NoWheelMixin, QSpinBox):
    pass


class NoWheelComboBox(_NoWheelMixin, QComboBox):
    pass


class NoWheelSlider(_NoWheelMixin, QSlider):
    pass


class NoWheelDial(_NoWheelMixin, QDial):
    pass
