# dependencies/modeling/leave_one_group_out_cv.py
from sklearn.model_selection import LeaveOneGroupOut


def leave_one_group_out_cv() -> LeaveOneGroupOut:
    """Return Leave-One-Group-Out CV splitter."""
    return LeaveOneGroupOut()
