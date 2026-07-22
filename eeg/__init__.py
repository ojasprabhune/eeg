try:
    from .data_collection import Joint, JointData
except Exception:  # optional hardware deps (e.g. pywinusb) absent in containers
    Joint = JointData = None
