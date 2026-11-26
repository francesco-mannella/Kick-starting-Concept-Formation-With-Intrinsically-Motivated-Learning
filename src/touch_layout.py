CLASSIC = (slice(10, 20), slice(20, 30))
SKETCH = (slice(6, 20), slice(20, 34))


def inner_slices(obj_set):
    """Sensor index ranges of the inner face of the left and of the right finger."""
    return SKETCH if obj_set == "grasp3s" else CLASSIC

# sensor positions (x, y in [0.1, 0.9], y up) in sensor order, and the finger links, read off the drawing sensors.jpg
SKETCH_POINTS = [[0.222, 0.276], [0.153, 0.351], [0.1, 0.44], [0.117, 0.581], [0.184, 0.73], [0.259, 0.875], [0.404, 0.141], [0.379, 0.18], [0.36, 0.21], [0.345, 0.236], [0.212, 0.381], [0.19, 0.419], [0.173, 0.454], [0.159, 0.48], [0.138, 0.507], [0.226, 0.681], [0.247, 0.722], [0.266, 0.762], [0.287, 0.811], [0.308, 0.858], [0.455, 0.143], [0.487, 0.166], [0.524, 0.195], [0.55, 0.217], [0.702, 0.364], [0.738, 0.393], [0.772, 0.417], [0.801, 0.442], [0.829, 0.47], [0.773, 0.657], [0.762, 0.689], [0.752, 0.729], [0.737, 0.764], [0.726, 0.812], [0.679, 0.25], [0.767, 0.312], [0.854, 0.384], [0.9, 0.543], [0.852, 0.662], [0.795, 0.805]]
SKETCH_OUTLINE = [[[0.111, 0.511], [0.286, 0.88]], [[0.399, 0.108], [0.111, 0.511]], [[0.447, 0.1], [0.873, 0.48]], [[0.873, 0.48], [0.745, 0.845]]]


def sensor_outline(obj_set):
    """Finger links drawn behind the sensors of the touch plots (none for the classic layout)."""
    return SKETCH_OUTLINE if obj_set == "grasp3s" else []
