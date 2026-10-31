import numpy as np

ARM_LENGTHS = (1.0, 1.0, 1.0)
FINGER_LENGTHS = (0.5, 0.5)
FINGER_OFFSET = 0.093   # the finger hinges sit at the two edges of the palm (0.54 of a 2.9-long link, drawn with length 0.5)


def finger_points(base, direction, proximal, distal, lengths=FINGER_LENGTHS):
    """Middle and tip of a finger hinged at `base`; angles are relative, as the joint angles of the simulator."""
    first = direction + proximal
    second = first + distal
    middle = base + lengths[0] * np.array([np.cos(first), np.sin(first)])
    tip = middle + lengths[1] * np.array([np.cos(second), np.sin(second)])
    return middle, tip


def arm_and_gripper(angles_deg, arm_lengths=ARM_LENGTHS, finger_lengths=FINGER_LENGTHS):
    """Planar stick figure of the 8-joint arm.

    angles_deg: shoulder, elbow, third arm joint, wrist, then proximal and distal joint of finger 1 and of
    finger 2 (the order of JOINT_POSITIONS). A zero shoulder angle points straight up; finger 1 is on the
    right-hand side of the hand direction, finger 2 on the left.
    Returns the arm polyline (4 points, base to the wrist hinge) and the gripper polyline
    (tip of finger 2, middle of finger 2, base of finger 2, base of finger 1, middle of finger 1, tip of finger 1).
    """
    angles = np.radians(np.asarray(angles_deg, dtype=float))
    angles[0] += np.pi / 2
    cumulative = np.cumsum(angles[:3])
    points = [np.zeros(2)]
    for length, angle in zip(arm_lengths, cumulative):
        points.append(points[-1] + length * np.array([np.cos(angle), np.sin(angle)]))
    arm = np.array(points)
    hinge = arm[-1]
    direction = cumulative[-1] + angles[3]
    right = np.array([np.sin(direction), -np.cos(direction)])
    base1, base2 = hinge + FINGER_OFFSET * right, hinge - FINGER_OFFSET * right
    middle1, tip1 = finger_points(base1, direction, angles[4], angles[5], finger_lengths)
    middle2, tip2 = finger_points(base2, direction, angles[6], angles[7], finger_lengths)
    return arm, np.vstack([tip2, middle2, base2, base1, middle1, tip1])
