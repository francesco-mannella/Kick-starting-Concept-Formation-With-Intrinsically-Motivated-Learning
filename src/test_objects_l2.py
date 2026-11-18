import box2dsim  # noqa: F401
import numpy as np

from params import Parameters
from SMEnv import SMEnv

ROT = {0: 0.0, 90: 1.570796}


def make(object_set=None, rot=0):
    kwargs = {"obj_stretch_conditions": [1], "obj_rotation_conditions": [ROT[rot]]}
    if object_set is not None:
        kwargs["obj_set"] = object_set
    return SMEnv(0, Parameters(**kwargs))


def bbox(env, world):
    env.reset(world=world)
    body = env.b2d_env.sim.bodies[env.b2d_env.object_names[0]]
    verts = np.array([body.transform * v for v in body.fixtures[0].shape.vertices])
    return np.round(verts.max(0) - verts.min(0), 2), body


def observations(env, world, n=30):
    env.reset(world=world)
    action = np.array([0.3, -0.4, 0.5, 0.6, 0.8])
    return np.array([np.hstack([o["JOINT_POSITIONS"], o["TOUCH_SENSORS"], np.ravel(o["OBJ_POSITION"])])
                     for o in (env.step(action) for _ in range(n))])


def test_default_is_orig():
    assert Parameters().obj_set == "orig"
    for world in (1, 2, 3):
        assert np.array_equal(observations(make(), world), observations(make("orig"), world))


def test_l2_polygons():
    for rot, plate, bar in ((0, [6.5, 2.5], [0.5, 6.0]), (90, [2.5, 6.5], [6.0, 0.5])):
        env = make("l2", rot)
        assert bbox(env, 2)[0].tolist() == plate
        assert bbox(env, 3)[0].tolist() == bar


def test_l2_bodies_and_context_one():
    env = make("l2")
    assert bbox(env, 2)[1].type == 2 and bbox(env, 3)[1].type == 2 and bbox(env, 1)[1].type == 0
    assert bbox(env, 1)[0].tolist() == bbox(make("orig"), 1)[0].tolist()
    assert np.array_equal(observations(make("l2"), 1), observations(make("orig"), 1))


def test_position_per_kind():
    assert Parameters().obj_pos_kind is None
    pos = {"still": [10.7, 9.3], "movable": [14.0, 6.0], "controllable": [14.0, 12.0]}
    env = SMEnv(0, Parameters(obj_pos_kind=pos, obj_stretch_conditions=[1], obj_rotation_conditions=[0.0]))
    for world, name in ((1, "still"), (2, "movable"), (3, "controllable")):
        env.reset(world=world)
        centre = env.b2d_env.sim.bodies[name].position
        assert abs(centre[0] - pos[name][0]) < 0.05 and abs(centre[1] - pos[name][1]) < 0.05, (name, centre)


def test_triangle_design():
    pos = {"still": [7.0, 12.0], "movable": [10.7, 9.3], "controllable": [14.0, 4.0]}
    for stretch, rot in ((1, 0.0), (2, 1.570796)):
        env = SMEnv(0, Parameters(obj_set="tri", obj_pos_kind=pos, obj_stretch_conditions=[stretch], obj_rotation_conditions=[rot]))
        boxes = []
        for world, name in ((1, "still"), (2, "movable"), (3, "controllable")):
            env.reset(world=world)
            body = env.b2d_env.sim.bodies[name]
            verts = np.array([body.transform * v for v in body.fixtures[0].shape.vertices])
            boxes.append(np.round(verts.max(0) - verts.min(0), 2).tolist())
            assert len(body.fixtures[0].shape.vertices) == 3, name
            assert abs(body.position[0] - pos[name][0]) < 0.05 and abs(body.position[1] - pos[name][1]) < 0.05
            assert body.type == (0 if name == "still" else 2), (name, body.type)
        assert boxes[0] == boxes[1] == boxes[2], boxes


def test_mix_design():
    pos = {"still": [14.0, 12.0], "movable": [7.0, 12.0], "controllable": [14.0, 4.0]}
    sizes = {}
    for stretch, rot in ((1, 0.0), (2, 1.570796)):
        env = SMEnv(0, Parameters(obj_set="mix", obj_pos_kind=pos, obj_stretch_conditions=[stretch], obj_rotation_conditions=[rot]))
        for world, name, nverts in ((1, "still", 4), (2, "movable", 3), (3, "controllable", 3)):
            env.reset(world=world)
            body = env.b2d_env.sim.bodies[name]
            assert len(body.fixtures[0].shape.vertices) == nverts, (name, len(body.fixtures[0].shape.vertices))
            assert abs(body.position[0] - pos[name][0]) < 0.05 and abs(body.position[1] - pos[name][1]) < 0.05
            assert body.type == (0 if name == "still" else 2), (name, body.type)
            verts = np.array([body.transform * v for v in body.fixtures[0].shape.vertices])
            sizes[(stretch, name)] = np.round(verts.max(0) - verts.min(0), 2).tolist()
    assert sizes[(1, "still")] == [4.8, 2.1] and sizes[(1, "controllable")] == [4.0, 4.0] and sizes[(1, "movable")] == [7.74, 7.84], sizes


if __name__ == "__main__":
    test_mix_design()
    test_triangle_design()
    test_position_per_kind()
    test_default_is_orig()
    test_l2_polygons()
    test_l2_bodies_and_context_one()
    print("ok")
