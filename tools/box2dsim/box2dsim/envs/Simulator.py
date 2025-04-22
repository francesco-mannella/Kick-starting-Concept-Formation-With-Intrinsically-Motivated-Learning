from . import JsonToPyBox2D as json2d
from .PID import PID
import time, sys, os, glob
from Box2D import b2ContactListener
from .mkvideo import vidManager
from matplotlib.path import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image, ImageDraw
import cv2

class ContactListener(b2ContactListener):
    def __init__(self, bodies):
        b2ContactListener.__init__(self)
        self.contact_db = {}
        self.bodies = bodies

        for h in bodies.keys():
            for k in bodies.keys():
                self.contact_db[(h, k)]= 0

    def BeginContact(self, contact):
        for name, body in self.bodies.items():
            if body == contact.fixtureA.body:
                bodyA = name
            elif body == contact.fixtureB.body:
                bodyB = name
            
        self.contact_db[(bodyA, bodyB)] = len(contact.manifold.points)

    def EndContact(self, contact):
        for name, body in self.bodies.items():
            if body == contact.fixtureA.body:
                bodyA = name
            elif body == contact.fixtureB.body:
                bodyB = name
            
        self.contact_db[(bodyA, bodyB)] = 0

    def PreSolve(self, contact, oldManifold):
        pass

    def PostSolve(self, contact, impulse):
        pass


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class  Box2DSim(object):
    """ 2D physics using box2d and a json conf file
    """
    @staticmethod
    def loadWorldJson(world_file):
        jsw = json2d.load_json_data(world_file)
        return jsw

    def __init__(self, world_file=None, world_dict=None, dt=1/80.0, vel_iters=30, pos_iters=2):
        """
        Args:

            world_file (string): the json file from which all objects are created
            world_dict (dict): the json object from which all objects are created
            dt (float): the amount of time to simulate, this should not vary.
            pos_iters (int): for the velocity constraint solver.
            vel_iters (int): for the position constraint solver.

        """
        if world_file is not None:
            world, bodies, joints = json2d.createWorldFromJson(world_file)
        else:
            world, bodies, joints = json2d.createWorldFromJsonObj(world_dict)

        self.contact_listener = ContactListener(bodies)
        
        self.dt = dt
        self.vel_iters = vel_iters
        self.pos_iters = pos_iters
        self.world = world
        self.world.contactListener = self.contact_listener
        self.bodies = bodies
        self.joints = joints
        self.joint_pids = { ("%s" % k): PID(dt=self.dt)
                for k in list(self.joints.keys()) }

        def is_body_visible(body):
            return not (body[1].color[0] == 1 and body[1].color[1] == 1 and body[1].color[2] == 1)
        self.visible_bodies = dict(filter(is_body_visible, bodies.items()))

    def contacts(self, bodyA, bodyB):
        """ Read contacts between two parts of the simulation

        Args:

            bodyA (string): the name of the object A
            bodyB (string): the name of the object B

        Returns:

            (int): number of contacts
        """
        c1 = 0
        c2 = 0
        db =  self.contact_listener.contact_db 
        if (bodyA, bodyB) in db.keys(): 
            c1 = self.contact_listener.contact_db[(bodyA, bodyB)]
        if (bodyB, bodyA) in db.keys(): 
            c2 = self.contact_listener.contact_db[(bodyB, bodyA)]

        return c1 + c2

    def move(self, joint_name, angle):
        """ change the angle of a joint

        Args:

            joint_name (string): the name of the joint to move
            angle (float): the new angle position

        """
        pid = self.joint_pids[joint_name]
        pid.setpoint = angle

    def step(self):
        """ A simulation step
        """
        for key in list(self.joints.keys()):
            self.joint_pids[key].step(self.joints[key].angle)
            self.joints[key].motorSpeed = (self.joint_pids[key].output)
        self.world.Step(self.dt, self.vel_iters, self.pos_iters)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class VisualSensor:
    """ Compute the retina state at each ste of simulation
    """

    def __init__(self, sim, shape, rng):
        """
        Args:

            sim (Box2DSim): a simulator object
            shape (int, int): width, height of the retina in pixels
            rng (float, float): x and y range in the task space

        """

        self.shape = list(shape)
        self.n_pixels = self.shape[0] * self.shape[1]

        # make a canvas with coordinates
        x = np.arange(-self.shape[0]//2, self.shape[0]//2) + 1
        y = np.arange(-self.shape[1]//2, self.shape[1]//2) + 1
        X, Y = np.meshgrid(x, y[::-1])
        self.grid = np.vstack((X.flatten(), Y.flatten())).T
        self.scale = np.array(rng)/shape
        self.radius = np.mean(np.array(rng)/shape)
        self.retina = np.zeros(self.shape + [3])

        self.reset(sim);

    def reset(self, sim):
        self.sim = sim

    def step(self, focus) :
        """ Run a single simulator step

        Args:

            sim (Box2DSim): a simulator object
            focus (float, float): x, y of visual field center

        Returns:

            (np.ndarray): a rescaled retina state
        """
        self.retina *= 0
        for body in self.sim.visible_bodies.values():
            if body.color is None:
                body.color = [0.5, 0.5, 0.5]
            color = np.array(body.color)

            data = np.array([body.GetWorldPoint(v) for v in body.fixtures[0].shape.vertices])
            vertices_t = np.round((data - focus) / self.scale) \
                    + [(self.shape[0]-1)//2, -self.shape[1]//2]
            vertices_t[:, 1] = -vertices_t[:, 1]
            cv2.fillPoly(self.retina, pts=[vertices_t.astype(np.int32)], color=1 - color)

        self.retina = np.maximum(0, 1 - (self.retina))
        return self.retina

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def merge_frames(frame1, frame2, alphacolor=(255,255,255,255)):

    frame1.putalpha(255)
    pixdata = np.asarray(frame1).copy()
    frame1 = Image.fromarray(pixdata)

    frame2.putalpha(255)
    pixdata = np.asarray(frame2).copy()
    whites = np.all(pixdata == np.reshape(alphacolor, (1, 1, -1)), axis=-1)
    pixdata[whites] = 0
    frame2 = Image.fromarray(pixdata)

    return Image.alpha_composite(frame1, frame2)

def concat_frames_h(im1, im2):
    dst = Image.new('RGBA', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


class TestPlotter:
    """ Plotter of simulations
    Builds a simple matplotlib graphic environment
    and render single steps of the simulation within it

    """

    def __init__(self, env, xlim=[-10, 30], ylim=[-10, 30],
                 int_xlim=[0, 10], int_ylim=[0, 10],
                 figsize=None, offline=False):
        """
        Args:
            env (Box2DSim): a emulator object
        """

        self.env = env
        self.offline = offline
        self.xlim = xlim
        self.ylim = ylim

        self.int_xlim = int_xlim
        self.int_ylim = int_ylim

        if figsize is None:
            self.fig = plt.figure()
        else:
            self.fig = plt.figure(figsize=figsize)
        
        if self.offline:
            self.vm = vidManager(self.fig, name="frame", duration=200)

        self.ax = None

        self.reset()

    def close(self, name=None):
        plt.close(self.fig)
        if self.offline and name is not None:
            self.vm.mk_video(name=name, dirname=".")
        self.vm = None


    def reset(self):

        if self.ax is not None:
            plt.delaxes(self.ax)
        self.ax = self.fig.add_subplot(111, aspect="equal")
        self.polygons = {}
        for key in self.env.sim.bodies.keys() :
            self.polygons[key] = Polygon([[0, 0]],
                    ec=self.env.sim.bodies[key].color + [1],
                    fc=self.env.sim.bodies[key].color + [1],
                    closed=True)

            self.ax.add_artist(self.polygons[key])

        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.axis("off")
        if not self.offline:
            self.fig.show()
        else:
            self.ts = 0


    def onStep(self):
        pass

    def step(self) :
        """ Run a single emulator step
        """

        for key in self.polygons:
            body = self.env.sim.bodies[key]
            vercs = np.vstack(body.fixtures[0].shape.vertices)
            data = np.vstack([ body.GetWorldPoint(vercs[x])
                for x in range(len(vercs))])
            self.polygons[key].set_xy(data)

        self.onStep()

        if not self.offline:
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
        else:
            self.fig.canvas.draw()
            self.vm.save_frame()
            self.ts += 1

    def add_info_to_frames(self, match_value, max_match, cum_match,
                           f_vp, f_ssp, f_pp, f_ap, f_gp, visual_map_path=None):
        # Make the rendered frames and info matching lengths
        if len(match_value) < len(self.vm.frames):
            self.vm.frames = self.vm.frames[:len(match_value)]

        n_steps = max(len(self.vm.frames), len(match_value))

        ax_r = None
        last_goal_reset = 0
        for i in range(n_steps):
            if i > 0 and cum_match[i] - cum_match[i-1] < 0:
                last_goal_reset = i

            # Plot match values
            if self.ax is not None:
                plt.delaxes(self.ax)
            self.ax = self.fig.add_subplot(111, aspect="equal")
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)
            self.ax.axis("off")

            self.ax.text(self.xlim[0], 0.9*self.ylim[1], f"t={i}", fontsize="large")

            # Current max match
            #self.ax.bar(
            #        self.xlim[0] + 0.8*(self.xlim[1] - self.xlim[0]),
            #        max_match[i]*self.ylim[1],
            #        )
            # Current cummulative success
            self.ax.bar(
                    self.xlim[0],
                    self.ylim[0] + cum_match[i] * (self.ylim[1] - self.ylim[0]),
                    bottom=self.ylim[0],
                    width=2
                    )
            self.ax.text(self.xlim[0] - 0.3, self.ylim[0] + (self.ylim[1] - self.ylim[0])*0.5, "cumulated touch", rotation=90,
                         fontsize="small", horizontalalignment="right",
                         verticalalignment="center")
            self.fig.subplots_adjust(left=0.15, bottom=0.25, right=0.85, top=0.9) 
            self.fig.canvas.draw()

            frame2 = Image.frombytes('RGBA', 
                    self.fig.canvas.get_width_height(), 
                    self.fig.canvas.buffer_rgba())
            
            merged_frame = merge_frames(self.vm.frames[i], frame2)

            # Plot internal representations
            if self.ax is not None:
                plt.delaxes(self.ax)
            self.ax = self.fig.add_subplot(111, aspect="equal")
            self.ax.set_xlim(self.int_xlim)
            self.ax.set_ylim(self.int_ylim)
            self.ax.axis("off")

            if visual_map_path is not None:
                im = plt.imread(visual_map_path)
                im = np.rot90(im)
                self.ax.imshow(im, alpha=0.3, aspect="auto", interpolation='nearest', extent=(0, 10, 0, 10))

            # Current match value
            self.ax.bar(
                    self.int_xlim[0] + 0.1,
                    self.int_ylim[0] + match_value[i] * (self.int_ylim[1] - self.int_ylim[0]),
                    bottom=self.int_ylim[0],
                    width=0.2
                    )
            self.ax.text(self.int_xlim[0] - 0.3, self.int_ylim[0] + (self.int_ylim[1] - self.int_ylim[0])*0.5, "match", rotation=90,
                         fontsize="small", horizontalalignment="right",
                         verticalalignment="center")

            self.ax.scatter(f_gp[i, 0], f_gp[i, 1], marker="s", label="goal", color="r", s=80)
            self.ax.scatter(f_vp[i, 0], f_vp[i, 1], marker="s", label="visual", color="b")
            self.ax.scatter(f_ssp[i, 0], f_ssp[i, 1], marker="s", label="somatosensory", color="g")
            self.ax.scatter(f_pp[i, 0], f_pp[i, 1], marker="s", label="proprioception", color="c")
            self.ax.scatter(f_ap[i, 0], f_ap[i, 1], marker="s", label="action", color="m")

            max_trace = 25
            t0 = i - max_trace
            if t0 < last_goal_reset:
                t0 = last_goal_reset
            for t in range(t0, i):
                self.ax.plot(f_vp[t:t+2, 0], f_vp[t:t+2, 1], color="b", alpha=(1.0-(t/max_trace))*0.5)
                self.ax.plot(f_ssp[t:t+2, 0], f_ssp[t:t+2, 1], color="g", alpha=(1.0-(t/max_trace))*0.5)
                self.ax.plot(f_pp[t:t+2, 0], f_pp[t:t+2, 1], color="c", alpha=(1.0-(t/max_trace))*0.5)
                self.ax.plot(f_gp[t:t+2, 0], f_gp[t:t+2, 1], color="r", alpha=(1.0-(t/max_trace))*0.5)
                self.ax.plot(f_ap[t:t+2, 0], f_ap[t:t+2, 1], color="m", alpha=(1.0-(t/max_trace))*0.5)

            self.ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize="small")
            self.fig.subplots_adjust(left=0.15, bottom=0.25, right=0.85, top=0.9) 
            self.fig.canvas.draw()

            frame2 = Image.frombytes('RGBA', 
                    self.fig.canvas.get_width_height(), 
                    self.fig.canvas.buffer_rgba())

            merged_frame = concat_frames_h(merged_frame, frame2)
            self.vm.frames[i] = merged_frame


class TestPlotterOneEye(TestPlotter):
    def __init__(self, *args, **kargs):

        super(TestPlotterOneEye, self).__init__(*args, **kargs)
        self.eye_pos, = self.ax.plot(0, 0, color="#888800")

    def reset(self):
        super(TestPlotterOneEye, self).reset()
        self.eye_pos, = self.ax.plot(0, 0, color="#888800")

    def onStep(self):

        pos = np.copy(self.env.eye_pos)
        x = pos[0] + np.array([-1, -1, 1,  1, -1])*self.env.fovea_height*0.5
        y = pos[1] + np.array([-1,  1, 1, -1, -1])*self.env.fovea_width*0.5
        self.eye_pos.set_data(x, y)



class TestPlotterVisualSalience(TestPlotterOneEye):

    def __init__(self, *args, **kargs):
        figsize = kargs["figsize"]  

        if figsize is None:
            self.fig_vis = plt.figure()
            self.fig_sal = plt.figure()
        else:
            self.fig_vis = plt.figure(figsize=figsize)
            self.fig_sal = plt.figure(figsize=figsize)
        
        self.vm_vis = vidManager(self.fig_vis, name="frame", dirname="visframes",  duration=30)
        self.vm_sal = vidManager(self.fig_sal, name="frame", dirname="salframes",   duration=30)
        self.ax_vis = self.fig_vis.add_subplot(111, aspect="equal")
        self.ax_sal = self.fig_sal.add_subplot(111, aspect="equal")
        self.ax_vis.set_axis_off()
        self.ax_sal.set_axis_off()
        self.fig_vis.tight_layout(pad=0)
        self.fig_sal.tight_layout(pad=0)


        self.vis_img = self.ax_vis.imshow(np.ones([10, 10, 3]))
        self.sal_img = self.ax_sal.imshow(np.ones([10, 10]), vmin=0, vmax=1, cmap=plt.cm.binary)
        super(TestPlotterVisualSalience, self).__init__(*args, **kargs)

   
    def onStep(self):

        super(TestPlotterVisualSalience, self).onStep()
       
        if self.offline:

            vis = self.env.observation["VISUAL_SENSORS"]
            sal = self.env.observation["VISUAL_SALIENCY"]
            self.sal_img.set_clim(
                    sal.min(),
                    sal.max(),
                    )

            self.vis_img.set_array(vis)
            self.sal_img.set_array(sal)

            self.fig_vis.canvas.draw()
            self.fig_sal.canvas.draw()
            self.vm_vis.save_frame()
            self.vm_sal.save_frame()
    
    def close(self, name=None):
        super(TestPlotterVisualSalience, self).close(name)

        plt.close(self.fig_vis)
        plt.close(self.fig_sal)
        if self.offline and name is not None:
            self.vm_vis.mk_video(name=f"{name}_vis", dirname=".")
            self.vm_sal.mk_video(name=f"{name}_sal", dirname=".")
        self.vm_vis = None
        self.vm_sal = None
