from setuptools import setup, find_packages
from setuptools.command.install import install as DistutilsInstall
from setuptools.command.egg_info import egg_info as EggInfo

class MyInstall(DistutilsInstall):
    def run(self):
        DistutilsInstall.run(self)

class MyEgg(EggInfo):
    def run(self):
        EggInfo.run(self)

setup(name='box2dsim',
        version='0.1',
        cmdclass = {
            'install': MyInstall,
            'egg_info': MyEgg
            },
        packages=find_packages(where="."),
        package_data={"": ["*.npy", "*.json"]},
        install_requires=['gymnasium', 'box2d_py', 'numpy', 
            'matplotlib','scikit-image']
        )
