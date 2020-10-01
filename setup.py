import setuptools
import shutil
import os


# path = os.path.dirname(os.path.abspath(__file__))
# shutil.copyfile(f"{path}/dmm.py", f"{path}/dmm/dmm.py")

setuptools.setup(
    name="DIOS-SSM",
    version="0.1",
    author="Ryosuke Kojima",
    author_email="kojima.ryosuke.8e@kyoto-u.ac.jp",
    description="controllable deep neural state-space model library",
    long_description="controllable deep neural state-space model library",
    long_description_content_type="text/markdown",
    url="https://github.com/clinfo/ConDeNS",
    packages=setuptools.find_packages(),
    entry_points={
        "console_scripts": [
            "dios= dios.dios:main",
            "dios-linear= dios.dios_linear:main",
            "dios-opt= dios.opt:main",
            "dios-opt-get= dios.opt_get:main",
            "dios-plot = dios.plot:main",
            "dios-field-plot = dios.dios_field_plot:main",
            "dios-map = dios.mapping:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
