import setuptools

import versioneer

with open('DESCRIPTION.rst', mode='r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name = "skw-full-code",
    version = versioneer.get_version(),
    packages = setuptools.find_packages('src'),
    package_dir = {'': 'src'},
    install_requires = [
        "numpy",
        "matplotlib>=2.2",
        "scikits.odes>=2.3.0dev0",
        "logbook",
        "arrow",
        "h5py>2.5",
        "h5preserve>=0.14",
        "stringtopy",
        "attrs",
        "disc-solver>=0.1.1",
    ],
    author = "James Tocknell",
    author_email = "aragilar@gmail.com",
    description = "Solver for jet solutions in PPDs",
    long_description = long_description,
    license = "GPLv3+",
    url = "http://disc-solver.rtfd.org",
    requires_python = ">=3.6.*",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    entry_points = {
        'console_scripts': [
            'skw-full-soln = skw_full_code.solve:main',
            'skw-full-info = skw_full_code.analyse.info:info_main',
            'skw-full-plot = skw_full_code.analyse.plot:plot_main',
        ],
    },
    cmdclass=versioneer.get_cmdclass(),
)
