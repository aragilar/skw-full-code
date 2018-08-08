import setuptools

import versioneer

#import codecs
#with codecs.open('DESCRIPTION.rst', 'r', 'utf-8') as f:
#    long_description = f.read()

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
        "corner",
        "attrs",
        "emcee",
    ],
    author = "James Tocknell",
    author_email = "aragilar@gmail.com",
    description = "Solver thing",
#    long_description = long_description,
    #license = "BSD",
    #keywords = "wheel",
    #url = "http://disc_solver.rtfd.org",
    #classifiers=[
    #    'Development Status :: 3 - Alpha',
    #    'Intended Audience :: Developers',
    #    "Topic :: System :: Shells",
    #    'License :: OSI Approved :: BSD License',
    #    'Programming Language :: Python :: 2',
    #    'Programming Language :: Python :: 2.6',
    #    'Programming Language :: Python :: 2.7',
    #    'Programming Language :: Python :: 3',
    #    'Programming Language :: Python :: 3.1',
    #    'Programming Language :: Python :: 3.2',
    #    'Programming Language :: Python :: 3.3',
    #    'Programming Language :: Python :: 3.4',
    #],
    entry_points = {
        'console_scripts': [
        ],
    },
    cmdclass=versioneer.get_cmdclass(),
)