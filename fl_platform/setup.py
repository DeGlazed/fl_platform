from setuptools import setup, find_packages

setup(
    name='fl-platform',
    version='0.1.0',
    author='Radu Tanase',
    author_email='raducutanase@gmail.com',
    description='An asynchronous federated learning platform implemented in Python.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/DeGlazed/fl_platform',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # TODO
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)