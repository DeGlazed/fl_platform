from setuptools import setup, find_packages

setup(
    name='federated-learning-platform',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A federated learning platform implemented in Python.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/federated-learning-platform',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your project dependencies here
        'tensorflow',  # Example dependency
        'torch',       # Example dependency
        # Add other dependencies as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)