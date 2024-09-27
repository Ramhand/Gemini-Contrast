from setuptools import setup, find_packages

setup(
    name='gemini-contrast',
    version='0.1.0',
    description='Gemini: A Novel Deep Learning Architecture',
    author='Ramhand',
    author_email='anemoiwanaka@protonmail.com',
    url='https://github.com/Ramhand/gemini-contrast',
    packages=find_packages(),
    install_requires=[
        'torch>=2.4.1',
        'torchvision>=0.19.1',
        'numpy>=1.18.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
