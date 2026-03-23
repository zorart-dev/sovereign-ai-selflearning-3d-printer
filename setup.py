"""
Setup configuration for Sovereign-v5.0
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements_path = Path(__file__).parent / 'requirements.txt'
requirements = [line.strip() for line in requirements_path.read_text().split('\n') 
                if line.strip() and not line.startswith('#')]

# Read README
readme_path = Path(__file__).parent / 'README.md'
long_description = readme_path.read_text(encoding='utf-8')

setup(
    name='sovereign-v5',
    version='5.0.0',
    description='Autonomous Edge AI for 3D Printers using PPO and LSTM',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Autonomous Systems Lab',
    author_email='info@autonomous.ai',
    url='https://github.com/anthropic/sovereign',
    license='MIT',
    
    packages=find_packages(),
    
    python_requires='>=3.9',
    install_requires=requirements,
    
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
            'black>=23.0',
            'flake8>=6.0',
            'mypy>=1.0',
            'sphinx>=5.0',
        ],
        'octoprint': [
            'requests>=2.28.0',
        ],
        'serial': [
            'pyserial>=3.5',
        ],
    },
    
    entry_points={
        'console_scripts': [
            'sovereign=main:main',
        ],
    },
    
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Manufacturing',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Manufacturing',
    ],
    
    keywords='reinforcement-learning ppo lstm 3d-printer autonomous ai edge-computing',
    
    project_urls={
        'Bug Reports': 'https://github.com/anthropic/sovereign/issues',
        'Documentation': 'https://github.com/anthropic/sovereign#readme',
        'Source Code': 'https://github.com/anthropic/sovereign',
    },
    
    include_package_data=True,
    zip_safe=False,
)
