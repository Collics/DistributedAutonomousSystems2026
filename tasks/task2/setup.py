from setuptools import find_packages, setup
from glob import glob

package_name = 'task2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ("share/" + package_name, glob("launch/task2_2_launch.py")),
        ('share/' + package_name, glob("launch/task2_3_launch.py")),
        ('share/' + package_name, glob('launch/rviz_visual_launch.py')),
        ('share/' + package_name, glob('config.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "task2_2_agent = task2.task2_2_agent:main",
            'task2_3_agent = task2.task2_3_agent:main',
            'central_visualizer = task2.centralized_visualizer:main',
        ],
    },
)
