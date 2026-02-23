from setuptools import find_packages, setup

package_name = 'novigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rob1',
    maintainer_email='Robot@example.com',
    description='TODO: Package description',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'path_planner = novigation.path_planner:main',
            'test_map_publisher = novigation.test_map_publisher:main',
            'test_pose_publisher = novigation.test_pose_publisher:main',
            'test_goal_sender = novigation.test_goal_sender:main',
            'sim_robot = novigation.sim_robot:main',
            'novigation = novigation.novigation:main',

        ],
    },
)
