from setuptools import find_packages, setup

package_name = "brian"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="rob1",
    maintainer_email="Robot@example.com",
    description="TODO: Package description",
    license="MIT",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "brain = brian.brain:main",
            "dummy_server = brian.dummy_server:main",
            "explorer_goal_service = brian.explorer_goal_service:main",
        ],
    },
)
