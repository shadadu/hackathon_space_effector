# =======================================
# Base image: ROS Noetic (Ubuntu 20.04)
# =======================================
FROM ros:noetic-ros-core

# =======================================
# Install basic tools and Python packages
# =======================================
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-catkin-tools \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    git \
    wget \
    curl \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init || true
RUN rosdep update

# =======================================
# Set up catkin workspace
# =======================================
ENV CATKIN_WS=/root/catkin_ws
RUN mkdir -p $CATKIN_WS/src
WORKDIR $CATKIN_WS/src

# =======================================
# Copy the astrobee_grasp package
# =======================================
COPY ./astrobee_grasp ./astrobee_grasp

# REQUIRED for catkin
RUN echo 'cmake_minimum_required(VERSION 3.0.2)\n\
include(/opt/ros/noetic/share/catkin/cmake/toplevel.cmake)' \
> $CATKIN_WS/src/CMakeLists.txt


WORKDIR $CATKIN_WS
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"




# =======================================
# Copy requirements.txt and install Python deps
# =======================================
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# =======================================
# Build the catkin workspace
# =======================================
WORKDIR $CATKIN_WS
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# =======================================
# Source ROS & catkin automatically
# =======================================
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN echo "source $CATKIN_WS/devel/setup.bash" >> /root/.bashrc

# =======================================
# Default working directory
# =======================================
WORKDIR $CATKIN_WS

# =======================================
# Default command: launch Gazebo + perception
# =======================================
CMD ["roslaunch", "astrobee_grasp", "gazebo_perception.launch"]
