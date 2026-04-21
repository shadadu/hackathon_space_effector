# =======================================
# Base image: ROS Noetic (Ubuntu 20.04)
# =======================================
#FROM ros:noetic-ros-core
ARG ROS_PLATFORM=linux/amd64
FROM --platform=${ROS_PLATFORM} ros:noetic-ros-core
#FROM --platform=linux/amd64 ros:noetic-desktop-full

# Install basic tools and Python packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-catkin-tools \
    python3-rosdep \
    git \
    ros-noetic-rospy \
    ros-noetic-tf2 \
    ros-noetic-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    ros-noetic-geometry-msgs \
    ros-noetic-sensor-msgs \
    ros-noetic-std-msgs \
    ros-noetic-visualization-msgs \
    ros-noetic-rviz \
    ros-noetic-gazebo-ros \
    ros-noetic-gazebo-ros-pkgs \
    && rm -rf /var/lib/apt/lists/*


# Initialize rosdep (Added fix for potential double-init)
RUN if [ ! -d /etc/ros/rosdep/sources.list.d/ ]; then rosdep init; fi && rosdep update

# =======================================
# Set up catkin workspace
# =======================================
ENV CATKIN_WS=/root/catkin_ws
RUN mkdir -p $CATKIN_WS/src
WORKDIR $CATKIN_WS/src

# Copy the package
COPY ./astrobee_grasp ./astrobee_grasp

# Initialize workspace properly
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_init_workspace"

# =======================================
# Install Dependencies (CRITICAL STEP)
# =======================================
# Install ROS dependencies defined in package.xml
WORKDIR $CATKIN_WS
RUN apt-get update && rosdep install --from-paths src --ignore-src -r -y && rm -rf /var/lib/apt/lists/*
#
## Install desktop + VNC
#RUN apt-get update && apt-get install -y \
#    xfce4 \
#    xfce4-goodies \
#    tightvncserver \
#    dbus-x11 \
#    xterm \
#    ros-noetic-rviz \
#    ros-noetic-gazebo-ros-pkgs \
#    ros-noetic-gazebo-ros-control \
#    && rm -rf /var/lib/apt/lists/*

## Create VNC startup script
#RUN mkdir -p /root/.vnc
#
#RUN echo '#!/bin/bash\n\
#xrdb $HOME/.Xresources\n\
#startxfce4 &' > /root/.vnc/xstartup
#
#RUN chmod +x /root/.vnc/xstartup

## Expose VNC port
#EXPOSE 5901

# Install Python dependencies
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt



# =======================================
# Build the catkin workspace
# =======================================
# Only build ONCE after all dependencies are present
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Source ROS & catkin automatically for interactive shells
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN echo "source $CATKIN_WS/devel/setup.bash" >> /root/.bashrc

WORKDIR $CATKIN_WS
CMD ["roslaunch", "astrobee_grasp", "gazebo_perception.launch"]
