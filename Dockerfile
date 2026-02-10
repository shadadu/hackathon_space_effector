# =========================================================
# Base Image: ROS Noetic + Gazebo + RViz
# =========================================================
FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic

# =========================================================
# System Dependencies
# =========================================================
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    wget \
    curl \
    vim \
    tmux \
    x11-apps \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# ROS Python Tools
# =========================================================
RUN apt-get update && apt-get install -y \
    python3-catkin-tools \
    ros-noetic-tf2-ros \
    ros-noetic-robot-state-publisher \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# Create Catkin Workspace
# =========================================================
ENV CATKIN_WS=/root/catkin_ws
RUN mkdir -p ${CATKIN_WS}/src
WORKDIR ${CATKIN_WS}

# =========================================================
# Copy astrobee_grasp Package
# =========================================================
COPY astrobee_grasp ${CATKIN_WS}/src/astrobee_grasp

# =========================================================
# Python Dependencies (pip only)
# =========================================================
RUN pip3 install --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# =========================================================
# Build Catkin Workspace
# =========================================================
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    catkin_make"

# =========================================================
# Environment Setup
# =========================================================
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source ${CATKIN_WS}/devel/setup.bash" >> /root/.bashrc

# =========================================================
# Default Command
# =========================================================
CMD ["/bin/bash"]

# ================================
# Source ROS on container startup
# ================================
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

WORKDIR /root/catkin_ws

# ================================
# Default command
# ================================
CMD ["roslaunch", "astrobee_grasp", "perception.launch"]
