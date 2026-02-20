FROM ros:noetic

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ros-noetic-moveit-commander \
    ros-noetic-rviz \
    ros-noetic-franka-description \
    ros-noetic-moveit-resources-panda-description \
    ros-noetic-moveit-resources-panda-moveit-config \
    ros-noetic-robot-state-publisher \
    ros-noetic-joint-state-publisher \
    python3-catkin-tools \
    python3-pip \
    build-essential \
    ros-noetic-moveit \
    ros-noetic-moveit-core \
    ros-noetic-geometry-msgs \
    ros-noetic-robot-state-publisher \
    ros-noetic-joint-state-publisher \
    && rm -rf /var/lib/apt/lists/*

#RUN #python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel
# CPU-only PyTorch (works in most docker-on-mac setups)
#RUN #python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# ---- Python ML deps ----
RUN apt-get update && apt-get install -y python3-pip && \
    pip3 install --no-cache-dir torch torchvision



# Create catkin workspace
RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws/src

# Copy ALL local ROS packages into the workspace
COPY catkin_ws_src/ /root/catkin_ws/src/


# Build
WORKDIR /root/catkin_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Auto-source
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

CMD ["bash"]