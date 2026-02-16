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

# Create catkin workspace
RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws/src

# Copy ALL local ROS packages into the workspace
COPY catkin_ws_src/ /root/catkin_ws/src/

#WORKDIR /root/catkin_ws/src
#COPY object_tracking ./object_tracking
#COPY jacobian_server ./jacobian_server
#COPY panda_benchmark_moveit ./panda_benchmark_moveit






# Build
WORKDIR /root/catkin_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Auto-source
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

CMD ["bash"]
