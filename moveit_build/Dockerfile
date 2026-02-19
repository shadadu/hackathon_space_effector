FROM ros:noetic

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    ros-noetic-moveit \
    ros-noetic-moveit-core \
    ros-noetic-moveit-commander \
    ros-noetic-rviz \
    ros-noetic-franka-description \
    ros-noetic-moveit-resources-panda-description \
    ros-noetic-moveit-resources-panda-moveit-config \
    ros-noetic-robot-state-publisher \
    ros-noetic-joint-state-publisher \
    ros-noetic-geometry-msgs \
    python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
