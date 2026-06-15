ARG ROS_PLATFORM=linux/amd64
ARG JAX_VARIANT=cpu
FROM --platform=${ROS_PLATFORM} ros:noetic

ARG JAX_VARIANT=cpu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-catkin-tools \
    python3-pip \
    ros-noetic-actionlib-msgs \
    ros-noetic-geometry-msgs \
    ros-noetic-joint-state-publisher \
    ros-noetic-message-generation \
    ros-noetic-message-runtime \
    ros-noetic-moveit \
    ros-noetic-moveit-commander \
    ros-noetic-moveit-core \
    ros-noetic-moveit-msgs \
    ros-noetic-moveit-resources-panda-description \
    ros-noetic-moveit-resources-panda-moveit-config \
    ros-noetic-moveit-ros-move-group \
    ros-noetic-moveit-ros-planning \
    ros-noetic-moveit-ros-planning-interface \
    ros-noetic-nav-msgs \
    ros-noetic-robot-state-publisher \
    ros-noetic-roscpp \
    ros-noetic-roslaunch \
    ros-noetic-rospy \
    ros-noetic-rviz \
    ros-noetic-sensor-msgs \
    ros-noetic-shape-msgs \
    ros-noetic-std-msgs \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-tf2-ros \
    ros-noetic-trajectory-msgs \
    ros-noetic-xacro \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 20.04's apt pip is too old to resolve some modern manylinux wheels.
# Keep packaging-tool pins compatible with ROS Noetic's Python 3.8.
RUN pip3 install --no-cache-dir --upgrade \
    pip==24.0 \
    setuptools==68.2.2 \
    wheel==0.42.0 \
    importlib-metadata==6.8.0

# Install Python dependencies
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# JAX wheels are installed separately because CPU and CUDA builds use different
# release indexes on the Python 3.8-compatible JAX line.
RUN if [ "$JAX_VARIANT" = "cpu" ]; then \
      pip3 install --no-cache-dir \
        -f https://storage.googleapis.com/jax-releases/jax_releases.html \
        jax==0.4.12 \
        jaxlib==0.4.12; \
    elif [ "$JAX_VARIANT" = "cuda12" ]; then \
      pip3 install --no-cache-dir \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
        jax==0.4.12 \
        'jaxlib==0.4.12+cuda12.cudnn89' \
        nvidia-cublas-cu12 \
        nvidia-cuda-cupti-cu12 \
        nvidia-cuda-nvcc-cu12 \
        nvidia-cuda-runtime-cu12 \
        'nvidia-cudnn-cu12>=8.9,<9' \
        nvidia-cufft-cu12 \
        nvidia-cusolver-cu12 \
        nvidia-cusparse-cu12; \
    elif [ "$JAX_VARIANT" = "cuda11" ]; then \
      pip3 install --no-cache-dir \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
        jax==0.4.12 \
        'jaxlib==0.4.12+cuda11.cudnn86' \
        'nvidia-cublas-cu11>=11.11' \
        'nvidia-cuda-cupti-cu11>=11.8' \
        'nvidia-cuda-nvcc-cu11>=11.8' \
        'nvidia-cuda-runtime-cu11>=11.8' \
        'nvidia-cudnn-cu11>=8.8,<9' \
        'nvidia-cufft-cu11>=10.9' \
        'nvidia-cusolver-cu11>=11.4' \
        'nvidia-cusparse-cu11>=11.7'; \
    else \
      echo "Unsupported JAX_VARIANT=$JAX_VARIANT; expected cpu, cuda12, or cuda11" >&2; \
      exit 1; \
    fi

# Copy application source after large dependency layers so code edits rebuild quickly.
RUN mkdir -p /root/catkin_ws/src
COPY catkin_ws_src/ /root/catkin_ws/src/
WORKDIR /root/catkin_ws

# Runtime output directories and executable ROS Python nodes
RUN mkdir -p /root/catkin_ws/src/object_tracking/models \
    /root/catkin_ws/src/object_tracking/results \
    /root/catkin_ws/src/object_tracking/data && \
    find /root/catkin_ws/src -path '*/src/*.py' -exec chmod +x {} \;

# Build
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Auto-source
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

CMD ["bash"]
