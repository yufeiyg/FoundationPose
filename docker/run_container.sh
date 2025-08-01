docker rm -f foundationpose
DIR=$(pwd)/../
xhost +  && docker run --gpus all --device /dev/bus/usb:/dev/bus/usb --privileged --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name foundationpose  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $DIR:$DIR -v /home:/home -v /mnt:/mnt -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp --privileged -v /dev/bus/usb:/dev/bus/usb --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE bibitbianchini/foundationpose:latest bash -c "cd $DIR && bash"
