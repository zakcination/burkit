# Run the Docker container and save the ID
docker run -itd --cpuset-cpus=64-95 -p 7778:7778 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=2,3,4,9 -v /raid/miras_zakaryanov/orddc2022:/workspace --name=miras_orddc2 --shm-size=32g nvcr.io/nvidia/pytorch:21.09-py3
