# The Final Race -
This project focuses on designing a perception and navigation stack for the DJI Tello EDU quadcopter to autonomously
navigate through three stages of an obstacle course in an autonomous drone race. For more details about the race track and different stages in it please refer to [project 5](https://rbe549.github.io/rbe595/fall2023/proj/p5/). This project is a culmination of work done in [project 3](https://github.com/Chaitanya-01/P3-mini-drone-race) and [project 4](https://github.com/Chaitanya-01/P4-navigating-the-unknown) with an additional algorithm to deal with the dynamic window in the third stage of the race.


## Steps to run the code
- Install Numpy, OpenCV, djitellopy, Ultralytics, torch, cudatoolkit, matplotlib libraries before running the code.
- Install all the library dependencies mentioned [here](https://github.com/princeton-vl/RAFT)
- Turn the drone on and connect to it.
- To run the main code run the `Wrapper.py` file after installing all dependancies. This will save the final output folders in `Code` folder itself.
- In Code folder:
  ```bash
  python3 Wrapper.py --model=RAFT/models/raft-sintel.pth
  ```
- In the `Code` folder we have the corresponding model weights for phase 1 and phase 2 in `YOLO Model` folder and `RAFT` folders.

## Report
For detailed description see the report [here](Report.pdf).

## Collaborators
Chaitanya Sriram Gaddipati - cgaddipati@wpi.edu

Shiva Surya Lolla - slolla@wpi.edu

Ankit Talele - amtalele@wpi.edu


  
