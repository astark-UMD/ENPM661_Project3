# ENPM661_Project3

- Aiden Stark (UID : 113907074)
- Masum Thakkar (UID : 121229076)
- Indraneel Mulakaloori (UID : 121377715)



**Libraries / Dependencies Used**:
- numpy
- cv2
- heapq
- copy
- time

**Video Output Explanation**:
- The video shows the node exploration and the generation of the optimal path. 
- The gray lines show the node exploration. 
The magenta circle is where the program starts and the cyan circle is the end goal. Gray lines extend from both directions
at the beginning because the program starts as bidirectional. 
- By performing a bidirectional search, we can ensure that the
final solution is exact regarding position and orientation. 
- However, the backward search of the bidirectional search is 
limited to 100 iterations. This ensures the free space is not excessively double searched by the forward and backward
searches, which increases computation time. Instead, it generates a small amount of path for the forward search to find,
overlap with, and connect to when finding a path that gets an exact solution. 

- For larger step sizes, the path may have small
'kinks' or corners in it. Since the path returns an exact solution, this is the most optimal path, but it requires some extra 
corners due to the large step size. 

**Color Key**:
- Black = Unexplored Free Space
- Dark Green = Explored Free Space
- Red = Obstacle Space
- Orange = Clearance Space Around Obstacles and Walls
- Magenta = Start Point
- Cyan = End Point
- White = Final Path

**How To Operate Program**:
1. Run the program via the terminal.
2. The terminal will prompt the user for the robot's radius. **Suggested values: 1-5**
    Excessively large values may consume all free space.
3. The terminal will prompt the user for the desired clearance radius around obstacles.
    **Suggested values: 1-5**. Excessively large values may consume all free space.
4. The terminal will prompt the user for the step size. This is how large each movement is
    that the robot takes. Larger steps sizes will compute faster, but may be less optimal due to
    weird turns that are required to reach the solution exactly. Small step sizes will return smoother
    and more optimal paths, but require more computation time. **Suggested values: 1-10.**
5. The terminal will prompt the user for the start and end coordinates. 
    1) It will ask for the coordinates as integers, in units of mm, and with respect
    to the bottom left corner as the origin. **Please note that the box is 600 long
    in x and and 250 long in y**. 
    2) If the point given is invalid, it will reprompt the user. It will also ask
    the user for the start and end orientations for the robot. It will require that the orientations given
    are between 0-360 and are a multiple of 30 since the robot can only move in 30 degree increments. 
    **Suggested Test Point:** 
     ``
     start_x = 20, start_y = 20, start_theta = 180 
     end_x = 550, end_y = 220, end_theta = 270
    `` 
    3) This will provide a valid path that tests the entire map. Additionally, the path will start going left
    `(start_theta = 180)` and end with the robot travelling downwards `(end_theta = 270)`.
6. The program will then output **Planning Path...**. Wait until the program outputs **Program Finished**. It
    will also return the time taken for completion.
7. When **Program Finished** is output to the terminal a window should appear that shows the total
    region explored and the final path. If the user hits any key on the keyboard, the window will close.
    Additionally, a video file should be generated in the same
    directory that the python file is run. It will be named `A_star_Proj3_phase1_video.mp4`

**Expected Time to Execute Program**: 
The longest paths that span the whole map take `~3 minutes`. 

In the zip file attached to our submission, we have provided three videos that demonstrate how
the solution differs based upon the step size given. Each video uses the same start and goal 
positions and orientations. 

Link To Code on GitHub:

