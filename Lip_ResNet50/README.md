------------------------
What does this model do?
------------------------
This model is for tracking human mouth lip shapes in camera images and video observing the lips, as viewed from in front of the mouth.

8 points are tracked:
 - Left / Right oral commissures
 - Upper and lower inner vermillion lip borders:
   - Symmetrically-middle point (laterally) on tubercle
   - Midpoints on tubercle between middle point and left/right commissures

In other words, this tracks the visible inner borders of the lips: The left/right corners of the mouth and 3 evenly spaced points along the flesh of the lips for each of the upper and lower lips.

As the flesh of the lips moves, the model is trained to track the movement. This model is designed to track horizontal movement of the lip anatomy in addition to vertical movement, so that asymmetrical lip movements are correctly tracked.

The model gives the most accurate results when the lips are centred in the image, with the camera oriented such that the nose is directly above the lips, and positioned/zoomed such that the distance between the commissures is approximately half the width of the image.


----------
Statistics
----------
This model has a Mean Square Distance to human-labelled ground-truth of:
 - 19.0 mm

Compared to the other models, the speed at which it can analyse data is:
 - Slow


----------------------------------------------------------
How does AAA automatically create splines from the points?
----------------------------------------------------------
AAA can create the following splines are from the tracked points:
 - "DLC_LipUpper": A contiguous spline of the upper lip points, including both commissures.
 - "DLC_LipLower": A contiguous spline of the lower lip points, including both commissures.

Note that both splines share the same left/right commissure points.


----------------------------------------------------------
How do I modify the way AAA interprets the tracked points?
----------------------------------------------------------
In this folder is a file called "AAAmodel" which you can edit using any text editor. It is a simple settings file containing multiple named parameters that define the behaviour of AAA in interpreting the points returned by DeepLabCut.
The file contains comments explaining what each parameter affects. You can change the parameters to suit your needs, and save the file. If you do not obey the format used in the "AAAmodel" file, it will likely not work and might crash AAA.