
------------------------
What does this model do?
------------------------
This model is for tracking human oral anatomy (tongue surface, hyoid, short-tendon, and mandible) in ultrasound images and video.

14 points are tracked:
 - 11 points of tongue surface
 - 1 point: Hyoid
 - 1 point: Mandible
 - 1 point: Short-tendon

In other words, this tracks the visible inner borders of the lips: The left/right corners of the mouth and 3 evenly spaced points along the flesh of the lips for each of the upper and lower lips.

As the flesh of the lips moves, the model is trained to track the movement. This model is designed to track horizontal movement of the lip anatomy in addition to vertical movement, so that asymmetrical lip movements are correctly tracked.

The model gives the most accurate results when the lips are centred in the image, with the camera oriented such that the nose is directly above the lips, and positioned/zoomed such that the distance between the commissures is approximately half the width of the image.


----------
Statistics
----------
This model has a Mean Square Distance to human-labelled ground-truth of:
 - 0.93 mm

Compared to the other models, the speed at which it can analyse data is:
 - Fast


----------------------------------------------------------
How does AAA automatically create splines from the points?
----------------------------------------------------------
AAA can create the following splines are from the tracked points:
 - "DLC_Tongue": A single contiguous spline of the 11 tongue surface points.
 - "DLC_HyoidMandible": A 2-point spline consisting of Hyoid and Mandible points.
 - "DLC_ShorttendonMandible": A 2-point spline consisting of Short-tendon and Mandible points.

Note that "DLC_HyoidMandible" and "DLC_ShorttendonMandible" both share the same mandible point.


----------------------------------------------------------
How do I modify the way AAA interprets the tracked points?
----------------------------------------------------------
In this folder is a file called "AAAmodel" which you can edit using any text editor. It is a simple settings file containing multiple named parameters that define the behaviour of AAA in interpreting the points returned by DeepLabCut.
The file contains comments explaining what each parameter affects. You can change the parameters to suit your needs, and save the file. If you do not obey the format used in the "AAAmodel" file, it will likely not work and might crash AAA.