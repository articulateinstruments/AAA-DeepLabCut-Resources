
------------------------
What does this model do?
------------------------
This model is for tracking human oral anatomy (tongue surface, hyoid, short-tendon, and mandible) in mid-sagittal ultrasound images and video.

14 points are tracked:
 - 11 points of tongue surface
 - 1 point: Hyoid
 - 1 point: Mandible
 - 1 point: Short-tendon

The 11 tongue surface points cover the upper tongue surface from the furthest extent of the tongue tip (the most rostral point) all the way back to the vallecula where the epiglottis meets the root of the tongue. The points near the tip of the tongue are more closely spaced than the other tongue surface points so as to reflect the greater flection of that region of the tongue. In other words, the spacings of each point between the 1st and 7th points (vallecula to tongue dorsum) are the same as each other, and then the spacings between the points from the 7th to 11th (dorsum to tip) are the same as each other, but twice as densely packed.

As viewed in mid-sagittal ultrasound:
- The "Hyoid" point tracks the centre of the hyoid bone.
- The "Mandible" point tracks the lower point of where the genio-hyoid attaches (inferior mental spine).
- The "Short-tendon" point tracks the superior tubercle (superior mental spine) of the mandible where the short-tendon attaches.

As the flesh of the tongue moves, the model is trained to track the movement. This model is designed to track horizontal movement of the tongue anatomy in addition to vertical movement, and place the points closer together where the tongue bunches and further apart where it stretches.

The model gives the most accurate results with exactly mid-sagittal ultrasound where both the mandible and hyoid are in view. The model is quite robust to different values of probe field-of-view from as low as 60 degrees or as high as 120 degrees, but works best with approximately 90 degrees. It's also robust to different probe depths and different sizes of human subject.

It is important to note that the model can be confused by user-interface elements drawn into the image such as text or scale bars, and these can disrupt the model's ability to track accurately. It also struggles with backgrounds to the ultrasound image which are not black. If you are using an ultrasound system which has unavoidable user-interface clutter that you are unable to disable, please contact us and send us a sample of your data: If we have time we can add it to the model's training so it can cope with screen clutter better.


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