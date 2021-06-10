# Identifying the pattern colleration using TensorFlow

<p align = "center">
  <img src = "https://raw.githubusercontent.com/hafiz-kamilin/research_tfPatternColleration/master/source/result.png" width = "992" height = "190"/>
</p>

A simple demonstration of using a machine learning to observe the pattern colleration between the input and output. The result shows the machine learning can predict the output based on the input given as long there are corellation between the input/output.

The model type used in the source code is a multi-task learning with hard parameter sharing.

## Test run

1. Assuming Python 3 programming environment already configured by the user; execute `pip install tensorflow pandas numpy` to install the required dependencies (TensorFlow version must be version 2.1.0 above for compatibility).
2. cd the console to the `source` directory and execute `python multiTaskLearning.py`.
