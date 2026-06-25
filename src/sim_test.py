import numpy as np
import matplotlib.pyplot as plt
import sys
import pinocchio as pin

import diffadmm.build.diffadmm # terrible. just add to path
"""
Tests forward dynamics to be used in the AGHF
"""

model = pin.buildModelFromUrdf("./data/urdf/kinova3_1arm.urdf")
data = model.createData()

q = pin.neutral(model) 
pin.computeJointJacobians(model, data, q)

for i in range(1, model.njoints):    
    J = pin.getJointJacobian(
        model, 
        data, 
        i, 
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    
    print(f"{model.names[i]} {J}")
