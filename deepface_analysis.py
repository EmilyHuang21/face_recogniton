# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 00:09:50 2022

@author: Emily
"""

from deepface import DeepFace


#face analysis
face_analysis = DeepFace.analyze(img_path="dataset/testing/modi1.jpg",
                                    actions=['emotion','age','gender','race'])

print(face_analysis)

DeepFace.stream(model_name='VGG-Face')



face_analysis = DeepFace.analyze(DeepFace.stream(),actions=['emotion','age','gender','race'])

                                    
print(face_analysis)

from deepface import DeepFace
models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib']
DeepFace.stream(db_path = "images/samples", model_name = models[5],time_threshold=2, frame_threshold=1,source=1)

