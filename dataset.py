from roboflow import Roboflow

rf = Roboflow(api_key="3Wjj61N5lBpUOxKFw62c")

project = rf.workspace("toys").project("toydetector")

dataset = project.version(1).download("yolov8")
