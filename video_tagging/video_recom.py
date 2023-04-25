import argparse

import io
from multiprocessing.sharedctypes import Value
from google.cloud import videointelligence
import urllib
import requests
import posixpath
import urllib 

from urllib import request

URL = "http://projectbodega.azurewebsites.net/bodega-api/product/?format=json&page=2"
r = requests.get(url = URL, params = None)
image_url = r.json()["results"][1]["product_image1"]
print(image_url)

response1 = request.urlretrieve(image_url, "media01")

path = "/Users/rohittiwari/firebase_test/{}".format("media01")

'''class Data:
    def __format__(self, spec):
        return '/Users/rohittiwari/firebase_test/' + spec


x = Data()
print(format(x, 'media'))
'''


"""Detect labels given a file path."""
video_client = videointelligence.VideoIntelligenceServiceClient()
features = [videointelligence.Feature.LABEL_DETECTION]




with io.open(path, "rb") as movie:
    input_content = movie.read()

operation = video_client.annotate_video(
    request={"features": features, "input_content": input_content}
)
print("\n====BODEGA COLD START=====")


result = operation.result(timeout=90)
print("\nFinished processing.")

aventador_list =[]

segment_labels = result.annotation_results[0].segment_label_annotations
for i, segment_label in enumerate(segment_labels):
    
    for category_entity in segment_label.category_entities:
        print("reached 1st step")

    for i, segment in enumerate(segment_label.segments):
        start_time = (
            segment.segment.start_time_offset.seconds
            + segment.segment.start_time_offset.microseconds / 1e6
        )
        end_time = (
            segment.segment.end_time_offset.seconds
            + segment.segment.end_time_offset.microseconds / 1e6
        )
        positions = "{}s to {}s".format(start_time, end_time)
        confidence = segment.confidence
        
        desc_conf= "{} : {:.2f}".format(segment_label.entity.description,confidence)
        json_object = (desc_conf)
        aventador_list.append(json_object)

print(aventador_list)


shot_labels = result.annotation_results[0].shot_label_annotations
for i, shot_label in enumerate(shot_labels):
    for category_entity in shot_label.category_entities:
        print("reached 2nd step")

    for i, shot in enumerate(shot_label.segments):
        start_time = (
            shot.segment.start_time_offset.seconds
            + shot.segment.start_time_offset.microseconds / 1e6
        )
        end_time = (
            shot.segment.end_time_offset.seconds
            + shot.segment.end_time_offset.microseconds / 1e6
        )
        positions = "{}s to {}s".format(start_time, end_time)
        confidence = shot.confidence
        
        desc_conf1= "{} : {:.2f}".format(shot_label.entity.description,confidence)
        json_object1 = (desc_conf1)
        aventador_list.append(json_object1)

print(aventador_list)


'''segment_labels = result.annotation_results[0].segment_label_annotations
for i, segment_label in enumerate(segment_labels):
    print("Video label description: {}".format(segment_label.entity.description))
    for category_entity in segment_label.category_entities:
        print(
            "\tLabel category description: {}".format(category_entity.description)
        )

    for i, segment in enumerate(segment_label.segments):
        start_time = (
            segment.segment.start_time_offset.seconds
            + segment.segment.start_time_offset.microseconds / 1e6
        )
        end_time = (
            segment.segment.end_time_offset.seconds
            + segment.segment.end_time_offset.microseconds / 1e6
        )
        positions = "{}s to {}s".format(start_time, end_time)
        confidence = segment.confidence
        print("\tSegment {}: {}".format(i, positions))
        print("\tConfidence: {}".format(confidence))
    print("\n")

shot_labels = result.annotation_results[0].shot_label_annotations
for i, shot_label in enumerate(shot_labels):
    print("Shot label description: {}".format(shot_label.entity.description))
    for category_entity in shot_label.category_entities:
        print(
            "\tLabel category description: {}".format(category_entity.description)
        )

    for i, shot in enumerate(shot_label.segments):
        start_time = (
            shot.segment.start_time_offset.seconds
            + shot.segment.start_time_offset.microseconds / 1e6
        )
        end_time = (
            shot.segment.end_time_offset.seconds
            + shot.segment.end_time_offset.microseconds / 1e6
        )
        positions = "{}s to {}s".format(start_time, end_time)
        confidence = shot.confidence
        print("\tSegment {}: {}".format(i, positions))
        print("\tConfidence: {}".format(confidence))
    print("\n")
'''