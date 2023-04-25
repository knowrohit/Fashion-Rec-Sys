
from weakref import ref
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import requests 
from array import array
import os
from PIL import Image
import sys
import time
from firebase_admin import credentials
from firebase_admin import db
import firebase_admin
from firebase_admin import db
import json
from firebase import firebase
import jsonify
import numpy as np
 


cred = firebase_admin.credentials.Certificate('/Users/rohittiwari/Downloads/recomai-firebase-adminsdk-za4vc-69e478f2fb.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://recomai-default-rtdb.firebaseio.com/"
})
print("done")

subscription_key = "9ce34950235d4aba8422ed848bef5b52"
endpoint = "https://bodegacv.cognitiveservices.azure.com/"

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))


def recommend_svj(request):
    global computervision_client, results_list, cred


    URL = "https://projectbodega.azurewebsites.net/bodega-api/product/"
    r = requests.get(url = URL, params = None)
    image_url = r.json()["results"][0]["product_image1"]
    remote_image_url = image_url

    tags_result_remote = computervision_client.tag_image(remote_image_url )


    results_list= []

    if (len(tags_result_remote.tags) == 0):
        print("No tags detected.")
    else:
        for tag in tags_result_remote.tags:
            tags_final  = "'{}' : {:.2f}".format(tag.name, tag.confidence * 100)
            json_obj= (tags_final)
            results_list.append(json_obj)
        

    b= json.dumps(results_list) 

    if r.method == 'POST':

        URL_meta = "https://bdgdao.azurewebsites.net/bodega-api/product_metadata/"
        r = requests.post(url= URL_meta, params= None)
        request_time = time.time()
        
        data = r.get_json()
        print('finished getting request json')


        t1 = time.time()
        prediction_probs = results_list
        predicted_label = np.argmax(prediction_probs, axis=None)[0]
        t2 = time.time()
        prediction_runtime = t2 - t1
        print('prediction_probs', prediction_probs)
        print('predicted_label', predicted_label)
        print('prediction_runtime', prediction_runtime)
        

    return (jsonify({ 
                     'predicted_label': str(predicted_label), 
                     'request_time': str(request_time), 
                     'prediction_probs': str(prediction_probs), 
                     'uploaded_filename': image_url,
                     'similiar_product_id' : "have to make a function for it, a query to find similiar tags in realtime db amonsgt products" 
                     }))

#looping thru pages
'''for i in range(1, 7, 1):
     
    URL = "https://projectbodega.azurewebsites.net/bodega-api/product/?page={}".format(i)
    print(URL)'''

'''URL = "https://projectbodega.azurewebsites.net/bodega-api/product/"
r = requests.get(url = URL, params = None)
image_url = r.json()["results"][7]["product_image1"]
remote_image_url = image_url

tags_result_remote = computervision_client.tag_image(remote_image_url )


results_list= []

if (len(tags_result_remote.tags) == 0):
    print("No tags detected.")
else:
    for tag in tags_result_remote.tags:
        tags_final  = "'{}' : {:.2f}".format(tag.name, tag.confidence * 100)
        json_obj= (tags_final)
        results_list.append(json_obj)
        

b= json.dumps(results_list)'''
'''
ref = db.reference ("results")

ref.update ({

	'9/productHashkey' : results_list	
})'''

'''fb_app = firebase.FirebaseApplication("https://recomai-default-rtdb.firebaseio.com/", None)
result = fb_app.get('/results', "1/productHashkey" )
print (result)'''