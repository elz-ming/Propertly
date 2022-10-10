import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import pickle

app = Flask("__name__")

input_dict = {
    'choice'  :None,
    'district':None,
    'state'   :None,
    'type'    :None,
    'details' :None,
    'bedroom' :None,
    'bathroom':None,
    'carpark' :None,
}

@app.route("/", methods=['POST'])
def webhook():
    data = request.get_json()
    intent = data["queryResult"]["intent"]["displayName"]

    if (intent == 'get-choice'):
        input_dict['choice'] = data["queryResult"]["parameters"]["choice"]
        if data["queryResult"]["parameters"]["boolean"] == "yes":
            input_dict['choice'] = 'buy'

        response = {
            'fulfillmentText':'Which state is the property located in?'
	    }
    if (intent == 'get-state'):
        input_dict['state'] = data["queryResult"]["parameters"]["state"]

        response = {
            'fulfillmentText':'Which district is the property located in?'
	    }

    if (intent == 'get-district'):
        input_dict['district'] = data["queryResult"]["parameters"]["district"]

        response = {
            'fulfillmentText':'What type of property is it? (If its a terrace, mention how many storeys it has.)'
	    }
    
    if (intent == 'get-type'):
        input_dict['type'] = data["queryResult"]["parameters"]["type"]

        response = {
            'fulfillmentText':'Any extra description of your property? (eg. studio, intermediate, corner lot, end lot, duplex, penthouse, dual key, triplex, soho...)'
	    }
    
    if (intent == 'get-detail'):
        input_dict['details'] = data["queryResult"]["parameters"]["detail"]

        if data["queryResult"]["parameters"]["boolean"] == 'no':
            input_dict['details'] = 'Default'

        response = {
            'fulfillmentText':'How many bedrooms does the property have?'
	    }
    
    if (intent == 'get-bedroom'):
        input_dict['bedroom'] = data["queryResult"]["parameters"]["bedroom"]

        response = {
            'fulfillmentText':'How many bathrooms does the property have?'
	    }

    if (intent == 'get-bathroom'):
        input_dict['bathroom'] = data["queryResult"]["parameters"]["bathroom"]

        response = {
            'fulfillmentText':'How many carparks does the property have?'
	    }

    if (intent == 'get-carpark'):
        input_dict['carpark'] = data["queryResult"]["parameters"]["carpark"]

        response = {
            'fulfillmentText':  'Is the following correct?\nState     : {}\nDistrict  : {}\nType      : {}\nDetails   : {}\nBedrooms  : {}\nBathrooms : {}\nCarparks  : {}'.format(input_dict['state'],input_dict['district'],input_dict['type'],input_dict['details'],input_dict['bedroom'],input_dict['bathroom'],input_dict['carpark'])
	    }
    
    if (intent == 'confirmation'):
        if data["queryResult"]["parameters"]["boolean"] == 'yes':
            num_arr = [[input_dict['bedroom'],input_dict['bathroom'],input_dict['carpark']]]
            cat_df = pd.DataFrame(data=input_dict, index=[0]).drop(['choice','bedroom','bathroom','carpark'],axis=1)
            if input_dict['choice'] == 'rent':
                rent_ohe = pickle.load(open('rent_ohe.pkl','rb'))
                cat_arr = rent_ohe.transform(cat_df).toarray()
                train_arr = np.concatenate((num_arr,cat_arr),axis=1)

                rent_model = pickle.load(open('rent_model.pkl','rb'))
                prediction = rent_model.predict(train_arr)[0]
                prediction = int(prediction)

                response = {
                    'fulfillmentText':"The property could be rented at {}.".format(prediction)
                }
            else:
                sale_ohe = pickle.load(open('sale_ohe.pkl','rb'))
                cat_arr = sale_ohe.transform(cat_df).toarray()
                train_arr = np.concatenate((num_arr,cat_arr),axis=1)

                sale_model = pickle.load(open('sale_model.pkl','rb'))
                prediction = sale_model.predict(train_arr)[0]
                prediction = int(prediction)

                response = {
                    'fulfillmentText':"The property could be bought/sold at {}.".format(prediction)
                }

    return jsonify(response)
    

if __name__ == '__main__':
    app.run(debug=True)