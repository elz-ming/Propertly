req = request.get_json(silent=True, force=True)
    res = req.get("queryResult")
    params = res.get("parameters")
    for key, value in input_dict.items():
        try:
            input_dict[key] = params.get(key)
        except:
            continue

    intent = res.get("intent").get('displayName')
    if (intent=='confirmation'):
        fulfillmentText = "Confirmation"
        res = {
            "fulfillmentText": fulfillmentText
        }
        r = make_response(res)
        r.headers['Content-Type'] = 'application/json'
        return r
    
    elif (intent=='prediction'):
        num_arr = [[input_dict['bedroom'],input_dict['bathroom'],input_dict['carpark']]]
        cat_df = pd.DataFrame(data=input_dict, index=[0]).drop(['choice','bedroom','bathroom','carpark'],axis=1)
        
        if input_dict['choice'] == 'rent' :
            rent_ohe = pickle.load(open('rent_ohe.pkl','rb'))
            cat_arr = rent_ohe.transform(cat_df).toarray()
            train_arr = np.concatenate((num_arr,cat_arr),axis=1)

            rent_model = pickle.load(open('rent_model.pkl','rb'))
            prediction = rent_model.predict(train_arr)

            fulfillmentText = "The property could be rented at {}.".format(prediction)
            res = {
            "fulfillmentText": fulfillmentText
            }
            r = make_response(res)
            r.headers['Content-Type'] = 'application/json'
            return r
        else:
            sale_ohe = pickle.load(open('sale_ohe.pkl','rb'))
            cat_arr = sale_ohe.transform(cat_df).toarray()
            train_arr = np.concatenate((num_arr,cat_arr),axis=1)

            sale_model = pickle.load(open('sale_model.pkl','rb'))
            prediction = sale_model.predict(train_arr)

            fulfillmentText = "The property could be sold/bought at {}.".format(prediction)
            res = {
            "fulfillmentText": fulfillmentText
            }
            r = make_response(res)
            r.headers['Content-Type'] = 'application/json'
            return r